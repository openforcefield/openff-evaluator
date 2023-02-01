import functools
import itertools
from enum import Enum
from typing import TYPE_CHECKING, List, Set, Tuple, Union

import numpy
import pandas
from pydantic import BaseModel, Field, conlist, validator
from typing_extensions import Literal

from openff.evaluator.datasets.curation.components import (
    CurationComponent,
    CurationComponentSchema,
)
from openff.evaluator.datasets.curation.components.filtering import (
    FilterByEnvironments,
    FilterByEnvironmentsSchema,
    FilterBySubstances,
    FilterBySubstancesSchema,
)
from openff.evaluator.datasets.utilities import (
    data_frame_to_substances,
    reorder_data_frame,
)
from openff.evaluator.utils.checkmol import ChemicalEnvironment
from openff.evaluator.utils.exceptions import MissingOptionalDependency

PropertyType = Tuple[str, int]

if TYPE_CHECKING:
    PositiveInt = int

    try:
        from openeye.oegraphsim import OEFingerPrint
    except ImportError:
        OEFingerPrint = None

else:
    from pydantic import PositiveInt


class State(BaseModel):
    temperature: float = Field(..., description="The temperature (K) of interest.")
    pressure: float = Field(..., description="The pressure (kPa) of interest.")

    mole_fractions: Tuple[float, ...] = Field(
        ..., description="The composition of interest."
    )


class TargetState(BaseModel):
    property_types: List[PropertyType] = Field(
        ..., description="The properties to select at the specified states."
    )
    states: List[State] = Field(
        ..., description="The states at which data points should be selected."
    )

    @classmethod
    @validator("property_types")
    def property_types_validator(cls, value):
        assert len(value) > 0
        n_components = value[0][1]

        assert all(x[1] == n_components for x in value)
        return value


class FingerPrintType(Enum):
    Tree = "Tree"
    MACCS166 = "MACCS166"


class SelectSubstancesSchema(CurationComponentSchema):
    type: Literal["SelectSubstances"] = "SelectSubstances"

    target_environments: conlist(ChemicalEnvironment, min_items=1) = Field(
        ...,
        description="The chemical environments which selected substances should "
        "contain.",
    )

    n_per_environment: PositiveInt = Field(
        ...,
        description="The number of substances to ideally select for each chemical "
        "environment of interest. In the case of pure substances, this will be the "
        "number of substances to choose per environment specified in "
        "`target_environments`. For binary mixtures, this will be the number of "
        "substances to select for each pair of environments constructed from the "
        "`target_environments` list and so on.",
    )

    substances_to_exclude: List[Tuple[str, ...]] = Field(
        default_factory=list,
        description="The substances to 1) filter from the available data before "
        "selecting substances and 2) to penalize similarity to. This field is "
        "mainly expected to be used when creating a test set which is distinct from "
        "a training set.",
    )

    finger_print_type: FingerPrintType = Field(
        FingerPrintType.Tree,
        description="The type of finger print to use in the distance metrics.",
    )

    per_property: bool = Field(
        ...,
        description="Whether the selection algorithm should be run once per "
        "property (e.g. select substances for pure densities, and then select "
        "substances for pure enthalpies of vaporization), or whether to run it "
        "once for the whole data set without consideration for if the selected "
        "substances have data points available for each property type in the set. "
        "This option should usually be set to false when the data set to select "
        "from strictly contains data points for each type of property for the same"
        "set of systems, and true otherwise.",
    )


class SelectSubstances(CurationComponent):
    """A component for selecting a specified number data points which were
    measured for systems containing a specified set of chemical functionalities.
    """

    @classmethod
    def _check_oe_available(cls):
        """Check if the `oechem` and `oegraphsim` modules are available for import.

        Raises
        -------
        MissingOptionalDependency
        """
        try:
            from openeye import oechem, oegraphsim
        except ImportError as e:
            raise MissingOptionalDependency(e.path, False)

        unlicensed_library = (
            "openeye.oechem"
            if not oechem.OEChemIsLicensed()
            else "openeye.oegraphsim"
            if not oegraphsim.OEGraphSimIsLicensed()
            else None
        )

        if unlicensed_library is not None:
            raise MissingOptionalDependency(unlicensed_library, True)

    @classmethod
    @functools.lru_cache(3000)
    def _compute_molecule_finger_print(
        cls, smiles: str, finger_print_type: FingerPrintType
    ) -> "OEFingerPrint":
        """Computes the finger print for a given molecule
        using the OpenEye toolkit.

        Parameters
        ----------
        smiles
            The smiles pattern to generate a finger print for.
        finger_print_type
            The type of finger print to generate.

        Returns
        -------
            The generated finger print.
        """
        from openeye.oegraphsim import (
            OEFingerPrint,
            OEFPType_MACCS166,
            OEFPType_Tree,
            OEMakeFP,
        )
        from openff.toolkit.topology import Molecule

        oe_molecule = Molecule.from_smiles(smiles).to_openeye()

        if finger_print_type == FingerPrintType.Tree:
            oe_finger_print_type = OEFPType_Tree
        elif finger_print_type == FingerPrintType.MACCS166:
            oe_finger_print_type = OEFPType_MACCS166
        else:
            raise NotImplementedError()

        finger_print = OEFingerPrint()
        OEMakeFP(finger_print, oe_molecule, oe_finger_print_type)

        return finger_print

    @classmethod
    def _compute_mixture_finger_print(
        cls, mixture: Tuple[str, ...], finger_print_type: FingerPrintType
    ) -> Tuple["OEFingerPrint", ...]:
        """Computes the finger print of a mixture of molecules as defined
        by their smiles patterns.

        Parameters
        ----------
        mixture
            The smiles patterns of the molecules in the mixture.
        finger_print_type
            The type of finger print to generate.

        Returns
        -------
        tuple of OEFingerPrint
            The finger print of each molecule in the mixture.
        """

        mixture_finger_print = tuple(
            cls._compute_molecule_finger_print(x, finger_print_type) for x in mixture
        )

        return mixture_finger_print

    @classmethod
    @functools.lru_cache(3000)
    def _compute_distance(
        cls,
        mixture_a: Tuple[str, ...],
        mixture_b: Tuple[str, ...],
        finger_print_type: FingerPrintType,
    ):
        """Computes the 'distance' between two mixtures based on
        their finger prints.

        The distance is defined as the minimum of

        - the OETanimoto distance between component a of mixture a and
          component a of mixture b + the OETanimoto distance between
          component b of mixture a and component b of mixture b

        and

        - the OETanimoto distance between component b of mixture a and
          component a of mixture b + the OETanimoto distance between
          component a of mixture a and component b of mixture b

        Parameters
        ----------
        mixture_a
            The smiles patterns of the molecules in mixture a.
        mixture_b
            The smiles patterns of the molecules in mixture b.
        finger_print_type
            The type of finger print to base the distance metric
            on.

        Returns
        -------
        float
            The distance between the mixtures
        """
        from openeye.oegraphsim import OETanimoto

        if sorted(mixture_a) == sorted(mixture_b):
            return 0.0

        finger_print_a = cls._compute_mixture_finger_print(mixture_a, finger_print_type)
        finger_print_b = cls._compute_mixture_finger_print(mixture_b, finger_print_type)

        if len(mixture_a) == 1 and len(mixture_b) == 1:
            distance = 1.0 - OETanimoto(finger_print_a[0], finger_print_b[0])

        elif len(mixture_a) == 2 and len(mixture_b) == 2:
            distance = min(
                (1.0 - OETanimoto(finger_print_a[0], finger_print_b[0]))
                + (1.0 - OETanimoto(finger_print_a[1], finger_print_b[1])),
                (1.0 - OETanimoto(finger_print_a[1], finger_print_b[0]))
                + (1.0 - OETanimoto(finger_print_a[0], finger_print_b[1])),
            )

        else:
            raise NotImplementedError()

        return distance

    @classmethod
    def _compute_distance_with_set(
        cls,
        mixture,
        mixtures: List[Tuple[str, ...]],
        finger_print_type: FingerPrintType,
    ) -> float:
        """Computes the distances between a given mixture and a set of other mixtures.

        This is computed as the sum of `_compute_distance(mixture, mixtures[i])`
        for all i in `mixtures`.

        Parameters
        ----------
        mixture
            The mixture to compute the distances from.
        mixtures
            The set of mixtures to compare with `mixture`.
        finger_print_type: OEFPTypeBase
            The type of finger print to base the distance metric
            on.

        Returns
        -------
            The calculated distance.
        """

        distance = sum(
            cls._compute_distance(mixture, x, finger_print_type) for x in mixtures
        )

        return distance

    @classmethod
    def _select_substances(
        cls,
        data_frame: pandas.DataFrame,
        n_substances: int,
        previously_chosen: List[Tuple[str, ...]],
        finger_print_type: FingerPrintType,
    ) -> List[Tuple[str, ...]]:
        # Store the substances which can be selected, and those which
        # have already been selected.
        open_list = [*data_frame_to_substances(data_frame)]
        closed_list = []

        # Determine the maximum number of substances which can be selected.
        max_n_possible = min(len(open_list), n_substances)

        while len(open_list) > 0 and len(closed_list) < max_n_possible:

            def distance_metric(mixture):
                return cls._compute_distance_with_set(
                    mixture, [*previously_chosen, *closed_list], finger_print_type
                )

            least_similar = sorted(open_list, key=distance_metric, reverse=True)[0]

            open_list.remove(least_similar)
            closed_list.append(least_similar)

        return closed_list

    @classmethod
    def _apply_to_data_frame(cls, data_frame, schema, n_processes):
        """Applies the selection algorithm to a specified data frame.

        Parameters
        ----------
        data_frame
            The data frame to apply the algorithm to.
        schema
            This component schema.
        n_processes
            The number of processes available to the component.
        """

        selected_substances = []

        min_n_components = data_frame["N Components"].min()
        max_n_components = data_frame["N Components"].max()

        # Perform the selection one for each size of substance (e.g. once
        # for pure, once for binary etc.)
        for n_components in range(min_n_components, max_n_components + 1):
            component_data = data_frame[data_frame["N Components"] == n_components]

            if len(component_data) == 0:
                continue

            # Define all permutations of the target environments.
            if n_components == 1:
                chemical_environments = [(x,) for x in schema.target_environments]

            elif n_components == 2:
                chemical_environments = [
                    *[(x, x) for x in schema.target_environments],
                    *itertools.combinations(schema.target_environments, r=2),
                ]

            else:
                raise NotImplementedError()

            # Keep a track of the selected substances
            selected_n_substances: Set[Tuple[str, ...]] = set()

            for chemical_environment in chemical_environments:
                # Filter out any environments not currently being considered.
                environment_filter = FilterByEnvironmentsSchema(
                    per_component_environments={
                        n_components: [[x] for x in chemical_environment]
                    }
                )

                environment_data = FilterByEnvironments.apply(
                    component_data, environment_filter, n_processes
                )

                if len(environment_data) == 0:
                    continue

                # Define the substances which the newly selected substance
                # should be unique to.
                substances_to_penalize = {
                    *selected_n_substances,
                    *[
                        x
                        for x in schema.substances_to_exclude
                        if len(x) == n_components
                    ],
                }

                environment_selected_substances = cls._select_substances(
                    environment_data,
                    schema.n_per_environment,
                    [*substances_to_penalize],
                    schema.finger_print_type,
                )
                selected_n_substances.update(environment_selected_substances)

                # Remove the newly selected substances from the pool to select from.
                component_data = FilterBySubstances.apply(
                    component_data,
                    FilterBySubstancesSchema(
                        substances_to_exclude=environment_selected_substances
                    ),
                    n_processes=n_processes,
                )

            selected_substances.extend(selected_n_substances)

        # Filter the data frame to retain only the selected substances.
        data_frame = FilterBySubstances.apply(
            data_frame,
            FilterBySubstancesSchema(substances_to_include=selected_substances),
            n_processes=n_processes,
        )

        return data_frame

    @classmethod
    def _apply(
        cls, data_frame: pandas.DataFrame, schema: SelectSubstancesSchema, n_processes
    ) -> pandas.DataFrame:
        # Make sure OpenEye is available for computing the finger prints.
        cls._check_oe_available()

        # Filter out any substances which should be excluded
        data_frame = FilterBySubstances.apply(
            data_frame,
            FilterBySubstancesSchema(
                substances_to_exclude=schema.substances_to_exclude
            ),
            n_processes=n_processes,
        )

        max_n_components = data_frame["N Components"].max()

        if max_n_components > 2:
            raise NotImplementedError()

        if schema.per_property:
            # Partition the data frame into ones which only contain a
            # single property type.
            property_headers = [
                header for header in data_frame if header.find(" Value ") >= 0
            ]

            data_frames_to_filter = [
                data_frame[data_frame[header].notna()] for header in property_headers
            ]

        else:
            data_frames_to_filter = [data_frame]

        filtered_data_frames = [
            cls._apply_to_data_frame(filtered_data_frame, schema, n_processes)
            for filtered_data_frame in data_frames_to_filter
        ]

        if len(filtered_data_frames) == 1:
            filtered_data_frame = filtered_data_frames[0]
        else:
            filtered_data_frame = pandas.concat(
                filtered_data_frames, ignore_index=True, sort=False
            )

        return filtered_data_frame


class SelectDataPointsSchema(CurationComponentSchema):
    type: Literal["SelectDataPoints"] = "SelectDataPoints"

    target_states: List[TargetState] = Field(
        ...,
        description="A list of the target states for which we would ideally include "
        "data points for (e.g. density data points measured at ambient conditions, or "
        "for density AND enthalpy of mixing measurements made for systems with a "
        "roughly 50:50 composition).",
    )


class SelectDataPoints(CurationComponent):
    """A component for selecting a set of data points which are
    measured as close as possible to a particular set of states.

    The points will be chosen so as to try and maximise the number of
    properties measured at the same condition (e.g. ideally we would
    have a data point for each property at T=298.15 and p=1atm) as this
    will maximise the chances that we can extract all properties from a
    single simulation.
    """

    @classmethod
    def _property_header(cls, data_frame, property_type):
        for column in data_frame:
            if column.find(f"{property_type} Value") < 0:
                continue

            return column

        return None

    @classmethod
    def _distances_to_state(cls, data_frame: pandas.DataFrame, state_point: State):
        distance_sqr = (
            data_frame["Temperature (K)"] - state_point.temperature
        ) ** 2 + (
            data_frame["Pressure (kPa)"] / 10.0 - state_point.pressure / 10.0
        ) ** 2

        for component_index in range(len(state_point.mole_fractions)):
            distance_sqr += (
                data_frame[f"Mole Fraction {component_index + 1}"]
                - state_point.mole_fractions[component_index]
            ) ** 2

        return distance_sqr

    @classmethod
    def _distances_to_cluster(
        cls, data_frame: pandas.DataFrame, target_state: TargetState
    ):
        distances_sqr = pandas.DataFrame()

        for index, state_point in enumerate(target_state.states):
            distances_sqr[index] = cls._distances_to_state(data_frame, state_point)

        return distances_sqr

    @classmethod
    def _select_substance_data_points(cls, original_data_frame, target_state):
        n_components = target_state.property_types[0][1]

        data_frame = original_data_frame[
            original_data_frame["N Components"] == n_components
        ].copy()
        data_frame["Property Type"] = ""

        property_types = [x[0] for x in target_state.property_types]

        for property_type in property_types:
            property_header = cls._property_header(data_frame, property_type)

            if not property_header:
                continue

            data_frame.loc[
                data_frame[property_header].notna(), "Property Type"
            ] = property_type

        data_frame["Temperature (K)"] = data_frame["Temperature (K)"].round(2)
        data_frame["Pressure (kPa)"] = data_frame["Pressure (kPa)"].round(1)

        for index in range(n_components):
            data_frame[f"Mole Fraction {index + 1}"] = data_frame[
                f"Mole Fraction {index + 1}"
            ].round(3)

        # Compute the distance to each cluster
        distances = cls._distances_to_cluster(data_frame, target_state)
        data_frame["Cluster"] = distances.idxmin(axis=1)

        cluster_headers = [
            "Temperature (K)",
            "Pressure (kPa)",
            *[f"Mole Fraction {index + 1}" for index in range(n_components)],
        ]

        # Compute how may data points are present for each state in the different
        # clusters.
        grouped_data = data_frame.groupby(
            by=[*cluster_headers, "Cluster"],
            as_index=False,
        ).agg({"Property Type": pandas.Series.nunique})

        selected_data = [False] * len(data_frame)

        for cluster_index in range(len(target_state.states)):
            # Calculate the distance between each clustered state and
            # the center of the cluster (i.e the clustered state).
            cluster_data = grouped_data[grouped_data["Cluster"] == cluster_index]
            cluster_data["Distance"] = cls._distances_to_state(
                cluster_data, target_state.states[cluster_index]
            )

            if len(cluster_data) == 0:
                continue

            open_list = [x[0] for x in target_state.property_types]

            while len(open_list) > 0 and len(cluster_data) > 0:
                # Find the clustered state which is closest to the center of
                # the cluster. Points measured at this state will becomes the
                # candidates to be selected.
                sorted_cluster_data = cluster_data.sort_values(
                    by=["Property Type", "Distance"], ascending=[False, True]
                )

                closest_index = sorted_cluster_data.index[0]

                # Find the data points which were measured at the clustered state.
                select_data = data_frame["Property Type"].isin(open_list)

                for cluster_header in cluster_headers:
                    select_data = select_data & numpy.isclose(
                        data_frame[cluster_header],
                        sorted_cluster_data.loc[closest_index, cluster_header],
                    )

                selected_property_types = data_frame[select_data][
                    "Property Type"
                ].unique()

                # Make sure to select a single data point for each type of property.
                for selected_property_type in selected_property_types:
                    selected_property_data = original_data_frame[
                        select_data
                        & (data_frame["Property Type"] == selected_property_type)
                    ]

                    if len(selected_property_data) <= 1:
                        continue

                    # Multiple data points were measured for this property type
                    # at the clustered state. We sort these multiple data points
                    # by their distance to the target state and select the closest.
                    # This is not guaranteed to be optimal but should be an ok
                    # approximation in most cases.
                    selected_data_distances = cls._distances_to_state(
                        selected_property_data, target_state.states[cluster_index]
                    )

                    sorted_data_distances = selected_data_distances.sort_values(
                        ascending=True
                    )

                    select_data[sorted_data_distances.index] = False
                    select_data[sorted_data_distances.index[0]] = True

                selected_data = selected_data | select_data

                for property_type in data_frame[select_data]["Property Type"].unique():
                    open_list.remove(property_type)

                cluster_data = cluster_data.drop(closest_index)

        if len(selected_data) == 0:
            return pandas.DataFrame()

        return original_data_frame[selected_data]

    @classmethod
    def _apply(
        cls, data_frame: pandas.DataFrame, schema: SelectDataPointsSchema, n_processes
    ) -> pandas.DataFrame:
        max_n_substances = data_frame["N Components"].max()
        component_headers = [f"Component {i + 1}" for i in range(max_n_substances)]

        # Re-order the data frame so that the components are alphabetically sorted.
        # This will make it easier to find unique substances.
        ordered_data_frame = reorder_data_frame(data_frame)

        # Find all of the unique substances in the data frame.
        unique_substances = ordered_data_frame[component_headers].drop_duplicates()

        selected_data = []

        # Start to choose the state points for each unique substance.
        for _, unique_substance in unique_substances.iterrows():
            substance_data_frame = ordered_data_frame

            for index, component in enumerate(unique_substance[component_headers]):
                if pandas.isnull(component):
                    substance_data_frame = substance_data_frame[
                        substance_data_frame[component_headers[index]].isna()
                    ]

                else:
                    substance_data_frame = substance_data_frame[
                        substance_data_frame[component_headers[index]] == component
                    ]

            for target_state in schema.target_states:
                substance_selected_data = cls._select_substance_data_points(
                    substance_data_frame, target_state
                )

                if len(substance_selected_data) == 0:
                    continue

                selected_data.append(substance_selected_data)

        selected_data = pandas.concat(selected_data, ignore_index=True, sort=False)
        return selected_data


SelectionComponentSchema = Union[SelectSubstancesSchema, SelectDataPointsSchema]
