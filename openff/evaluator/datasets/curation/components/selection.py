import collections
import functools
import itertools
import logging
from enum import Enum
from typing import TYPE_CHECKING, List, Set, Tuple, Union

import numpy
import pandas
from pydantic import BaseModel, Field, PositiveInt, validator
from typing_extensions import Literal

logger = logging.getLogger(__name__)

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
from openff.evaluator.utils.checkmol import ChemicalEnvironment, analyse_functional_groups
from openff.evaluator.utils.exceptions import MissingOptionalDependency

PropertyType = Tuple[str, int]

if TYPE_CHECKING:
    try:
        from openeye.oegraphsim import OEFingerPrint
    except ImportError:
        OEFingerPrint = None


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

    target_environments: list[ChemicalEnvironment] = Field(
        ...,
        description="The chemical environments which selected substances should "
        "contain.",
        min_length=1,
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
            else "openeye.oegraphsim" if not oegraphsim.OEGraphSimIsLicensed() else None
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

            data_frame.loc[data_frame[property_header].notna(), "Property Type"] = (
                property_type
            )

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


class SelectNumRepresentationSchema(CurationComponentSchema):
    type: Literal["SelectNumRepresentation"] = "SelectNumRepresentation"

    minimum_representation: int = Field(
        default=1,
        description="The minimum number of times a component or ssubstance should be represented "
        "in the data set. If a component is represented less than this number of times, it will be "
        "removed from the data set.",
    )
    maximum_representation: int = Field(
        default=-1,
        description="The maximum number of times a component or substance should be represented "
        "in the data set. If a component is represented more than this number of times, it will be "
        "removed from the data set. If this value is negative, no maximum is applied.",
    )

    per_component: bool = Field(
        default=False,
        description="Whether the selection should be applied per component (e.g. "
        "select components which are represented at least `minimum_representation` times) "
        "or per substance (e.g. select mixtures which are represented at least "
        "`minimum_representation` times). Note that the proportion of each component in the mixture "
        "is not considered in this selection, so a mixture of 2 components with 1:1 mole fraction "
        "will be considered the same as a mixture of 2 components with 1:2 mole fraction.",
    )


class SelectNumRepresentation(CurationComponent):
    """
    A component for selecting components or substances which are represented
    at least a specified number of times in the data set.
    """

    @classmethod
    def _apply(
        cls,
        data_frame: pandas.DataFrame,
        schema: SelectNumRepresentationSchema,
        n_processes,
    ) -> pandas.DataFrame:
        import math

        if schema.per_component:

            def update_counter_per_row(row, counter):
                n_components = row["N Components"]
                for index in range(n_components):
                    smiles = row[f"Component {index + 1}"]
                    counter[smiles] += 1

        else:

            def update_counter_per_row(row, counter):
                smiles = tuple(
                    sorted(
                        [
                            row[f"Component {index + 1}"]
                            for index in range(row["N Components"])
                        ]
                    )
                )
                counter[smiles] += 1

        max_repr = schema.maximum_representation
        if schema.maximum_representation < 0:
            max_repr = math.inf

        counter = collections.Counter()
        for _, row in data_frame.iterrows():
            update_counter_per_row(row, counter)

        allowed_smiles = set()
        for smiles, count in counter.items():
            if count >= schema.minimum_representation and count <= max_repr:
                allowed_smiles.add(smiles)

        if schema.per_component:

            def filter_function(row):
                n_components = row["N Components"]
                for index in range(n_components):
                    smiles = row[f"Component {index + 1}"]
                    if smiles not in allowed_smiles:
                        return False
                return True

        else:

            def filter_function(row):
                smiles = tuple(
                    sorted(
                        [
                            row[f"Component {index + 1}"]
                            for index in range(row["N Components"])
                        ]
                    )
                )
                return smiles in allowed_smiles

        filtered_data = data_frame[data_frame.apply(filter_function, axis=1)]
        return filtered_data


class SelectStratifiedSplitSchema(CurationComponentSchema):
    type: Literal["SelectStratifiedSplit"] = "SelectStratifiedSplit"

    keep_fraction: float = Field(
        ...,
        gt=0.0,
        le=1.0,
        description="Fraction of unique substances to select globally. Let N be the "
        "number of unique substances and B = max(1, round(N * keep_fraction)). "
        "Substances are ranked by a priority score (sum of matched smiles, target "
        "environment, and property-type strata flags), and selected by descending "
        "priority tiers until B is reached.",
    )
    seed: int = Field(
        42,
        description="Random seed. Given the same seed, the selection is deterministic "
        "regardless of row order (pair keys depend only on canonical SMILES).",
    )
    smiles_strata: List[str] = Field(
        default_factory=list,
        description="SMILES of molecules whose presence as a component of a substance "
        "is used as an additional boolean stratum (one flag per entry).",
    )
    target_environment_strata: List[ChemicalEnvironment] = Field(
        default_factory=list,
        description="Chemical environments whose presence in a substance is used as "
        "additional boolean strata (one flag per entry). Requires the checkmol binary; "
        "if unavailable, all substances fall into the same unclassified bucket for "
        "this dimension.",
    )
    property_type_strata: List[str] = Field(
        default_factory=list,
        description="Property-type name prefixes (e.g. 'EnthalpyOfMixing') whose data "
        "availability per substance is used as an additional boolean stratum.",
    )
    strata_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Optional per-stratum score weights used during priority scoring. "
        "Keys may match entries in smiles_strata (raw or canonical SMILES), "
        "target_environment_strata (enum name/value/string), or property_type_strata. "
        "Missing keys default to weight 1.0.",
    )
    diversity_selection: bool = Field(
        True,
        description="If True, select the most diverse substances within each priority "
        "tier using MaxMin selection on RDKit Morgan fingerprints (radius=2, 1024 "
        "bits) instead of random sampling. Requires RDKit.",
    )
    seed_smiles_strata: bool = Field(
        True,
        description="When True and diversity_selection=True, for each smiles_strata "
        "entry, seed one shortest-length matching substance within the "
        "current priority tier for MaxMin selection. Improves coverage of each "
        "smiles_strata member when it appears in a selected tier. "
        "Set to False when: (1) the number of smiles_strata entries is large "
        "relative to the per-tier budget, since seeds then fill the entire "
        "budget and MaxMin never runs; (2) the strata members are structurally "
        "similar, since seeding biases MaxMin toward a cluster and reduces "
        "global diversity; or (3) the priority scoring already guarantees "
        "coverage of all strata members and unbiased diversity is preferred.",
    )
    property_balance: bool = Field(
        True,
        description="When True, after priority-tier selection, trim substances that "
        "exclusively have data for the most-represented property_type_strata entry "
        "until count_max <= max_property_ratio * count_second. Only substances with "
        "data for exactly one property type are candidates for removal. "
        "Counts are measured by property_balance_mode, and least-diverse "
        "(last-added) substances are removed first.",
    )
    property_balance_mode: Literal["substance", "row"] = Field(
        "substance",
        description="How property_balance counts representation for each property type. "
        "'substance' counts selected substances with at least one value for a property; "
        "'row' counts selected data rows with a non-null value for that property.",
    )
    property_balance_by_n_components: bool = Field(
        True,
        description="When True, apply property_balance separately within each "
        "substance size bucket (N Components = 1, 2, ...), rather than globally "
        "across all selected substances.",
    )
    max_property_ratio: float = Field(
        2.0,
        ge=1.0,
        description="Maximum allowed ratio between the most- and second-most-represented "
        "property_type_strata entry. Only used when property_balance=True and at least "
        "two property_type_strata entries have coverage.",
    )
    include_subset_substances: bool = Field(
        True,
        description="When True, after selecting canonical substance keys, also include "
        "rows whose canonical substance key is a subset of any selected key. "
        "This can preserve pure-component rows (e.g. pure densities) when a "
        "containing mixture was selected.",
    )


class SelectStratifiedSplit(CurationComponent):
    """Select a priority-tiered fraction of substances from a data set.

    Selection is performed at the **substance** level (order-independent canonical
    SMILES tuples), independent of state variables such as mole fractions,
    temperature, and pressure.

    Each substance is assigned boolean flags in three groups:

        - ``smiles_strata`` membership flags.
        - ``target_environment_strata`` membership flags.
        - ``property_type_strata`` availability flags.

    A priority score is computed as the total number of ``True`` flags across all
    groups. Substances are selected by descending priority score tiers until the
    global budget ``B = max(1, round(N * keep_fraction))`` is fulfilled.

    Within a priority tier, selection is either MaxMin diversity
    (``diversity_selection=True``, default) or random permutation.

    When ``seed_smiles_strata=True`` and ``diversity_selection=True``, one seed per
    active smiles stratum in the current tier is injected into MaxMin selection,
    improving coverage of requested ``smiles_strata`` members.

    Optional ``property_balance`` trimming is applied after selection; if trimming
    removes entries, the selection is refilled from the remaining candidates by the
    same priority-tier + diversity policy to preserve the target budget, then
    balanced again to avoid refill-induced ratio regressions.

    When ``include_subset_substances=True``, rows whose canonical substance key is a
    subset of any selected key are also retained. This is useful when selected
    mixtures should carry along related pure-component data.

    Notes
    -----
    Stratum assignment uses canonical SMILES internally; the input data frame is
    never mutated.
    """

    @classmethod
    def _canonicalize_smiles(cls, smiles: str) -> str:
        """Returns a canonical SMILES string for internal comparisons."""
        from rdkit import Chem

        molecule = Chem.MolFromSmiles(smiles)

        if molecule is None:
            return smiles

        return Chem.MolToSmiles(molecule)

    @classmethod
    def _canonicalize_substance(cls, substance: Tuple[str, ...]) -> Tuple[str, ...]:
        """Returns a canonical, order-independent key for a substance."""
        return tuple(sorted(cls._canonicalize_smiles(smiles) for smiles in substance))

    @classmethod
    def _substance_environments(
        cls, substance: Tuple[str, ...]
    ) -> set[ChemicalEnvironment]:
        """All checkmol environments present across all components."""
        pair_envs: set = set()

        for smi in cls._canonicalize_substance(substance):
            result = analyse_functional_groups(smi)

            if result is not None:
                pair_envs |= set(result.keys())

        return pair_envs

    @classmethod
    def _substance_morgan_fp(cls, substance: Tuple[str, ...]):
        """Order-independent Morgan FP for a substance built from component unions."""
        from rdkit import Chem, DataStructs
        from rdkit.Chem import rdFingerprintGenerator

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
        substance_fp = DataStructs.ExplicitBitVect(1024)

        for smiles in cls._canonicalize_substance(substance):
            molecule = Chem.MolFromSmiles(smiles)

            if molecule is None:
                continue

            molecule_fp = generator.GetFingerprint(molecule)

            if molecule_fp is not None:
                substance_fp |= molecule_fp

        return substance_fp

    @classmethod
    def _maxmin_select(
        cls,
        substances: List[Tuple[str, ...]],
        n_pick: int,
        rng: numpy.random.Generator,
        first_picks: List[int] = (),
    ) -> List[Tuple[str, ...]]:
        """MaxMin Tanimoto-diversity selection over Morgan fingerprints."""
        from rdkit.SimDivFilters import rdSimDivPickers

        n = len(substances)

        if n_pick >= n:
            return substances

        # If seeds already satisfy the budget, return them directly.
        first_picks = list(first_picks)[:n_pick]
        if len(first_picks) >= n_pick:
            return [substances[i] for i in first_picks]

        picker = rdSimDivPickers.MaxMinPicker()
        substance_fps = [cls._substance_morgan_fp(substance) for substance in substances]
        picker_seed = int(rng.integers(0, numpy.iinfo(numpy.int32).max))
        picked_indices = picker.LazyBitVectorPick(
            substance_fps, len(substance_fps), n_pick,
            firstPicks=first_picks, seed=picker_seed,
        )

        return [substances[index] for index in picked_indices]

    @classmethod
    def _seed_indices_for_tier(
        cls,
        tier_substances: List[Tuple[str, ...]],
        canon_smiles_strata: List[str],
    ) -> List[int]:
        """Build per-smiles seeds for a priority tier.

        For each requested smiles stratum, choose the shortest-length matching
        substance in the tier that contains that smiles as a component.

        In case of ties, use tuple ordering as a deterministic fallback.
        """
        seed_indices: List[int] = []
        seen_indices = set()

        for smiles in canon_smiles_strata:
            candidates = [i for i, substance in enumerate(tier_substances) if smiles in substance]

            if not candidates:
                continue

            best_index = min(
                candidates,
                key=lambda i: (
                    sum(len(component_smiles) for component_smiles in tier_substances[i]),
                    tier_substances[i],
                ),
            )

            if best_index in seen_indices:
                continue

            seen_indices.add(best_index)
            seed_indices.append(best_index)

        return seed_indices

    @classmethod
    def _select_from_priority_tiers(
        cls,
        candidates: List[Tuple[str, ...]],
        priority_scores: dict,
        n_take: int,
        rng: numpy.random.Generator,
        diversity_selection: bool,
        seed_smiles_strata: bool,
        canon_smiles_strata: List[str],
    ) -> List[Tuple[str, ...]]:
        """Select candidates by descending priority score and in-tier policy.

        This routine is a greedy tiered selector:

        1. Group candidates by ``priority_scores[substance]``.
        2. Iterate tiers from highest score to lowest.
        3. Fill the remaining budget from the current tier, then stop when the
                requested total ``n_take`` is reached.

        Consequences of this design:

        - It is greedy at the tier level. Lower-priority tiers are never considered
            until all higher-priority tiers are exhausted or clipped by budget.
        - It does not solve a global combinatorial optimum across tiers; optimization
            only happens within the partially used final tier.

        In-tier behavior:

        - If a tier fully fits in the remaining budget, all tier members are kept.
        - If the tier overflows the budget and ``diversity_selection=True``, choose
            ``remaining_budget`` members via RDKit MaxMin over substance fingerprints.
            When ``seed_smiles_strata=True``, first-pick seeds are injected for strata
            represented in that tier before MaxMin fills the rest.
        - If ``diversity_selection=False``, sample a random permutation within the
            tier using the provided RNG.

        Determinism:

        - Tier member order is sorted before selection.
        - Given fixed inputs and RNG seed, output is deterministic.
        """
        if n_take <= 0 or len(candidates) == 0:
            return []

        by_score = collections.defaultdict(list)
        for substance in candidates:
            by_score[priority_scores[substance]].append(substance)

        selected: List[Tuple[str, ...]] = []

        for score in sorted(by_score, reverse=True):
            tier = sorted(by_score[score])

            if len(selected) >= n_take:
                break

            remaining_budget = n_take - len(selected)

            if len(tier) <= remaining_budget:
                selected.extend(tier)
                continue

            if diversity_selection:
                tier_first_picks: List[int] = []

                if seed_smiles_strata and canon_smiles_strata:
                    tier_first_picks = cls._seed_indices_for_tier(
                        tier, canon_smiles_strata
                    )

                selected.extend(
                    cls._maxmin_select(
                        tier,
                        remaining_budget,
                        rng,
                        first_picks=tier_first_picks,
                    )
                )
            else:
                perm = rng.permutation(len(tier))
                selected.extend([tier[i] for i in perm[:remaining_budget]])

        return selected

    @classmethod
    def _property_balance_counts(
        cls,
        selected_substances: List[Tuple[str, ...]],
        prop_types: List[str],
        prop_available: dict,
        prop_row_counts: dict,
        balance_mode: str,
    ) -> dict:
        """Compute per-property representation counts for balancing.

        The returned counts drive ``_apply_property_balance`` and therefore control
        which property is considered over-represented.

        - ``balance_mode == "row"``:
          Count data rows with non-null values for each property across the currently
          selected substances. A substance contributes as many counts as it has rows
          for that property in ``prop_row_counts``.

        - any other mode (currently ``"substance"``):
          Count selected substances that have at least one value for each property.
          Each substance contributes at most one count per property using
          ``prop_available``.

        Returns
        -------
        dict
            Mapping ``property_type_strata`` entry -> representation count according
            to the chosen ``balance_mode``.
        """
        if balance_mode == "row":
            return {
                prop: sum(prop_row_counts[prop].get(substance, 0) for substance in selected_substances)
                for prop in prop_types
            }

        return {
            prop: sum(1 for substance in selected_substances if prop_available[prop].get(substance, False))
            for prop in prop_types
        }

    @classmethod
    def _apply_property_balance(
        cls,
        selected_substances: List[Tuple[str, ...]],
        prop_types: List[str],
        prop_available: dict,
        prop_row_counts: dict,
        balance_mode: str,
        max_property_ratio: float,
        by_n_components: bool,
    ) -> List[Tuple[str, ...]]:
        """Trim single-property substances until the property ratio threshold is met."""
        if len(prop_types) < 2:
            return selected_substances

        balanced = list(selected_substances)

        def _trim_bucket(bucket_n_components: int | None = None) -> None:
            while True:
                if bucket_n_components is None:
                    bucket = balanced
                else:
                    bucket = [s for s in balanced if len(s) == bucket_n_components]

                if len(bucket) == 0:
                    break

                prop_counts = cls._property_balance_counts(
                    bucket,
                    prop_types,
                    prop_available,
                    prop_row_counts,
                    balance_mode,
                )

                sorted_counts = sorted(prop_counts.values(), reverse=True)
                count_max, count_second = sorted_counts[0], sorted_counts[1]

                if count_second == 0 or count_max <= max_property_ratio * count_second:
                    break

                p_max = max(prop_counts, key=prop_counts.get)

                # Remove the last single-property substance contributing to p_max.
                trimmed = False
                for i in range(len(balanced) - 1, -1, -1):
                    substance = balanced[i]

                    if (
                        bucket_n_components is not None
                        and len(substance) != bucket_n_components
                    ):
                        continue

                    props_for_substance = [
                        prop
                        for prop in prop_types
                        if prop_available[prop].get(substance, False)
                    ]

                    if len(props_for_substance) == 1 and props_for_substance[0] == p_max:
                        balanced.pop(i)
                        trimmed = True
                        break

                if not trimmed:
                    break

        if by_n_components:
            for n_components in sorted({len(s) for s in balanced}):
                _trim_bucket(n_components)
        else:
            _trim_bucket()

        return balanced

    @classmethod
    def _apply(
        cls,
        data_frame: pandas.DataFrame,
        schema: SelectStratifiedSplitSchema,
        n_processes,
    ) -> pandas.DataFrame:
        rng = numpy.random.default_rng(schema.seed)

        # Sort for deterministic ordering across Python runs (set has no guaranteed order)
        raw_substances = sorted(data_frame_to_substances(data_frame))
        canonical_to_raw_substances = collections.defaultdict(list)

        for substance in raw_substances:
            canonical_to_raw_substances[cls._canonicalize_substance(substance)].append(
                substance
            )

        all_substances = sorted(canonical_to_raw_substances)

        if not all_substances:
            logger.debug("SelectStratifiedSplit: no substances found, returning input unchanged")
            return data_frame

        # n_target: number of substances to select based on keep_fraction, bounded by [1, N]
        n_target = max(1, round(len(all_substances) * schema.keep_fraction))
        n_target = min(n_target, len(all_substances))

        # Canonical SMILES strata: pre-canonicalize for consistent matching and scoring.
        canon_smiles_strata: List[str] = [
            cls._canonicalize_smiles(smiles) for smiles in schema.smiles_strata
        ]

        def _lookup_weight(*keys: str | None) -> float:
            for key in keys:
                if key is None:
                    continue
                if key in schema.strata_weights:
                    return float(schema.strata_weights[key])
            return 1.0

        smiles_weights = [
            _lookup_weight(raw_smiles, canonical_smiles)
            for raw_smiles, canonical_smiles in zip(
                schema.smiles_strata,
                canon_smiles_strata,
            )
        ]
        env_weights = [
            _lookup_weight(
                str(environment),
                getattr(environment, "name", None),
                str(getattr(environment, "value", "")) or None,
            )
            for environment in schema.target_environment_strata
        ]

        if schema.strata_weights:
            logger.debug(
                f"SelectStratifiedSplit: custom strata weights active keys={sorted(schema.strata_weights)}"
            )

        # Property-type strata: find value column per property type name prefix.
        # Use "<prop> Value" prefix to avoid spurious substring matches.
        prop_value_cols = {}
        for prop in schema.property_type_strata:
            cols = [c for c in data_frame.columns if c.startswith(f"{prop} Value")]
            if cols:
                prop_value_cols[prop] = cols[0]
        prop_weights = [
            _lookup_weight(property_type)
            for property_type in schema.property_type_strata
            if property_type in prop_value_cols
        ]

        # Per-substance property availability and row counts via groupby on reordered frame
        prop_available: dict = {}
        prop_row_counts: dict = {}
        if prop_value_cols:
            ordered = reorder_data_frame(data_frame)
            max_n = ordered["N Components"].max()
            comp_cols = [
                f"Component {i + 1}"
                for i in range(max_n)
                if f"Component {i + 1}" in ordered.columns
            ]
            # set a temporary "_subst" column for groupby
            ordered["_subst"] = ordered[comp_cols].apply(
                lambda r: cls._canonicalize_substance(
                    tuple(v for v in r if pandas.notna(v))
                ),
                axis=1,
            )
            for prop, col in prop_value_cols.items():
                prop_available[prop] = ordered.groupby("_subst")[col].any().to_dict()
                prop_row_counts[prop] = (
                    ordered[ordered[col].notna()].groupby("_subst").size().to_dict()
                )

        # Build per-substance flag tuples and priority scores.
        priority_scores: dict[Tuple[str, ...], int] = {}
        max_n_substance = max(len(s) for s in all_substances)
        for substance in all_substances:
            smiles_flags = tuple(smi in substance for smi in canon_smiles_strata)

            env_flags = ()
            if schema.target_environment_strata:
                substance_environments = cls._substance_environments(substance)
                env_flags = tuple(
                    environment in substance_environments
                    for environment in schema.target_environment_strata
                )

            prop_flags = [
                prop_available[prop].get(substance, False)
                for prop in schema.property_type_strata
                if prop in prop_available
            ]
            # extra prioritise substances with all property types available
            if all(prop_flags):
                prop_flags.append(2)

            # Inverse-size weighting to help smaller substances (e.g. pure) compete
            # with larger systems that naturally match more strata flags.
            size_priority_flag = 5 * (
                max_n_substance - len(substance)
            )

            smiles_score = sum(
                weight for flag, weight in zip(smiles_flags, smiles_weights) if flag
            )
            env_score = sum(
                weight for flag, weight in zip(env_flags, env_weights) if flag
            )
            prop_score = sum(
                weight for flag, weight in zip(prop_flags, prop_weights) if flag
            )
            if len(prop_flags) > len(prop_weights):
                # Add bonus from the all-property-coverage heuristic above.
                prop_score += sum(prop_flags[len(prop_weights):])

            priority_scores[substance] = (
                smiles_score + env_score + prop_score + size_priority_flag
            )

        def _fold_in_subset_substances(substances: List[Tuple[str, ...]]) -> List[Tuple[str, ...]]:
            """Expand selected substances by subset closure."""
            if not schema.include_subset_substances:
                return substances

            expanded_set = set(substances)
            seed_set = set(substances)

            for candidate_substance in all_substances:
                if any(
                    set(candidate_substance).issubset(set(selected_substance))
                    for selected_substance in seed_set
                ):
                    expanded_set.add(candidate_substance)

            return sorted(expanded_set)

        # greedily select from priority tiers until the target budget is reached
        selected_substances = cls._select_from_priority_tiers(
            candidates=all_substances,
            priority_scores=priority_scores,
            n_take=n_target,
            rng=rng,
            diversity_selection=schema.diversity_selection,
            seed_smiles_strata=schema.seed_smiles_strata,
            canon_smiles_strata=canon_smiles_strata,
        )
        logger.debug(
            f"SelectStratifiedSplit: selected {len(selected_substances)} substances after tier selection"
        )
        selected_substances = _fold_in_subset_substances(selected_substances)

        # Optional property balancing: trim single-property substances for the
        # most-over-represented property_type_strata entry.
        prop_types = [p for p in schema.property_type_strata if p in prop_available]

        def _property_counts_by_n_components(substances: List[Tuple[str, ...]]) -> dict:
            breakdown = {}
            for prop in prop_types:
                by_n_components = collections.defaultdict(int)
                for substance in substances:
                    n_components = len(substance)
                    if schema.property_balance_mode == "row":
                        by_n_components[n_components] += prop_row_counts[prop].get(substance, 0)
                    else:
                        by_n_components[n_components] += int(
                            prop_available[prop].get(substance, False)
                        )
                breakdown[prop] = dict(sorted(by_n_components.items()))
            return breakdown

        def _apply_and_log_property_balance(
            substances: List[Tuple[str, ...]],
            label: str,
        ) -> List[Tuple[str, ...]]:
            if not (schema.property_balance and len(prop_types) >= 2):
                return substances

            before = len(substances)
            balanced_substances = cls._apply_property_balance(
                substances,
                prop_types,
                prop_available,
                prop_row_counts,
                schema.property_balance_mode,
                schema.max_property_ratio,
                schema.property_balance_by_n_components,
            )
            prop_counts = cls._property_balance_counts(
                balanced_substances,
                prop_types,
                prop_available,
                prop_row_counts,
                schema.property_balance_mode,
            )
            logger.debug(
                f"SelectStratifiedSplit: {label} removed={before - len(balanced_substances)} "
                f"remaining={len(balanced_substances)} counts={prop_counts} "
                f"counts_by_n_components={_property_counts_by_n_components(balanced_substances)}"
            )
            return balanced_substances

        selected_substances = _apply_and_log_property_balance(
            selected_substances,
            "property balance pass",
        )

        # Preserve the target budget after optional balancing.
        if len(selected_substances) < n_target:
            selected_set = set(selected_substances)
            remaining_candidates = [
                substance for substance in all_substances if substance not in selected_set
            ]
            refill_needed = n_target - len(selected_substances)
            selected_substances.extend(
                cls._select_from_priority_tiers(
                    candidates=remaining_candidates,
                    priority_scores=priority_scores,
                    n_take=refill_needed,
                    rng=rng,
                    diversity_selection=schema.diversity_selection,
                    seed_smiles_strata=schema.seed_smiles_strata,
                    canon_smiles_strata=canon_smiles_strata,
                )
            )
            logger.debug(
                f"SelectStratifiedSplit: refill requested={refill_needed} "
                f"remaining_candidates={len(remaining_candidates)} "
                f"new_total={len(selected_substances)}"
            )
            selected_substances = _fold_in_subset_substances(selected_substances)

        # Re-apply balancing after refill to avoid ratio regressions.
        selected_substances = _apply_and_log_property_balance(
            selected_substances,
            "post-refill balance pass",
        )

        selected_set = set(selected_substances)

        selected_raw_substances = list(
            dict.fromkeys(
                raw_substance
                for canonical_substance in sorted(selected_set)
                for raw_substance in canonical_to_raw_substances[canonical_substance]
            )
        )

        # Use the internal implementation to avoid duplicate component-level
        # logging from nested FilterBySubstances.apply(...) calls.
        filtered_data = FilterBySubstances._apply(
            data_frame,
            FilterBySubstancesSchema(substances_to_include=selected_raw_substances),
            n_processes=n_processes,
        )
        logger.debug(
            f"SelectStratifiedSplit: final rows selected={len(filtered_data)} "
            f"removed={len(data_frame) - len(filtered_data)}"
        )
        return filtered_data


SelectionComponentSchema = Union[
    SelectSubstancesSchema,
    SelectDataPointsSchema,
    SelectNumRepresentationSchema,
    SelectStratifiedSplitSchema,
]
