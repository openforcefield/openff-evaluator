import functools
import itertools
import logging
from collections import defaultdict
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy
import pandas
from openff.units import unit
from pydantic import Field, root_validator, validator
from scipy.optimize import linear_sum_assignment
from typing_extensions import Literal

from openff.evaluator.datasets.curation.components import (
    CurationComponent,
    CurationComponentSchema,
)
from openff.evaluator.datasets.utilities import (
    data_frame_to_substances,
    reorder_data_frame,
)
from openff.evaluator.utils.checkmol import (
    ChemicalEnvironment,
    analyse_functional_groups,
)

if TYPE_CHECKING:

    conint = int
    confloat = float
    PositiveInt = int
    PositiveFloat = float

else:

    from pydantic import PositiveFloat, PositiveInt, confloat, conint, constr

logger = logging.getLogger(__name__)

ComponentEnvironments = List[List[ChemicalEnvironment]]
MoleFractionRange = Tuple[confloat(ge=0.0, le=1.0), confloat(ge=0.0, le=1.0)]


class FilterDuplicatesSchema(CurationComponentSchema):

    type: Literal["FilterDuplicates"] = "FilterDuplicates"

    temperature_precision: conint(ge=0) = Field(
        2,
        description="The number of decimal places to compare temperatures (K) to "
        "within.",
    )
    pressure_precision: conint(ge=0) = Field(
        3,
        description="The number of decimal places to compare pressures (kPa) to "
        "within.",
    )
    mole_fraction_precision: conint(ge=0) = Field(
        6,
        description="The number of decimal places to compare mole fractions to within.",
    )


class FilterDuplicates(CurationComponent):
    """A component to remove duplicate data points (within a specified precision)
    from a data set.
    """

    @classmethod
    def _apply(
        cls, data_frame: pandas.DataFrame, schema: FilterDuplicatesSchema, n_processes
    ) -> pandas.DataFrame:

        if len(data_frame) == 0:
            return data_frame

        data_frame = data_frame.copy()
        data_frame = reorder_data_frame(data_frame)

        minimum_n_components = data_frame["N Components"].min()
        maximum_n_components = data_frame["N Components"].max()

        filtered_data = []

        for n_components in range(minimum_n_components, maximum_n_components + 1):

            component_data = data_frame[
                data_frame["N Components"] == n_components
            ].copy()

            component_data["Temperature (K)"] = component_data["Temperature (K)"].round(
                schema.temperature_precision
            )
            component_data["Pressure (kPa)"] = component_data["Pressure (kPa)"].round(
                schema.pressure_precision
            )

            subset_columns = ["Temperature (K)", "Pressure (kPa)", "Phase"]

            for index in range(n_components):

                component_data[f"Mole Fraction {index + 1}"] = component_data[
                    f"Mole Fraction {index + 1}"
                ].round(schema.mole_fraction_precision)

                subset_columns.extend(
                    [
                        f"Component {index + 1}",
                        f"Role {index + 1}",
                        f"Mole Fraction {index + 1}",
                        f"Exact Amount {index + 1}",
                    ]
                )

            subset_columns = [x for x in subset_columns if x in component_data]
            value_headers = [x for x in component_data if x.find(" Value ") >= 0]

            sorted_filtered_data = []

            for value_header in value_headers:

                uncertainty_header = value_header.replace("Value", "Uncertainty")

                property_data = component_data[component_data[value_header].notna()]

                if uncertainty_header in component_data:
                    property_data = property_data.sort_values(
                        uncertainty_header, na_position="first"
                    )

                property_data = property_data.drop_duplicates(
                    subset=subset_columns, keep="last"
                )

                sorted_filtered_data.append(property_data)

            sorted_filtered_data = pandas.concat(
                sorted_filtered_data, ignore_index=True, sort=False
            )

            filtered_data.append(sorted_filtered_data)

        filtered_data = pandas.concat(filtered_data, ignore_index=True, sort=False)
        return filtered_data


class FilterByTemperatureSchema(CurationComponentSchema):

    type: Literal["FilterByTemperature"] = "FilterByTemperature"

    minimum_temperature: Optional[PositiveFloat] = Field(
        ...,
        description="Retain data points measured for temperatures above this value (K)",
    )
    maximum_temperature: Optional[PositiveFloat] = Field(
        ...,
        description="Retain data points measured for temperatures below this value (K)",
    )

    @root_validator
    def _min_max(cls, values):
        minimum_temperature = values.get("minimum_temperature")
        maximum_temperature = values.get("maximum_temperature")

        if minimum_temperature is not None and maximum_temperature is not None:
            assert maximum_temperature > minimum_temperature

        return values


class FilterByTemperature(CurationComponent):
    """A component which will filter out data points which were measured outside of a
    specified temperature range
    """

    @classmethod
    def _apply(
        cls,
        data_frame: pandas.DataFrame,
        schema: FilterByTemperatureSchema,
        n_processes,
    ) -> pandas.DataFrame:

        filtered_frame = data_frame

        if schema.minimum_temperature is not None:
            filtered_frame = filtered_frame[
                schema.minimum_temperature < filtered_frame["Temperature (K)"]
            ]

        if schema.maximum_temperature is not None:
            filtered_frame = filtered_frame[
                filtered_frame["Temperature (K)"] < schema.maximum_temperature
            ]

        return filtered_frame


class FilterByPressureSchema(CurationComponentSchema):

    type: Literal["FilterByPressure"] = "FilterByPressure"

    minimum_pressure: Optional[PositiveFloat] = Field(
        ...,
        description="Retain data points measured for pressures above this value (kPa)",
    )
    maximum_pressure: Optional[PositiveFloat] = Field(
        ...,
        description="Retain data points measured for pressures below this value (kPa)",
    )

    @root_validator
    def _min_max(cls, values):
        minimum_pressure = values.get("minimum_pressure")
        maximum_pressure = values.get("maximum_pressure")

        if minimum_pressure is not None and maximum_pressure is not None:
            assert maximum_pressure > minimum_pressure

        return values


class FilterByPressure(CurationComponent):
    """A component which will filter out data points which were measured outside of a
    specified pressure range.
    """

    @classmethod
    def _apply(
        cls, data_frame: pandas.DataFrame, schema: FilterByPressureSchema, n_processes
    ) -> pandas.DataFrame:

        filtered_frame = data_frame

        if schema.minimum_pressure is not None:
            filtered_frame = filtered_frame[
                schema.minimum_pressure < filtered_frame["Pressure (kPa)"]
            ]

        if schema.maximum_pressure is not None:
            filtered_frame = filtered_frame[
                filtered_frame["Pressure (kPa)"] < schema.maximum_pressure
            ]

        return filtered_frame


class FilterByMoleFractionSchema(CurationComponentSchema):

    type: Literal["FilterByMoleFraction"] = "FilterByMoleFraction"

    mole_fraction_ranges: Dict[conint(gt=1), List[List[MoleFractionRange]]] = Field(
        ...,
        description="The ranges of mole fractions to retain. Each key in the "
        "dictionary corresponds to a number of components in the system. Each value "
        "is a list of the allowed mole fraction ranges for all but one of the "
        "components, i.e for a binary system, the allowed mole fraction for only the "
        "first component must be specified.",
    )

    @validator("mole_fraction_ranges")
    def _validate_ranges(cls, value: Dict[int, List[List[MoleFractionRange]]]):

        for n_components, ranges in value.items():

            assert len(ranges) == n_components - 1

            assert all(
                mole_fraction_range[0] < mole_fraction_range[1]
                for component_ranges in ranges
                for mole_fraction_range in component_ranges
            )

        return value


class FilterByMoleFraction(CurationComponent):
    """A component which will filter out data points which were measured outside of a
    specified mole fraction range.
    """

    @classmethod
    def _apply(
        cls,
        data_frame: pandas.DataFrame,
        schema: FilterByMoleFractionSchema,
        n_processes,
    ) -> pandas.DataFrame:

        filtered_frame = data_frame

        full_query = ~filtered_frame["N Components"].isin(schema.mole_fraction_ranges)

        for n_components, ranges in schema.mole_fraction_ranges.items():

            # Build the query to apply
            n_component_query = filtered_frame["N Components"] == n_components

            for index, component_ranges in enumerate(ranges):

                component_query = None

                for mole_fraction_range in component_ranges:

                    fraction_query = (
                        filtered_frame[f"Mole Fraction {index + 1}"]
                        > mole_fraction_range[0]
                    ) & (
                        filtered_frame[f"Mole Fraction {index + 1}"]
                        < mole_fraction_range[1]
                    )

                    if component_query is None:
                        component_query = fraction_query
                    else:
                        component_query |= fraction_query

                n_component_query &= component_query

            full_query |= n_component_query

        filtered_frame = filtered_frame[full_query]
        return filtered_frame


class FilterByRacemicSchema(CurationComponentSchema):

    type: Literal["FilterByRacemic"] = "FilterByRacemic"


class FilterByRacemic(CurationComponent):
    """A component which will filter out data points which were measured for racemic
    mixtures.
    """

    @classmethod
    def _apply(
        cls,
        data_frame: pandas.DataFrame,
        schema: FilterByMoleFractionSchema,
        n_processes,
    ) -> pandas.DataFrame:

        # Begin building the query. All pure substances should be
        # retained by default.
        query = data_frame["N Components"] < 2

        for n_components in range(2, data_frame["N Components"].max() + 1):

            component_data = data_frame[data_frame["N Components"] == n_components]

            if len(component_data) == 0:
                continue

            component_combinations = itertools.combinations(range(n_components), 2)

            is_racemic = None

            for index_0, index_1 in component_combinations:

                components_racemic = component_data[
                    f"Component {index_0 + 1}"
                ].str.replace("@", "") == component_data[
                    f"Component {index_1 + 1}"
                ].str.replace(
                    "@", ""
                )

                is_racemic = (
                    components_racemic
                    if is_racemic is None
                    else (is_racemic | components_racemic)
                )

            not_racemic = ~is_racemic
            query |= not_racemic

        filtered_frame = data_frame[query]
        return filtered_frame


class FilterByElementsSchema(CurationComponentSchema):

    type: Literal["FilterByElements"] = "FilterByElements"

    allowed_elements: Optional[List[constr(min_length=1)]] = Field(
        None,
        description="The only elements which must be present in the measured system "
        "for the data point to be retained. This option is mutually exclusive with "
        "`forbidden_elements`",
    )
    forbidden_elements: Optional[List[constr(min_length=1)]] = Field(
        None,
        description="The elements which must not be present in the measured system for "
        "the data point to be retained. This option is mutually exclusive with "
        "`allowed_elements`",
    )

    @root_validator
    def _validate_mutually_exclusive(cls, values):

        allowed_elements = values.get("allowed_elements")
        forbidden_elements = values.get("forbidden_elements")

        assert allowed_elements is not None or forbidden_elements is not None
        assert allowed_elements is None or forbidden_elements is None

        return values


class FilterByElements(CurationComponent):
    """A component which will filter out data points which were measured for systems
    which contain specific elements."""

    @classmethod
    def _apply(
        cls, data_frame: pandas.DataFrame, schema: FilterByElementsSchema, n_processes
    ) -> pandas.DataFrame:

        from openff.toolkit.topology import Molecule

        def filter_function(data_row):

            n_components = data_row["N Components"]

            for index in range(n_components):

                smiles = data_row[f"Component {index + 1}"]
                molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

                if schema.allowed_elements is not None and not all(
                    [x.symbol in schema.allowed_elements for x in molecule.atoms]
                ):
                    return False

                if schema.forbidden_elements is not None and any(
                    [x.symbol in schema.forbidden_elements for x in molecule.atoms]
                ):
                    return False

            return True

        # noinspection PyTypeChecker
        return data_frame[data_frame.apply(filter_function, axis=1)]


class FilterByPropertyTypesSchema(CurationComponentSchema):

    type: Literal["FilterByPropertyTypes"] = "FilterByPropertyTypes"

    property_types: List[constr(min_length=1)] = Field(
        ...,
        description="The types of property to retain.",
    )
    n_components: Dict[constr(min_length=1), List[PositiveInt]] = Field(
        default_factory=dict,
        description="Optionally specify the number of components that a property "
        "should have been measured for (e.g. pure, binary) in order for that data "
        "point to be retained.",
    )

    strict: bool = Field(
        False,
        description="If true, only substances (defined without consideration for their "
        "mole fractions or exact amount) which have data available for all of the "
        "specified property types will be retained. Note that the data points aren't "
        "required to have been measured at the same state.",
    )

    @root_validator
    def _validate_n_components(cls, values):

        property_types = values.get("property_types")
        n_components = values.get("n_components")

        assert all(x in property_types for x in n_components)

        return values


class FilterByPropertyTypes(CurationComponent):
    """A component which will apply a filter which only retains properties of specified
    types."""

    @classmethod
    def _apply(
        cls,
        data_frame: pandas.DataFrame,
        schema: FilterByPropertyTypesSchema,
        n_processes,
    ) -> pandas.DataFrame:

        property_headers = [
            header for header in data_frame if header.find(" Value ") >= 0
        ]

        # Removes the columns for properties which are not of interest.
        for header in property_headers:

            property_type = header.split(" ")[0]

            if property_type in schema.property_types:
                continue

            data_frame = data_frame.drop(header, axis=1)

            uncertainty_header = header.replace(" Value ", " Uncertainty ")

            if uncertainty_header in data_frame:
                data_frame = data_frame.drop(uncertainty_header, axis=1)

        # Drop any rows which do not contain any values for the property types of
        # interest.
        property_headers = [
            header
            for header in property_headers
            if header.split(" ")[0] in schema.property_types
        ]

        data_frame = data_frame.dropna(subset=property_headers, how="all")

        # Apply a more specific filter which only retain which contain values
        # for the specific property types, and which were measured for the
        # specified number of components.
        for property_type, n_components in schema.n_components.items():

            property_header = next(
                iter(x for x in property_headers if x.find(f"{property_type} ") == 0),
                None,
            )

            if property_header is None:
                continue

            data_frame = data_frame[
                data_frame[property_header].isna()
                | data_frame["N Components"].isin(n_components)
            ]

        # Apply the strict filter if requested
        if schema.strict:

            reordered_data_frame = reorder_data_frame(data_frame)

            # Build a dictionary of which properties should be present partitioned
            # by the number of components they should have been be measured for.
            property_types = defaultdict(list)

            if len(schema.n_components) > 0:

                for property_type, n_components in schema.n_components.items():

                    for n_component in n_components:
                        property_types[n_component].append(property_type)

                min_n_components = min(property_types)
                max_n_components = max(property_types)

            else:

                min_n_components = reordered_data_frame["N Components"].min()
                max_n_components = reordered_data_frame["N Components"].max()

                for n_components in range(min_n_components, max_n_components + 1):
                    property_types[n_components].extend(schema.property_types)

            substances_with_data = set()
            components_with_data = {}

            # For each N component find substances which have data points for
            # all of the specified property types.
            for n_components in range(min_n_components, max_n_components + 1):

                component_data = reordered_data_frame[
                    reordered_data_frame["N Components"] == n_components
                ]

                if n_components not in property_types or len(component_data) == 0:
                    continue

                n_component_headers = [
                    header
                    for header in property_headers
                    if header.split(" ")[0] in property_types[n_components]
                    and header in component_data
                ]

                if len(n_component_headers) != len(property_types[n_components]):
                    continue

                n_component_substances = set.intersection(
                    *[
                        data_frame_to_substances(
                            component_data[component_data[header].notna()]
                        )
                        for header in n_component_headers
                    ]
                )
                substances_with_data.update(n_component_substances)
                components_with_data[n_components] = {
                    component
                    for substance in n_component_substances
                    for component in substance
                }

            if len(schema.n_components) > 0:
                components_with_all_data = set.intersection(
                    *components_with_data.values()
                )

                # Filter out any smiles for don't appear in all of the N component
                # substances.
                data_frame = FilterBySmiles.apply(
                    data_frame,
                    FilterBySmilesSchema(smiles_to_include=[*components_with_all_data]),
                )

            # Filter out any substances which (within each N component) don't have
            # all of the specified data types.
            data_frame = FilterBySubstances.apply(
                data_frame,
                FilterBySubstancesSchema(substances_to_include=[*substances_with_data]),
            )

        data_frame = data_frame.dropna(axis=1, how="all")
        return data_frame


class FilterByStereochemistrySchema(CurationComponentSchema):

    type: Literal["FilterByStereochemistry"] = "FilterByStereochemistry"


class FilterByStereochemistry(CurationComponent):
    """A component which filters out data points measured for systems whereby the
    stereochemistry of a number of components is undefined."""

    @classmethod
    def _apply(
        cls,
        data_frame: pandas.DataFrame,
        schema: FilterByStereochemistrySchema,
        n_processes,
    ) -> pandas.DataFrame:

        from openff.toolkit.topology import Molecule
        from openff.toolkit.utils import UndefinedStereochemistryError

        def filter_function(data_row):

            n_components = data_row["N Components"]

            for index in range(n_components):

                smiles = data_row[f"Component {index + 1}"]

                try:
                    Molecule.from_smiles(smiles)
                except UndefinedStereochemistryError:
                    return False

            return True

        # noinspection PyTypeChecker
        return data_frame[data_frame.apply(filter_function, axis=1)]


class FilterByChargedSchema(CurationComponentSchema):

    type: Literal["FilterByCharged"] = "FilterByCharged"


class FilterByCharged(CurationComponent):
    """A component which filters out data points measured for substances where any of
    the constituent components have a net non-zero charge.
    """

    @classmethod
    def _apply(
        cls, data_frame: pandas.DataFrame, schema: FilterByChargedSchema, n_processes
    ) -> pandas.DataFrame:

        from openff.toolkit.topology import Molecule

        def filter_function(data_row):

            n_components = data_row["N Components"]

            for index in range(n_components):

                smiles = data_row[f"Component {index + 1}"]
                molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

                # noinspection PyUnresolvedReferences
                atom_charges = [
                    atom.formal_charge
                    if isinstance(atom.formal_charge, int)
                    else atom.formal_charge.m_as(unit.elementary_charge)
                    for atom in molecule.atoms
                ]

                if numpy.isclose(sum(atom_charges), 0.0):
                    continue

                return False

            return True

        # noinspection PyTypeChecker
        return data_frame[data_frame.apply(filter_function, axis=1)]


class FilterByIonicLiquidSchema(CurationComponentSchema):
    type: Literal["FilterByIonicLiquid"] = "FilterByIonicLiquid"


class FilterByIonicLiquid(CurationComponent):
    """A component which filters out data points measured for substances which
    contain or are classed as an ionic liquids.
    """

    @classmethod
    def _apply(
        cls,
        data_frame: pandas.DataFrame,
        schema: FilterByIonicLiquidSchema,
        n_processes,
    ) -> pandas.DataFrame:
        def filter_function(data_row):

            n_components = data_row["N Components"]

            for index in range(n_components):

                smiles = data_row[f"Component {index + 1}"]

                if "." in smiles:
                    return False

            return True

        # noinspection PyTypeChecker
        return data_frame[data_frame.apply(filter_function, axis=1)]


class FilterBySmilesSchema(CurationComponentSchema):
    type: Literal["FilterBySmiles"] = "FilterBySmiles"

    smiles_to_include: Optional[List[str]] = Field(
        None,
        description="The smiles patterns to retain. This option is mutually "
        "exclusive with `smiles_to_exclude`",
    )
    smiles_to_exclude: Optional[List[str]] = Field(
        None,
        description="The smiles patterns to exclude. This option is mutually "
        "exclusive with `smiles_to_include`",
    )
    allow_partial_inclusion: bool = Field(
        False,
        description="If False, all the components in a substance must appear in "
        "the `smiles_to_include` list, otherwise, only some must appear. "
        "This option only applies when `smiles_to_include` is set.",
    )

    @root_validator
    def _validate_mutually_exclusive(cls, values):

        smiles_to_include = values.get("smiles_to_include")
        smiles_to_exclude = values.get("smiles_to_exclude")

        assert smiles_to_include is not None or smiles_to_exclude is not None
        assert smiles_to_include is None or smiles_to_exclude is None

        return values


class FilterBySmiles(CurationComponent):
    """A component which filters the data set so that it only contains either a
    specific set of smiles, or does not contain any of a set of specifically excluded
    smiles.
    """

    @classmethod
    def _apply(
        cls, data_frame: pandas.DataFrame, schema: FilterBySmilesSchema, n_processes
    ) -> pandas.DataFrame:

        smiles_to_include = schema.smiles_to_include
        smiles_to_exclude = schema.smiles_to_exclude

        if smiles_to_include is not None:
            smiles_to_exclude = []
        elif smiles_to_exclude is not None:
            smiles_to_include = []

        def filter_function(data_row):

            n_components = data_row["N Components"]

            component_smiles = [
                data_row[f"Component {index + 1}"] for index in range(n_components)
            ]

            if any(x in smiles_to_exclude for x in component_smiles):
                return False
            elif len(smiles_to_exclude) > 0:
                return True

            if not schema.allow_partial_inclusion and not all(
                x in smiles_to_include for x in component_smiles
            ):
                return False

            if schema.allow_partial_inclusion and not any(
                x in smiles_to_include for x in component_smiles
            ):
                return False

            return True

        # noinspection PyTypeChecker
        return data_frame[data_frame.apply(filter_function, axis=1)]


class FilterBySmirksSchema(CurationComponentSchema):

    type: Literal["FilterBySmirks"] = "FilterBySmirks"

    smirks_to_include: Optional[List[str]] = Field(
        None,
        description="The smirks patterns which must be matched by a substance in "
        "order to retain a measurement. This option is mutually exclusive with "
        "`smirks_to_exclude`",
    )
    smirks_to_exclude: Optional[List[str]] = Field(
        None,
        description="The smirks patterns which must not be matched by a substance in "
        "order to retain a measurement. This option is mutually exclusive with "
        "`smirks_to_include`",
    )
    allow_partial_inclusion: bool = Field(
        False,
        description="If False, all the components in a substance must match at least "
        "one pattern in `smirks_to_include` in order to retain a measurement, "
        "otherwise, only a least one component must match. This option only applies "
        "when `smirks_to_include` is set.",
    )

    @root_validator
    def _validate_mutually_exclusive(cls, values):

        smirks_to_include = values.get("smirks_to_include")
        smirks_to_exclude = values.get("smirks_to_exclude")

        assert smirks_to_include is not None or smirks_to_exclude is not None
        assert smirks_to_include is None or smirks_to_exclude is None

        return values


class FilterBySmirks(CurationComponent):
    """A component which filters a data set so that it only contains measurements made
    for molecules which contain (or don't) a set of chemical environments
    represented by SMIRKS patterns.
    """

    @staticmethod
    @functools.lru_cache(1000)
    def _find_smirks_matches(smiles_pattern, *smirks_patterns):
        """Determines which (if any) of the specified smirks match the specified
        molecule.

        Parameters
        ----------
        smiles_pattern: str
            The SMILES representation to try and match against.
        smirks_patterns: str
            The smirks patterns to try and match.

        Returns
        -------
        list of str
            The matched smirks patterns.
        """

        from openff.toolkit.topology import Molecule

        if len(smirks_patterns) == 0:
            return []

        molecule = Molecule.from_smiles(smiles_pattern, allow_undefined_stereo=True)

        matches = [
            smirks
            for smirks in smirks_patterns
            if len(molecule.chemical_environment_matches(smirks)) > 0
        ]

        return matches

    @classmethod
    def _apply(
        cls, data_frame: pandas.DataFrame, schema: FilterBySmirksSchema, n_processes
    ) -> pandas.DataFrame:

        smirks_to_match = (
            schema.smirks_to_include
            if schema.smirks_to_include
            else schema.smirks_to_exclude
        )

        def filter_function(data_row):

            n_components = data_row["N Components"]

            component_smiles = [
                data_row[f"Component {index + 1}"] for index in range(n_components)
            ]

            smirks_matches = {
                smiles: cls._find_smirks_matches(smiles, *smirks_to_match)
                for smiles in component_smiles
            }

            if schema.smirks_to_exclude is not None:
                return not any(len(x) > 0 for x in smirks_matches.values())

            if schema.allow_partial_inclusion:
                return any(len(x) > 0 for x in smirks_matches.values())

            return all(len(x) > 0 for x in smirks_matches.values())

        # noinspection PyTypeChecker
        return data_frame[data_frame.apply(filter_function, axis=1)]


class FilterByNComponentsSchema(CurationComponentSchema):

    type: Literal["FilterByNComponents"] = "FilterByNComponents"

    n_components: List[PositiveInt] = Field(
        ...,
        description="The number of components that measurements should have been "
        "measured for in order to be retained.",
    )


class FilterByNComponents(CurationComponent):
    """A component which filters out data points measured for systems with specified
    number of components.
    """

    @classmethod
    def _apply(
        cls,
        data_frame: pandas.DataFrame,
        schema: FilterByNComponentsSchema,
        n_processes,
    ) -> pandas.DataFrame:

        return data_frame[data_frame["N Components"].isin(schema.n_components)]


class FilterBySubstancesSchema(CurationComponentSchema):

    type: Literal["FilterBySubstances"] = "FilterBySubstances"

    substances_to_include: Optional[List[Tuple[str, ...]]] = Field(
        None,
        description="The substances compositions to retain, where each tuple in the "
        "list contains the smiles patterns which make up the substance to include. "
        "This option is mutually exclusive with `substances_to_exclude`.",
    )
    substances_to_exclude: Optional[List[Tuple[str, ...]]] = Field(
        None,
        description="The substances compositions to retain, where each tuple in the "
        "list contains the smiles patterns which make up the substance to exclude. "
        "This option is mutually exclusive with `substances_to_include`.",
    )

    @root_validator
    def _validate_mutually_exclusive(cls, values):

        substances_to_include = values.get("substances_to_include")
        substances_to_exclude = values.get("substances_to_exclude")

        assert substances_to_include is not None or substances_to_exclude is not None
        assert substances_to_include is None or substances_to_exclude is None

        return values


class FilterBySubstances(CurationComponent):
    """A component which filters the data set so that it only contains properties
    measured for particular substances.

    This method is similar to `filter_by_smiles`, however here we explicitly define
    the full substances compositions, rather than individual smiles which should
    either be included or excluded.

    Examples
    --------
    To filter the data set to only include measurements for pure methanol, pure
    benzene or an aqueous ethanol mix:

    >>> schema = FilterBySubstancesSchema(
    >>>     substances_to_include=[
    >>>         ('CO',),
    >>>         ('C1=CC=CC=C1',),
    >>>         ('CCO', 'O')
    >>>     ]
    >>> )

    To filter out measurements made for an aqueous mix of benzene:

    >>> schema = FilterBySubstancesSchema(
    >>>     substances_to_exclude=[('O', 'C1=CC=CC=C1')]
    >>> )
    """

    @classmethod
    def _apply(
        cls, data_frame: pandas.DataFrame, schema: FilterBySubstancesSchema, n_processes
    ) -> pandas.DataFrame:
        def filter_function(data_row):

            n_components = data_row["N Components"]

            substances_to_include = schema.substances_to_include
            substances_to_exclude = schema.substances_to_exclude

            if substances_to_include is not None:
                substances_to_include = [
                    tuple(sorted(x)) for x in substances_to_include
                ]
            if substances_to_exclude is not None:
                substances_to_exclude = [
                    tuple(sorted(x)) for x in substances_to_exclude
                ]

            substance = tuple(
                sorted(
                    [
                        data_row[f"Component {index + 1}"]
                        for index in range(n_components)
                    ]
                )
            )

            return (
                substances_to_exclude is not None
                and substance not in substances_to_exclude
            ) or (
                substances_to_include is not None and substance in substances_to_include
            )

        # noinspection PyTypeChecker
        return data_frame[data_frame.apply(filter_function, axis=1)]


class FilterByEnvironmentsSchema(CurationComponentSchema):

    type: Literal["FilterByEnvironments"] = "FilterByEnvironments"

    per_component_environments: Optional[Dict[int, ComponentEnvironments]] = Field(
        None,
        description="The environments which should be present in the components of "
        "the substance for which the measurements were made. Each dictionary "
        "key corresponds to a number of components in the system, and each "
        "value the environments which should be matched by those n components. "
        "This option is mutually exclusive with `environments`.",
    )
    environments: Optional[List[ChemicalEnvironment]] = Field(
        None,
        description="The environments which should be present in the substances for "
        "which measurements were made. This option is mutually exclusive with "
        "`per_component_environments`.",
    )

    at_least_one_environment: bool = Field(
        True,
        description="If true, data points will only be retained if all of the "
        "components in the measured system contain at least one of the specified "
        "environments. This option is mutually exclusive with "
        "`strictly_specified_environments`.",
    )
    strictly_specified_environments: bool = Field(
        False,
        description="If true, data points will only be retained if all of the "
        "components in the measured system strictly contain only the specified "
        "environments and no others. This option is mutually exclusive with "
        "`at_least_one_environment`.",
    )

    @validator("per_component_environments")
    def _validate_per_component_environments(cls, value):

        if value is None:
            return value

        assert all(len(y) == x for x, y in value.items())
        return value

    @root_validator
    def _validate_mutually_exclusive(cls, values):

        at_least_one_environment = values.get("at_least_one_environment")
        strictly_specified_environments = values.get("strictly_specified_environments")

        assert (
            at_least_one_environment is True or strictly_specified_environments is True
        )
        assert (
            at_least_one_environment is False
            or strictly_specified_environments is False
        )

        per_component_environments = values.get("per_component_environments")
        environments = values.get("environments")

        assert per_component_environments is not None or environments is not None
        assert per_component_environments is None or environments is None

        return values


class FilterByEnvironments(CurationComponent):
    """A component which filters a data set so that it only contains measurements made
    for substances which contain specific chemical environments.
    """

    @classmethod
    def _find_environments_per_component(cls, data_row: pandas.Series):

        n_components = data_row["N Components"]

        component_smiles = [
            data_row[f"Component {index + 1}"] for index in range(n_components)
        ]
        component_moieties = [analyse_functional_groups(x) for x in component_smiles]

        if any(x is None for x in component_moieties):

            logger.info(
                f"Checkmol was unable to parse the system with components="
                f"{component_smiles} and so this data point was discarded."
            )

            return None

        return component_moieties

    @classmethod
    def _is_match(cls, component_environments, environments_to_match, schema):

        operator = all if schema.strictly_specified_environments else any

        return operator(
            environment in environments_to_match
            for environment in component_environments
        )

    @classmethod
    def _filter_by_environments(cls, data_row, schema: FilterByEnvironmentsSchema):

        environments_per_component = cls._find_environments_per_component(data_row)

        if environments_per_component is None:
            return False

        return all(
            cls._is_match(component_environments, schema.environments, schema)
            for component_environments in environments_per_component
        )

    @classmethod
    def _filter_by_per_component(cls, data_row, schema: FilterByEnvironmentsSchema):

        n_components = data_row["N Components"]

        if (
            schema.per_component_environments is not None
            and n_components not in schema.per_component_environments
        ):
            # No filter was specified for this number of components.
            return True

        environments_per_component = cls._find_environments_per_component(data_row)

        if environments_per_component is None:
            return False

        match_matrix = numpy.zeros((n_components, n_components))

        for component_index, component_environments in enumerate(
            environments_per_component
        ):

            # noinspection PyUnresolvedReferences
            for environments_index, environments_to_match in enumerate(
                schema.per_component_environments[n_components]
            ):

                match_matrix[component_index, environments_index] = cls._is_match(
                    component_environments, environments_to_match, schema
                )

        x_indices, y_indices = linear_sum_assignment(match_matrix, maximize=True)

        return numpy.all(match_matrix[x_indices, y_indices] > 0)

    @classmethod
    def _apply(
        cls,
        data_frame: pandas.DataFrame,
        schema: FilterByEnvironmentsSchema,
        n_processes,
    ) -> pandas.DataFrame:

        if schema.environments is not None:
            filter_function = functools.partial(
                cls._filter_by_environments, schema=schema
            )
        else:
            filter_function = functools.partial(
                cls._filter_by_per_component, schema=schema
            )

        # noinspection PyTypeChecker
        return data_frame[data_frame.apply(filter_function, axis=1)]


FilterComponentSchema = Union[
    FilterDuplicatesSchema,
    FilterByTemperatureSchema,
    FilterByPressureSchema,
    FilterByMoleFractionSchema,
    FilterByRacemicSchema,
    FilterByElementsSchema,
    FilterByPropertyTypesSchema,
    FilterByStereochemistrySchema,
    FilterByChargedSchema,
    FilterByIonicLiquidSchema,
    FilterBySmilesSchema,
    FilterBySmirksSchema,
    FilterByNComponentsSchema,
    FilterBySubstancesSchema,
    FilterByEnvironmentsSchema,
]
