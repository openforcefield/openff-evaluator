from typing import List, Tuple

import numpy
import pandas
import pytest
from openff.units import unit
from pydantic import ValidationError

from openff.evaluator.datasets import (
    MeasurementSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.datasets.curation.components.filtering import (
    FilterByCharged,
    FilterByChargedSchema,
    FilterByElements,
    FilterByElementsSchema,
    FilterByEnvironments,
    FilterByEnvironmentsSchema,
    FilterByIonicLiquid,
    FilterByIonicLiquidSchema,
    FilterByMoleFraction,
    FilterByMoleFractionSchema,
    FilterByNComponents,
    FilterByNComponentsSchema,
    FilterByPressure,
    FilterByPressureSchema,
    FilterByPropertyTypes,
    FilterByPropertyTypesSchema,
    FilterByRacemic,
    FilterByRacemicSchema,
    FilterBySmiles,
    FilterBySmilesSchema,
    FilterBySmirks,
    FilterBySmirksSchema,
    FilterByStereochemistry,
    FilterByStereochemistrySchema,
    FilterBySubstances,
    FilterBySubstancesSchema,
    FilterByTemperature,
    FilterByTemperatureSchema,
    FilterDuplicates,
    FilterDuplicatesSchema,
)
from openff.evaluator.datasets.curation.workflow import (
    CurationWorkflow,
    CurationWorkflowSchema,
)
from openff.evaluator.datasets.utilities import data_frame_to_substances
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.substances import Component, MoleFraction, Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.checkmol import ChemicalEnvironment


def _build_entry(*smiles: str) -> Density:
    """Builds a density data entry measured at ambient conditions
    and for a system containing the specified smiles patterns in
    equal amounts.

    Parameters
    ----------
    smiles
        The smiles to build components for.

    Returns
    -------
        The built components.
    """
    assert len(smiles) > 0

    return Density(
        thermodynamic_state=ThermodynamicState(
            temperature=298.15 * unit.kelvin,
            pressure=101.325 * unit.kilopascal,
        ),
        phase=PropertyPhase.Liquid,
        value=1.0 * Density.default_unit(),
        uncertainty=1.0 * Density.default_unit(),
        source=MeasurementSource(doi=" "),
        substance=Substance.from_components(*smiles),
    )


def _build_data_frame(
    property_types: List[str],
    substance_entries: List[Tuple[Tuple[str, ...], Tuple[bool, ...]]],
) -> pandas.DataFrame:
    data_rows = []

    for substance, include_properties in substance_entries:
        for property_type, include_property in zip(property_types, include_properties):
            if not include_property:
                continue

            data_row = {
                "N Components": len(substance),
                f"{property_type} Value (unit)": 1.0,
            }

            for index, component in enumerate(substance):
                data_row[f"Component {index + 1}"] = component

            data_rows.append(data_row)

    data_frame = pandas.DataFrame(data_rows)
    return data_frame


@pytest.fixture(scope="module")
def data_frame() -> pandas.DataFrame:
    temperatures = [298.15, 318.15]
    pressures = [101.325, 101.0]

    properties = [Density, EnthalpyOfMixing]

    mole_fractions = [(1.0,), (1.0,), (0.25, 0.75), (0.75, 0.25)]
    smiles = {1: [("C(F)(Cl)(Br)",), ("C",)], 2: [("CO", "C"), ("C", "CO")]}

    loop_variables = [
        (
            temperature,
            pressure,
            property_type,
            mole_fraction,
        )
        for temperature in temperatures
        for pressure in pressures
        for property_type in properties
        for mole_fraction in mole_fractions
    ]

    data_entries = []

    for temperature, pressure, property_type, mole_fraction in loop_variables:
        n_components = len(mole_fraction)

        for smiles_tuple in smiles[n_components]:
            substance = Substance()

            for smiles_pattern, x in zip(smiles_tuple, mole_fraction):
                substance.add_component(Component(smiles_pattern), MoleFraction(x))

            data_entries.append(
                property_type(
                    thermodynamic_state=ThermodynamicState(
                        temperature=temperature * unit.kelvin,
                        pressure=pressure * unit.kilopascal,
                    ),
                    phase=PropertyPhase.Liquid,
                    value=1.0 * property_type.default_unit(),
                    uncertainty=1.0 * property_type.default_unit(),
                    source=MeasurementSource(doi=" "),
                    substance=substance,
                )
            )

    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(*data_entries)

    return data_set.to_pandas()


def test_filter_duplicates(data_frame):
    filtered_frame = FilterDuplicates.apply(data_frame, FilterDuplicatesSchema(), 1)

    pure_data: pandas.DataFrame = filtered_frame[filtered_frame["N Components"] == 1]

    assert len(pure_data) == 16

    assert len(pure_data[pure_data["EnthalpyOfMixing Value (kJ / mol)"].notna()]) == 8
    assert len(pure_data[pure_data["Density Value (g / ml)"].notna()]) == 8

    binary_data: pandas.DataFrame = filtered_frame[filtered_frame["N Components"] == 2]

    assert len(filtered_frame[filtered_frame["N Components"] == 2]) == 16

    assert (
        len(binary_data[binary_data["EnthalpyOfMixing Value (kJ / mol)"].notna()]) == 8
    )
    assert len(binary_data[binary_data["Density Value (g / ml)"].notna()]) == 8


def test_validate_filter_by_temperature():
    # Ensure a valid schema passes
    FilterByTemperatureSchema(minimum_temperature=1.0, maximum_temperature=2.0)

    # Test that an exception is raised when the minimum temperature is
    # greater than the maximum.
    with pytest.raises(ValidationError):
        FilterByTemperatureSchema(minimum_temperature=2.0, maximum_temperature=1.0)


def test_filter_by_temperature(data_frame):
    # Apply a filter which should have no effect.
    filtered_frame = FilterByTemperature.apply(
        data_frame,
        FilterByTemperatureSchema(minimum_temperature=290.0, maximum_temperature=320.0),
    )

    assert len(filtered_frame) == len(data_frame)

    # Filter out the minimum values
    filtered_frame = FilterByTemperature.apply(
        data_frame,
        FilterByTemperatureSchema(minimum_temperature=300.0, maximum_temperature=None),
    )

    assert len(filtered_frame) == len(data_frame) / 2

    temperatures = filtered_frame["Temperature (K)"].unique()

    assert len(temperatures) == 1
    assert numpy.isclose(temperatures[0], 318.15)

    # Filter out the maximum values
    filtered_frame = FilterByTemperature.apply(
        data_frame,
        FilterByTemperatureSchema(
            minimum_temperature=None,
            maximum_temperature=300.0,
        ),
    )

    assert len(filtered_frame) == len(data_frame) / 2

    temperatures = filtered_frame["Temperature (K)"].unique()

    assert len(temperatures) == 1
    assert numpy.isclose(temperatures[0], 298.15)


def test_validate_filter_by_pressure():
    # Ensure a valid schema passes
    FilterByPressureSchema(minimum_pressure=1.0, maximum_pressure=2.0)

    # Test that an exception is raised when the minimum pressure is
    # greater than the maximum.
    with pytest.raises(ValidationError):
        FilterByPressureSchema(minimum_pressure=2.0, maximum_pressure=1.0)


def test_filter_by_pressure(data_frame):
    # Apply a filter which should have no effect.
    filtered_frame = FilterByPressure.apply(
        data_frame,
        FilterByPressureSchema(minimum_pressure=100.0, maximum_pressure=140.0),
    )

    assert len(filtered_frame) == len(data_frame)

    # Filter out the minimum values
    filtered_frame = FilterByPressure.apply(
        data_frame,
        FilterByPressureSchema(minimum_pressure=101.2, maximum_pressure=None),
    )

    assert len(filtered_frame) == len(data_frame) / 2

    pressures = filtered_frame["Pressure (kPa)"].unique()

    assert len(pressures) == 1
    assert numpy.isclose(pressures[0], 101.325)

    # Filter out the maximum values
    filtered_frame = FilterByPressure.apply(
        data_frame,
        FilterByPressureSchema(
            minimum_pressure=None,
            maximum_pressure=101.2,
        ),
    )

    assert len(filtered_frame) == len(data_frame) / 2

    pressures = filtered_frame["Pressure (kPa)"].unique()

    assert len(pressures) == 1
    assert numpy.isclose(pressures[0], 101.0)


def test_validate_filter_by_mole_fraction():
    # Ensure a valid schema passes
    FilterByMoleFractionSchema(
        mole_fraction_ranges={2: [[(0.2, 0.8)]], 3: [[(0.1, 0.2)], [(0.4, 0.5)]]}
    )

    # Test that an exception is raised when the wrong number of component
    # lists is provided.
    with pytest.raises(ValidationError):
        FilterByMoleFractionSchema(
            mole_fraction_ranges={2: [[(0.2, 0.8)], [(0.2, 0.8)]]}
        )

    with pytest.raises(ValidationError):
        FilterByMoleFractionSchema(mole_fraction_ranges={3: [[(0.2, 0.8)]]})

    # Test that an exception is raised when a bad range is provided.
    with pytest.raises(ValidationError):
        FilterByMoleFractionSchema(mole_fraction_ranges={2: [[(0.8, 0.2)]]})

    with pytest.raises(ValidationError):
        FilterByMoleFractionSchema(mole_fraction_ranges={2: [[(-0.8, 0.2)]]})

    with pytest.raises(ValidationError):
        FilterByMoleFractionSchema(mole_fraction_ranges={2: [[(0.8, 1.2)]]})


def test_filter_by_mole_fraction(data_frame):
    data_rows = [
        {"N Components": 1, "Component 1": "CCCCC", "Mole Fraction 1": 1.0},
        {
            "N Components": 2,
            "Component 1": "CCCCC",
            "Mole Fraction 1": 0.2,
            "Component 2": "CCCCCO",
            "Mole Fraction 2": 0.8,
        },
        {
            "N Components": 2,
            "Component 1": "CCCCC",
            "Mole Fraction 1": 0.8,
            "Component 2": "CCCCCO",
            "Mole Fraction 2": 0.2,
        },
        {
            "N Components": 2,
            "Component 1": "CCCCC",
            "Mole Fraction 1": 0.5,
            "Component 2": "CCCCCO",
            "Mole Fraction 2": 0.5,
        },
    ]

    data_frame = pandas.DataFrame(data_rows)

    # Apply a filter which should have no effect.
    filtered_frame = FilterByMoleFraction.apply(
        data_frame, FilterByMoleFractionSchema(mole_fraction_ranges={})
    )

    assert len(filtered_frame) == len(data_frame)

    # Retain only the minimum value
    filtered_frame = FilterByMoleFraction.apply(
        data_frame, FilterByMoleFractionSchema(mole_fraction_ranges={2: [[(0.1, 0.3)]]})
    )

    assert len(filtered_frame) == 2
    assert len(filtered_frame[filtered_frame["N Components"] == 1]) == 1
    assert len(filtered_frame[filtered_frame["N Components"] == 2]) == 1

    filtered_frame = filtered_frame[filtered_frame["N Components"] == 2]
    assert numpy.isclose(filtered_frame["Mole Fraction 1"], 0.2)

    # Drop the pure data point to make the test cleaner from this point on.
    data_frame = data_frame[data_frame["N Components"] == 2]

    # Retain only the maximum value
    filtered_frame = FilterByMoleFraction.apply(
        data_frame, FilterByMoleFractionSchema(mole_fraction_ranges={2: [[(0.7, 0.9)]]})
    )

    assert len(filtered_frame) == 1
    assert numpy.isclose(filtered_frame["Mole Fraction 1"], 0.8)

    # Retain both the minimum and maximum values
    filtered_frame = FilterByMoleFraction.apply(
        data_frame,
        FilterByMoleFractionSchema(
            mole_fraction_ranges={2: [[(0.1, 0.3), (0.7, 0.9)]]}
        ),
    )

    assert len(filtered_frame) == 2
    assert all(filtered_frame["Mole Fraction 1"].round(1).isin([0.2, 0.8]))


def test_filter_by_racemic():
    data_rows = [
        {"N Components": 1, "Component 1": "N[C@H](C)C(=O)O"},
        {"N Components": 1, "Component 1": "N[C@@H](C)C(=O)O"},
        {"N Components": 2, "Component 1": "C", "Component 2": "N[C@H](C)C(=O)O"},
        {
            "N Components": 2,
            "Component 1": "N[C@@H](C)C(=O)O",
            "Component 2": "N[C@H](C)C(=O)O",
        },
        {
            "N Components": 3,
            "Component 1": "C",
            "Component 2": "N[C@@H](C)C(=O)O",
            "Component 3": "N[C@H](C)C(=O)O",
        },
        {
            "N Components": 3,
            "Component 1": "N[C@@H](C)C(=O)O",
            "Component 2": "C",
            "Component 3": "N[C@H](C)C(=O)O",
        },
        {
            "N Components": 3,
            "Component 1": "N[C@@H](C)C(=O)O",
            "Component 2": "N[C@H](C)C(=O)O",
            "Component 3": "C",
        },
    ]

    data_frame = pandas.DataFrame(data_rows)

    # Apply the filter
    filtered_frame = FilterByRacemic.apply(data_frame, FilterByRacemicSchema())

    assert len(filtered_frame[filtered_frame["N Components"] == 1]) == 2

    assert len(filtered_frame[filtered_frame["N Components"] == 2]) == 1
    binary_data = filtered_frame[filtered_frame["N Components"] == 2]
    assert binary_data["Component 1"].unique()[0] == "C"

    assert len(filtered_frame[filtered_frame["N Components"] == 3]) == 0


def test_validate_filter_by_elements():
    # Ensure a valid schema passes
    FilterByElementsSchema(allowed_elements=["C"])
    FilterByElementsSchema(forbidden_elements=["C"])

    # Test that an exception is raised when mutually exclusive options
    # are provided.
    with pytest.raises(ValidationError):
        FilterByElementsSchema(allowed_elements=["C"], forbidden_elements=["C"])


def test_filter_by_elements(data_frame):
    # Apply a filter which should have no effect.
    filtered_frame = FilterByElements.apply(
        data_frame,
        FilterByElementsSchema(allowed_elements=["C", "O", "H", "F", "Cl", "Br"]),
    )

    assert len(filtered_frame) == len(data_frame)

    filtered_frame = FilterByElements.apply(
        data_frame,
        FilterByElementsSchema(forbidden_elements=[]),
    )

    assert len(filtered_frame) == len(data_frame)

    # Filter out all oxygen containing molecules. This should leave pure
    # only measurements.
    filtered_frame = FilterByElements.apply(
        data_frame,
        FilterByElementsSchema(forbidden_elements=["O"]),
    )

    assert len(filtered_frame[filtered_frame["N Components"] == 1]) == 32

    # Filter out any non-hydrocarbons.
    filtered_frame = FilterByElements.apply(
        data_frame,
        FilterByElementsSchema(allowed_elements=["C", "H"]),
    )

    assert len(filtered_frame) == 16
    assert len(filtered_frame["Component 1"].unique()) == 1
    assert filtered_frame["Component 1"].unique()[0] == "C"
    assert filtered_frame["N Components"].max() == 1


def test_validate_filter_by_property():
    # Ensure a valid schema passes
    FilterByPropertyTypesSchema(property_types=["Density"])
    FilterByPropertyTypesSchema(
        property_types=["Density"], n_components={"Density": [1]}
    )

    # Test that an exception is raised when a property type is included
    # in `n_components` but not `property_types`
    with pytest.raises(ValidationError):
        FilterByPropertyTypesSchema(property_types=[], n_components={"Density": [1]})


def test_filter_by_property(data_frame):
    # Apply a filter which should have no effect.
    filtered_frame = FilterByPropertyTypes.apply(
        data_frame,
        FilterByPropertyTypesSchema(property_types=["Density", "EnthalpyOfMixing"]),
    )

    assert len(filtered_frame) == len(data_frame)

    # Filter out all density measurements.
    filtered_frame = FilterByPropertyTypes.apply(
        data_frame, FilterByPropertyTypesSchema(property_types=["EnthalpyOfMixing"])
    )

    assert len(filtered_frame) == len(data_frame) / 2

    filtered_frame = FilterByPropertyTypes.apply(
        data_frame,
        FilterByPropertyTypesSchema(
            property_types=["EnthalpyOfMixing"],
            n_components={"EnthalpyOfMixing": [1, 2]},
        ),
    )

    assert len(filtered_frame) == len(data_frame) / 2

    # Filter out anything but pure density measurements.
    filtered_frame = FilterByPropertyTypes.apply(
        data_frame,
        FilterByPropertyTypesSchema(
            property_types=["Density"], n_components={"Density": [1]}
        ),
    )

    assert len(filtered_frame) == 16
    assert filtered_frame["N Components"].max() == 1

    assert len(filtered_frame[filtered_frame["Density Value (g / ml)"].notna()]) == 16
    assert "EnthalpyOfMixing Value (kJ / mol)" not in filtered_frame

    # Retain only pure densities and binary enthalpies of mixing.
    filtered_frame = FilterByPropertyTypes.apply(
        data_frame,
        FilterByPropertyTypesSchema(
            property_types=["Density", "EnthalpyOfMixing"],
            n_components={"Density": [1], "EnthalpyOfMixing": [2]},
        ),
    )

    assert len(filtered_frame) == 32

    assert len(filtered_frame[filtered_frame["N Components"] == 1]) == 16
    assert len(filtered_frame[filtered_frame["N Components"] == 2]) == 16

    assert len(filtered_frame[filtered_frame["Density Value (g / ml)"].notna()]) == 16
    assert (
        len(filtered_frame[filtered_frame["EnthalpyOfMixing Value (kJ / mol)"].notna()])
        == 16
    )

    assert (
        filtered_frame[filtered_frame["Density Value (g / ml)"].notna()][
            "N Components"
        ].max()
        == 1
    )
    assert (
        filtered_frame[filtered_frame["EnthalpyOfMixing Value (kJ / mol)"].notna()][
            "N Components"
        ].min()
        == 2
    )


def test_filter_by_property_strict():
    """Tests that the FilterByPropertyTypes filter works
    correctly when strict mode is set but n_components is not.
    """

    property_types = ["Density", "DielectricConstant"]
    substance_entries = [
        (("CC",), (True, True)),
        (("CCC",), (True, False)),
        (("CCCCC",), (True, True)),
        (("CC", "CCC"), (True, True)),
        (("CCC", "CCC"), (True, False)),
        (("CCC", "CCCC"), (False, True)),
    ]

    data_frame = _build_data_frame(property_types, substance_entries)

    filtered_frame = FilterByPropertyTypes.apply(
        data_frame,
        FilterByPropertyTypesSchema(property_types=property_types, strict=True),
    )

    assert len(filtered_frame) == 6

    assert data_frame_to_substances(filtered_frame) == {
        ("CC",),
        ("CCCCC",),
        ("CC", "CCC"),
    }


def test_filter_by_property_strict_n_components():
    """Tests that the FilterByPropertyTypes filter works
    correctly when strict mode and n_components is set.
    """

    property_types = ["Density", "EnthalpyOfVaporization", "EnthalpyOfMixing"]
    substance_entries = [
        (("CC",), (True, True, False)),
        (("CCC",), (True, True, False)),
        (("CCCCC",), (True, False, False)),
        (("CCCCCC",), (True, True, False)),
        (("CC", "CCC"), (True, False, True)),
        (("CC", "CCCCC"), (True, False, True)),
        (("CCC", "CCC"), (True, False, False)),
        (("CCC", "CCCC"), (False, False, True)),
    ]

    data_frame = _build_data_frame(property_types, substance_entries)

    filtered_frame = FilterByPropertyTypes.apply(
        data_frame,
        FilterByPropertyTypesSchema(
            property_types=property_types,
            n_components={
                "Density": [1, 2],
                "EnthalpyOfVaporization": [1],
                "EnthalpyOfMixing": [2],
            },
            strict=True,
        ),
    )

    assert len(filtered_frame) == 6

    assert data_frame_to_substances(filtered_frame) == {
        ("CC",),
        ("CCC",),
        ("CC", "CCC"),
    }


def test_filter_stereochemistry(data_frame):
    # Ensure molecules with undefined stereochemistry are filtered.
    filtered_frame = FilterByStereochemistry.apply(
        data_frame,
        FilterByStereochemistrySchema(),
    )

    assert len(filtered_frame) == len(data_frame) - 16


def test_filter_charged():
    thermodynamic_state = ThermodynamicState(
        temperature=298.15 * unit.kelvin,
        pressure=101.325 * unit.kilopascal,
    )

    # Ensure charged molecules are filtered.
    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(
        Density(
            thermodynamic_state=thermodynamic_state,
            phase=PropertyPhase.Liquid,
            value=1.0 * Density.default_unit(),
            uncertainty=1.0 * Density.default_unit(),
            source=MeasurementSource(doi=" "),
            substance=Substance.from_components("[Cl-]"),
        ),
        Density(
            thermodynamic_state=thermodynamic_state,
            phase=PropertyPhase.Liquid,
            value=1.0 * Density.default_unit(),
            uncertainty=1.0 * Density.default_unit(),
            source=MeasurementSource(doi=" "),
            substance=Substance.from_components("[Cl-]", "C"),
        ),
        Density(
            thermodynamic_state=thermodynamic_state,
            phase=PropertyPhase.Liquid,
            value=1.0 * Density.default_unit(),
            uncertainty=1.0 * Density.default_unit(),
            source=MeasurementSource(doi=" "),
            substance=Substance.from_components("C"),
        ),
    )

    data_frame = data_set.to_pandas()

    filtered_frame = FilterByCharged.apply(
        data_frame,
        FilterByChargedSchema(),
    )

    assert len(filtered_frame) == 1
    assert filtered_frame["N Components"].max() == 1


def test_filter_ionic_liquid():
    thermodynamic_state = ThermodynamicState(
        temperature=298.15 * unit.kelvin,
        pressure=101.325 * unit.kilopascal,
    )

    # Ensure ionic liquids are filtered.
    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(
        Density(
            thermodynamic_state=thermodynamic_state,
            phase=PropertyPhase.Liquid,
            value=1.0 * Density.default_unit(),
            uncertainty=1.0 * Density.default_unit(),
            source=MeasurementSource(doi=" "),
            substance=Substance.from_components("[Na+].[Cl-]"),
        ),
        Density(
            thermodynamic_state=thermodynamic_state,
            phase=PropertyPhase.Liquid,
            value=1.0 * Density.default_unit(),
            uncertainty=1.0 * Density.default_unit(),
            source=MeasurementSource(doi=" "),
            substance=Substance.from_components("C"),
        ),
    )

    data_frame = data_set.to_pandas()

    filtered_frame = FilterByIonicLiquid.apply(
        data_frame,
        FilterByIonicLiquidSchema(),
    )

    assert len(filtered_frame) == 1


def test_validate_filter_by_smiles():
    # Ensure a valid schema passes
    FilterBySmilesSchema(smiles_to_include=["C"])
    FilterBySmilesSchema(smiles_to_exclude=["C"])

    # Test that an exception is raised when mutually exclusive options
    # are provided.
    with pytest.raises(ValidationError):
        FilterBySmilesSchema(smiles_to_include=["C"], smiles_to_exclude=["C"])


def test_filter_by_smiles(data_frame):
    # Strictly only retain hydrocarbons. This should only leave pure
    # properties.
    filtered_frame = FilterBySmiles.apply(
        data_frame,
        FilterBySmilesSchema(smiles_to_include=["C"]),
    )

    assert len(filtered_frame) == 16
    assert filtered_frame["N Components"].max() == 1

    assert {*filtered_frame["Component 1"].unique()} == {"C"}

    # Make sure that partial inclusion works well when there are only
    # pure components.
    pure_data = data_frame[data_frame["N Components"] == 1]

    filtered_frame = FilterBySmiles.apply(
        pure_data,
        FilterBySmilesSchema(smiles_to_include=["C"], allow_partial_inclusion=True),
    )

    assert len(filtered_frame) == 16
    assert {*filtered_frame["Component 1"].unique()} == {"C"}

    # Now retain only retain hydrocarbons or mixtures containing hydrocarbons.
    filtered_frame = FilterBySmiles.apply(
        data_frame,
        FilterBySmilesSchema(smiles_to_include=["C"], allow_partial_inclusion=True),
    )

    pure_data = filtered_frame[filtered_frame["N Components"] == 1]
    binary_data = filtered_frame[filtered_frame["N Components"] == 2]

    assert len(pure_data) == 16
    assert {*pure_data["Component 1"].unique()} == {"C"}

    assert len(binary_data) == len(data_frame[data_frame["N Components"] == 2])

    # Exclude any hydrocarbons
    filtered_frame = FilterBySmiles.apply(
        data_frame,
        FilterBySmilesSchema(smiles_to_exclude=["C"]),
    )

    pure_data = filtered_frame[filtered_frame["N Components"] == 1]
    binary_data = filtered_frame[filtered_frame["N Components"] == 2]

    assert len(pure_data) == 16

    unique_components = {*pure_data["Component 1"].unique()}
    assert unique_components == {"C(F)(Cl)(Br)"} or unique_components == {"FC(Cl)Br"}

    assert len(binary_data) == 0


def test_validate_filter_by_smirks():
    # Ensure a valid schema passes
    FilterBySmirksSchema(smirks_to_include=["[#6]"])
    FilterBySmirksSchema(smirks_to_exclude=["[#6]"])

    # Test that an exception is raised when mutually exclusive options
    # are provided.
    with pytest.raises(ValidationError):
        FilterBySmirksSchema(smirks_to_include=["[#6]"], smirks_to_exclude=["[#6]"])


def test_find_smirks_matches():
    """A simple test that the `FilterBySmirks` smirks matching utility
    functions as expected."""

    # Test that nothing is returned when no smirks are provided.
    assert FilterBySmirks._find_smirks_matches("CCC") == []

    # Test that an alkane is correctly matched
    assert FilterBySmirks._find_smirks_matches("CCC", "[#6:1]") == ["[#6:1]"]

    # Test that no matches are found for water
    assert FilterBySmirks._find_smirks_matches("O", "[#6:1]") == []

    # See issue 502
    assert FilterBySmirks._find_smirks_matches("[2H]OC", "[2H]") == ["[2H]"]


def test_filter_by_smirks(data_frame):
    # Apply a filter which should do nothing.
    filtered_frame = FilterBySmirks.apply(
        data_frame,
        FilterBySmirksSchema(smirks_to_include=["[#6]"]),
    )

    assert len(filtered_frame) == len(data_frame) == 64

    # Retain only oxygen or halogen containing compounds.
    filtered_frame = FilterBySmirks.apply(
        data_frame,
        FilterBySmirksSchema(
            smirks_to_include=["[#8]", "[#9,#17,#35]"], allow_partial_inclusion=True
        ),
    )

    pure_data = filtered_frame[filtered_frame["N Components"] == 1]
    binary_data = filtered_frame[filtered_frame["N Components"] == 2]

    assert len(pure_data) == 16

    unique_components = {*pure_data["Component 1"].unique()}
    assert unique_components == {"C(F)(Cl)(Br)"} or unique_components == {"FC(Cl)Br"}

    assert len(binary_data) == len(data_frame[data_frame["N Components"] == 2])

    # Exclude all oxygen containing compounds
    filtered_frame = FilterBySmirks.apply(
        data_frame,
        FilterBySmirksSchema(smirks_to_exclude=["[#8]"]),
    )

    pure_data = filtered_frame[filtered_frame["N Components"] == 1]
    binary_data = filtered_frame[filtered_frame["N Components"] == 2]

    assert len(pure_data) == 32
    assert len(binary_data) == 0


def test_filter_by_n_components(data_frame):
    # Apply a filter which should do nothing
    filtered_frame = FilterByNComponents.apply(
        data_frame, FilterByNComponentsSchema(n_components=[1, 2])
    )

    assert len(filtered_frame) == len(data_frame)

    # Retrain only pure measurements
    filtered_frame = FilterByNComponents.apply(
        data_frame, FilterByNComponentsSchema(n_components=[1])
    )

    assert len(filtered_frame) == len(data_frame) / 2
    assert filtered_frame["N Components"].max() == 1

    # Retrain only binary measurements
    filtered_frame = FilterByNComponents.apply(
        data_frame, FilterByNComponentsSchema(n_components=[2])
    )

    assert len(filtered_frame) == len(data_frame) / 2
    assert filtered_frame["N Components"].min() == 2


def test_validate_filter_by_substances():
    # Ensure a valid schema passes
    FilterBySubstancesSchema(substances_to_include=[("C",)])
    FilterBySubstancesSchema(substances_to_exclude=[("C",)])

    # Test that an exception is raised when mutually exclusive options
    # are provided.
    with pytest.raises(ValidationError):
        FilterBySubstancesSchema(
            substances_to_include=[("C",)], substances_to_exclude=[("C",)]
        )


def test_filter_by_substances(data_frame):
    # Retain only the pure hydrocarbons.
    filtered_frame = FilterBySubstances.apply(
        data_frame, FilterBySubstancesSchema(substances_to_include=[("C",)])
    )

    assert len(filtered_frame) == 16
    assert filtered_frame["N Components"].max() == 1

    assert {*filtered_frame["Component 1"].unique()} == {"C"}

    # Retain only the mixtures, making sure the filter is invariant to component
    # order.
    filtered_frame = FilterBySubstances.apply(
        data_frame, FilterBySubstancesSchema(substances_to_include=[("C", "CO")])
    )

    assert len(filtered_frame) == 32
    assert filtered_frame["N Components"].min() == 2

    filtered_frame = FilterBySubstances.apply(
        data_frame, FilterBySubstancesSchema(substances_to_include=[("CO", "C")])
    )

    assert len(filtered_frame) == 32
    assert filtered_frame["N Components"].min() == 2

    # Exclude the mixtures, making sure the filter is invariant to component
    # order.
    filtered_frame = FilterBySubstances.apply(
        data_frame, FilterBySubstancesSchema(substances_to_exclude=[("C", "CO")])
    )

    assert len(filtered_frame) == 32
    assert filtered_frame["N Components"].max() == 1

    filtered_frame = FilterBySubstances.apply(
        data_frame, FilterBySubstancesSchema(substances_to_exclude=[("CO", "C")])
    )

    assert len(filtered_frame) == 32
    assert filtered_frame["N Components"].max() == 1


def test_validate_environment():
    # Ensure a valid schema passes
    FilterByEnvironmentsSchema(
        per_component_environments={1: [[ChemicalEnvironment.Alcohol]]},
    )
    FilterByEnvironmentsSchema(
        environments=[ChemicalEnvironment.Alcohol],
        at_least_one_environment=True,
        strictly_specified_environments=False,
    )
    FilterByEnvironmentsSchema(
        environments=[ChemicalEnvironment.Alcohol],
        at_least_one_environment=False,
        strictly_specified_environments=True,
    )

    # Test that an exception is raised when mutually exclusive options
    # are provided.
    with pytest.raises(ValidationError):
        FilterByEnvironmentsSchema(
            per_component_environments={1: [[ChemicalEnvironment.Alcohol]]},
            environments=[ChemicalEnvironment.Alcohol],
        )

    with pytest.raises(ValidationError):
        FilterByEnvironmentsSchema(
            environments=[ChemicalEnvironment.Alcohol],
            at_least_one_environment=True,
            strictly_specified_environments=True,
        )

    # Test that the validation logic which checks the correct number of
    # environment lists have been provided to per_component_environments
    with pytest.raises(ValidationError):
        FilterByEnvironmentsSchema(
            per_component_environments={1: []},
        )

    with pytest.raises(ValidationError):
        FilterByEnvironmentsSchema(
            per_component_environments={2: [[ChemicalEnvironment.Alcohol]]},
        )


def test_filter_by_environment_list():
    """Test that the ``FilterByEnvironments`` filter works well with the
    ``environments`` schema option"""

    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(
        _build_entry("O"),
        _build_entry("C"),
        _build_entry("C", "O"),
        _build_entry("O", "CC(=O)CC=O"),
        _build_entry("CC(=O)CC=O", "O"),
    )

    data_frame = data_set.to_pandas()

    # Retain only aqueous functionality
    filtered_frame = FilterByEnvironments.apply(
        data_frame,
        FilterByEnvironmentsSchema(
            environments=[ChemicalEnvironment.Aqueous], at_least_one_environment=True
        ),
    )

    assert len(filtered_frame) == 1
    assert filtered_frame["N Components"].max() == 1
    assert {*filtered_frame["Component 1"].unique()} == {"O"}

    # Retain both aqueous and aldehyde functionality but not strictly
    filtered_frame = FilterByEnvironments.apply(
        data_frame,
        FilterByEnvironmentsSchema(
            environments=[ChemicalEnvironment.Aqueous, ChemicalEnvironment.Aldehyde],
            at_least_one_environment=True,
        ),
    )

    assert len(filtered_frame) == 3

    assert filtered_frame["N Components"].min() == 1
    assert filtered_frame["N Components"].max() == 2

    pure_data = filtered_frame[filtered_frame["N Components"] == 1]
    binary_data = filtered_frame[filtered_frame["N Components"] == 2]

    assert len(pure_data) == 1
    assert {*pure_data["Component 1"].unique()} == {"O"}

    assert len(binary_data) == 2

    assert {
        *binary_data["Component 1"].unique(),
        *binary_data["Component 2"].unique(),
    } == {"CC(=O)CC=O", "O"}

    # Ensure enforcing the strict behaviour correctly filters out the
    # combined aldehyde and ketone functionality when only aldehyde and
    # aqueous is permitted.
    filtered_frame = FilterByEnvironments.apply(
        data_frame,
        FilterByEnvironmentsSchema(
            environments=[ChemicalEnvironment.Aqueous, ChemicalEnvironment.Aldehyde],
            at_least_one_environment=False,
            strictly_specified_environments=True,
        ),
    )

    assert len(filtered_frame) == 1
    assert filtered_frame["N Components"].max() == 1
    assert {*filtered_frame["Component 1"].unique()} == {"O"}


def test_filter_by_environment_per_component():
    """Test that the ``FilterByEnvironments`` filter works well with the
    ``per_component_environments`` schema option"""

    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(
        _build_entry("O"),
        _build_entry("C"),
        _build_entry("C", "O"),
        _build_entry("O", "CC(=O)CC=O"),
        _build_entry("CC(=O)CC=O", "O"),
    )

    data_frame = data_set.to_pandas()

    # Retain only aqueous functionality
    filtered_frame = FilterByEnvironments.apply(
        data_frame,
        FilterByEnvironmentsSchema(
            per_component_environments={
                1: [[ChemicalEnvironment.Aqueous]],
                2: [[ChemicalEnvironment.Aqueous], [ChemicalEnvironment.Aqueous]],
            },
            at_least_one_environment=True,
        ),
    )

    assert len(filtered_frame) == 1
    assert filtered_frame["N Components"].max() == 1
    assert {*filtered_frame["Component 1"].unique()} == {"O"}

    # Retain any pure component data, and only aqueous aldehyde mixture data.
    filtered_frame = FilterByEnvironments.apply(
        data_frame,
        FilterByEnvironmentsSchema(
            per_component_environments={
                2: [[ChemicalEnvironment.Aldehyde], [ChemicalEnvironment.Aqueous]]
            },
            at_least_one_environment=True,
        ),
    )
    assert len(filtered_frame) == 4

    assert filtered_frame["N Components"].min() == 1
    assert filtered_frame["N Components"].max() == 2

    pure_data = filtered_frame[filtered_frame["N Components"] == 1]
    binary_data = filtered_frame[filtered_frame["N Components"] == 2]

    assert len(pure_data) == 2
    assert {*pure_data["Component 1"].unique()} == {"O", "C"}

    assert len(binary_data) == 2

    assert {
        *binary_data["Component 1"].unique(),
        *binary_data["Component 2"].unique(),
    } == {"CC(=O)CC=O", "O"}

    # Repeat the last test but this time make the filtering strict.
    filtered_frame = FilterByEnvironments.apply(
        data_frame,
        FilterByEnvironmentsSchema(
            per_component_environments={
                2: [[ChemicalEnvironment.Aldehyde], [ChemicalEnvironment.Aqueous]]
            },
            at_least_one_environment=False,
            strictly_specified_environments=True,
        ),
    )
    assert len(filtered_frame) == 2

    assert filtered_frame["N Components"].max() == 1
    assert {*filtered_frame["Component 1"].unique()} == {"O", "C"}

    filtered_frame = FilterByEnvironments.apply(
        data_frame,
        FilterByEnvironmentsSchema(
            per_component_environments={
                2: [
                    [
                        ChemicalEnvironment.Aldehyde,
                        ChemicalEnvironment.Ketone,
                        ChemicalEnvironment.Carbonyl,
                    ],
                    [ChemicalEnvironment.Aqueous],
                ]
            },
            at_least_one_environment=False,
            strictly_specified_environments=True,
        ),
    )
    assert len(filtered_frame) == 4

    assert filtered_frame["N Components"].min() == 1
    assert filtered_frame["N Components"].max() == 2

    pure_data = filtered_frame[filtered_frame["N Components"] == 1]
    binary_data = filtered_frame[filtered_frame["N Components"] == 2]

    assert len(pure_data) == 2
    assert {*pure_data["Component 1"].unique()} == {"O", "C"}

    assert len(binary_data) == 2


def test_curation_does_not_alter_precision():
    """See issue #629"""
    # most faithful in-memory representation of how pandas sees missing data in CSVs
    nan = numpy.nan

    # copy-pasted from dataset.csv in linked issue
    data_frame = pandas.DataFrame.from_dict(
        {
            "Unnamed: 0": {0: 267, 1: 268},
            "Id": {
                0: "4031d37fd75649b7b143b42e62ca3338",
                1: "eeaf958a3e8d4ee1931bfb8a5258ef49",
            },
            "Temperature (K)": {0: 298.15, 1: 298.15},
            "Pressure (kPa)": {0: 101.0, 1: 101.0},
            "Phase": {0: "Liquid", 1: "Liquid"},
            "N Components": {0: 2, 1: 2},
            "Component 1": {0: "NCCO", 1: "NCCO"},
            "Role 1": {0: "Solvent", 1: "Solvent"},
            "Mole Fraction 1": {0: 0.7912, 1: 0.2112},
            "Density Value (g / ml)": {0: 1.002592, 1: 0.9825660000000004},
            "Density Uncertainty (g / ml)": {0: 0.000101, 1: 0.0001015},
            "Source": {0: "10.1021/je400184t", 1: "10.1021/je400184t"},
            "Component 2": {0: "c1ccncc1", 1: "c1ccncc1"},
            "Role 2": {0: "Solvent", 1: "Solvent"},
            "Mole Fraction 2": {0: 0.2088, 1: 0.7888},
            "EnthalpyOfMixing Value (kJ / mol)": {0: nan, 1: nan},
            "EnthalpyOfMixing Uncertainty (kJ / mol)": {0: nan, 1: nan},
            "Exact Amount 1": {0: nan, 1: nan},
            "Exact Amount 2": {0: nan, 1: nan},
        }
    )

    filtered = CurationWorkflow.apply(
        data_frame,
        CurationWorkflowSchema(
            component_schemas=[
                FilterDuplicatesSchema(
                    mole_fraction_precision=2,
                ),
            ]
        ),
    )

    assert list(data_frame["Mole Fraction 1"]) == list(filtered["Mole Fraction 1"])
