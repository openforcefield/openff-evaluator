"""
Units tests for openff.evaluator.datasets
"""
import json

import pytest

from openff.evaluator import unit
from openff.evaluator.datasets import (
    CalculationSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.properties import (
    Density,
    DielectricConstant,
    EnthalpyOfMixing,
    ExcessMolarVolume,
)
from openff.evaluator.substances import Component, MoleFraction, Substance
from openff.evaluator.tests.utils import (
    create_dummy_property,
    create_filterable_data_set,
)
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.serialization import TypedJSONEncoder


def test_physical_property_state_methods():

    dummy_property = create_dummy_property(Density)
    property_state = dummy_property.__getstate__()

    recreated_property = Density()
    recreated_property.__setstate__(property_state)

    recreated_state = recreated_property.__getstate__()

    original_json = json.dumps(property_state, cls=TypedJSONEncoder)
    recreated_json = json.dumps(recreated_state, cls=TypedJSONEncoder)

    assert original_json == recreated_json


def test_physical_property_id_generation():

    dummy_property_1 = create_dummy_property(Density)
    dummy_property_2 = create_dummy_property(Density)

    assert dummy_property_1.id != dummy_property_2.id


def test_serialization():
    """A test to ensure that data sets are JSON serializable."""

    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(create_dummy_property(Density))

    data_set_json = data_set.json()

    parsed_data_set = PhysicalPropertyDataSet.parse_json(data_set_json)
    assert len(data_set) == len(parsed_data_set)

    parsed_data_set_json = parsed_data_set.json()
    assert parsed_data_set_json == data_set_json


def test_to_pandas():
    """A test to ensure that data sets are convertable to pandas objects."""

    source = CalculationSource("Dummy", {})

    pure_substance = Substance.from_components("C")
    binary_substance = Substance.from_components("C", "O")

    data_set = PhysicalPropertyDataSet()

    for temperature in [298 * unit.kelvin, 300 * unit.kelvin, 302 * unit.kelvin]:

        thermodynamic_state = ThermodynamicState(
            temperature=temperature, pressure=1.0 * unit.atmosphere
        )

        density_property = Density(
            thermodynamic_state=thermodynamic_state,
            phase=PropertyPhase.Liquid,
            substance=pure_substance,
            value=1 * unit.gram / unit.milliliter,
            uncertainty=0.11 * unit.gram / unit.milliliter,
            source=source,
        )

        dielectric_property = DielectricConstant(
            thermodynamic_state=thermodynamic_state,
            phase=PropertyPhase.Liquid,
            substance=pure_substance,
            value=1 * unit.dimensionless,
            uncertainty=0.11 * unit.dimensionless,
            source=source,
        )

        data_set.add_properties(density_property)
        data_set.add_properties(dielectric_property)

    for temperature in [298 * unit.kelvin, 300 * unit.kelvin, 302 * unit.kelvin]:

        thermodynamic_state = ThermodynamicState(
            temperature=temperature, pressure=1.0 * unit.atmosphere
        )

        enthalpy_property = EnthalpyOfMixing(
            thermodynamic_state=thermodynamic_state,
            phase=PropertyPhase.Liquid,
            substance=binary_substance,
            value=1 * unit.kilojoules / unit.mole,
            uncertainty=0.11 * unit.kilojoules / unit.mole,
            source=source,
        )

        excess_property = ExcessMolarVolume(
            thermodynamic_state=thermodynamic_state,
            phase=PropertyPhase.Liquid,
            substance=binary_substance,
            value=1 * unit.meter ** 3 / unit.mole,
            uncertainty=0.11 * unit.meter ** 3 / unit.mole,
            source=source,
        )

        data_set.add_properties(enthalpy_property)
        data_set.add_properties(excess_property)

    data_set_pandas = data_set.to_pandas()

    required_columns = [
        "Id",
        "Temperature (K)",
        "Pressure (kPa)",
        "Phase",
        "N Components",
        "Source",
        "Component 1",
        "Role 1",
        "Mole Fraction 1",
        "Exact Amount 1",
        "Component 2",
        "Role 2",
        "Mole Fraction 2",
        "Exact Amount 2",
    ]

    assert all(x in data_set_pandas for x in required_columns)

    assert data_set_pandas is not None
    assert data_set_pandas.shape == (12, 21)

    data_set_without_na = data_set_pandas.dropna(axis=1, how="all")
    assert data_set_without_na.shape == (12, 19)


def test_sources_substances():

    physical_property = create_dummy_property(Density)

    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(physical_property)

    assert next(iter(data_set.sources)) == physical_property.source
    assert next(iter(data_set.substances)) == physical_property.substance


def test_properties_by_type():

    density = create_dummy_property(Density)
    dielectric = create_dummy_property(DielectricConstant)

    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(density, dielectric)

    densities = [x for x in data_set.properties_by_type("Density")]
    assert len(densities) == 1
    assert densities[0] == density

    dielectrics = [x for x in data_set.properties_by_type("DielectricConstant")]
    assert len(dielectrics) == 1
    assert dielectrics[0] == dielectric


def test_filter_by_property_types():
    """A test to ensure that data sets may be filtered by property type."""

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_property_types("Density")

    assert len(dummy_data_set) == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_property_types("Density", "DielectricConstant")

    assert len(dummy_data_set) == 2


def test_filter_by_phases():
    """A test to ensure that data sets may be filtered by phases."""

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_phases(phases=PropertyPhase.Liquid)

    assert len(dummy_data_set) == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_phases(
        phases=PropertyPhase(PropertyPhase.Liquid | PropertyPhase.Solid)
    )

    assert len(dummy_data_set) == 2

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_phases(
        phases=PropertyPhase(
            PropertyPhase.Liquid | PropertyPhase.Solid | PropertyPhase.Gas
        )
    )

    assert len(dummy_data_set) == 3


def test_filter_by_temperature():
    """A test to ensure that data sets may be filtered by temperature."""

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_temperature(
        min_temperature=287 * unit.kelvin, max_temperature=289 * unit.kelvin
    )

    assert len(dummy_data_set) == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_temperature(
        min_temperature=287 * unit.kelvin, max_temperature=299 * unit.kelvin
    )

    assert len(dummy_data_set) == 2

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_temperature(
        min_temperature=287 * unit.kelvin, max_temperature=309 * unit.kelvin
    )

    assert len(dummy_data_set) == 3


def test_filter_by_pressure():
    """A test to ensure that data sets may be filtered by pressure."""

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_pressure(
        min_pressure=0.4 * unit.atmosphere, max_pressure=0.6 * unit.atmosphere
    )

    assert len(dummy_data_set) == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_pressure(
        min_pressure=0.4 * unit.atmosphere, max_pressure=1.1 * unit.atmosphere
    )

    assert len(dummy_data_set) == 2

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_pressure(
        min_pressure=0.4 * unit.atmosphere, max_pressure=1.6 * unit.atmosphere
    )

    assert len(dummy_data_set) == 3


def test_filter_by_components():
    """A test to ensure that data sets may be filtered by the number of components."""

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_components(number_of_components=1)

    assert len(dummy_data_set) == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_components(number_of_components=2)

    assert len(dummy_data_set) == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_components(number_of_components=3)

    assert len(dummy_data_set) == 1


def test_filter_by_elements():
    """A test to ensure that data sets may be filtered by which elements their
    measured properties contain."""

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_elements("H", "C")

    assert len(dummy_data_set) == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_elements("H", "C", "N")

    assert len(dummy_data_set) == 2

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_elements("H", "C", "N", "O")

    assert len(dummy_data_set) == 3


def test_filter_by_smiles():
    """A test to ensure that data sets may be filtered by which smiles their
    measured properties contain."""

    methanol_substance = Substance()
    methanol_substance.add_component(Component("CO"), MoleFraction(1.0))

    ethanol_substance = Substance()
    ethanol_substance.add_component(Component("CCO"), MoleFraction(1.0))

    property_a = create_dummy_property(Density)
    property_a.substance = methanol_substance

    property_b = create_dummy_property(Density)
    property_b.substance = ethanol_substance

    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(property_a, property_b)

    data_set.filter_by_smiles("CO")

    assert len(data_set) == 1
    assert methanol_substance in data_set.substances
    assert ethanol_substance not in data_set.substances


def test_phase_from_string():

    assert PropertyPhase.from_string("") == PropertyPhase.Undefined

    phase_enums = [
        PropertyPhase.Undefined,
        PropertyPhase.Solid,
        PropertyPhase.Liquid,
        PropertyPhase.Gas,
        PropertyPhase.Solid | PropertyPhase.Liquid,
        PropertyPhase.Solid | PropertyPhase.Gas,
        PropertyPhase.Liquid | PropertyPhase.Gas,
        PropertyPhase.Solid | PropertyPhase.Liquid | PropertyPhase.Gas,
    ]

    assert all(x == PropertyPhase.from_string(str(x)) for x in phase_enums)


def test_validate_data_set():

    valid_property = Density(
        ThermodynamicState(298 * unit.kelvin, 1 * unit.atmosphere),
        PropertyPhase.Liquid,
        Substance.from_components("O"),
        0.0 * unit.gram / unit.milliliter,
        0.0 * unit.gram / unit.milliliter,
    )

    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(valid_property)

    data_set.validate()

    invalid_property = Density(
        ThermodynamicState(-1 * unit.kelvin, 1 * unit.atmosphere),
        PropertyPhase.Liquid,
        Substance.from_components("O"),
        0.0 * unit.gram / unit.milliliter,
        0.0 * unit.gram / unit.milliliter,
    )

    with pytest.raises(AssertionError):
        data_set.add_properties(invalid_property)

    data_set.add_properties(invalid_property, validate=False)

    with pytest.raises(AssertionError):
        data_set.validate()
