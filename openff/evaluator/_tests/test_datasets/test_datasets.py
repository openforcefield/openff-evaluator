"""
Units tests for openff.evaluator.datasets
"""

import json

import numpy
import pytest
from openff.units import unit

from openff.evaluator._tests.utils import create_dummy_property
from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.datasets import (
    CalculationSource,
    MeasurementSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.properties import (
    Density,
    DielectricConstant,
    EnthalpyOfMixing,
    EnthalpyOfVaporization,
    ExcessMolarVolume,
)
from openff.evaluator.substances import Substance
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
            value=1 * unit.meter**3 / unit.mole,
            uncertainty=0.11 * unit.meter**3 / unit.mole,
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
    assert data_set_pandas.shape == (12, 22)

    data_set_without_na = data_set_pandas.dropna(axis=1, how="all")
    assert data_set_without_na.shape == (12, 20)


def test_from_pandas():
    """A test to ensure that data sets may be created from pandas objects."""

    thermodynamic_state = ThermodynamicState(
        temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
    )

    original_data_set = PhysicalPropertyDataSet()
    original_data_set.add_properties(
        Density(
            thermodynamic_state=thermodynamic_state,
            phase=PropertyPhase.Liquid,
            substance=Substance.from_components("CO", "O"),
            value=1.0 * unit.kilogram / unit.meter**3,
            uncertainty=1.0 * unit.kilogram / unit.meter**3,
            source=MeasurementSource(doi="10.5281/zenodo.596537"),
        ),
        EnthalpyOfVaporization(
            thermodynamic_state=thermodynamic_state,
            phase=PropertyPhase.from_string("Liquid + Gas"),
            substance=Substance.from_components("C"),
            value=2.0 * unit.kilojoule / unit.mole,
            source=MeasurementSource(reference="2"),
        ),
        DielectricConstant(
            thermodynamic_state=thermodynamic_state,
            phase=PropertyPhase.Liquid,
            substance=Substance.from_components("C"),
            value=3.0 * unit.dimensionless,
            source=MeasurementSource(reference="3"),
        ),
    )

    data_frame = original_data_set.to_pandas()

    recreated_data_set = PhysicalPropertyDataSet.from_pandas(data_frame)
    assert len(original_data_set) == len(recreated_data_set)

    for original_property in original_data_set:
        recreated_property = next(
            x for x in recreated_data_set if x.id == original_property.id
        )

        assert (
            original_property.thermodynamic_state
            == recreated_property.thermodynamic_state
        )
        assert original_property.phase == recreated_property.phase
        assert original_property.substance == recreated_property.substance
        assert numpy.isclose(original_property.value, recreated_property.value)

        if original_property.uncertainty == UNDEFINED:
            assert original_property.uncertainty == recreated_property.uncertainty
        else:
            assert numpy.isclose(
                original_property.uncertainty, recreated_property.uncertainty
            )

        assert original_property.source.doi == recreated_property.source.doi
        assert original_property.source.reference == recreated_property.source.reference


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
