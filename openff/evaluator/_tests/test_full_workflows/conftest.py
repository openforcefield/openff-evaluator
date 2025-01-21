import pytest
from openff.units import unit

from openff.evaluator.datasets import (
    MeasurementSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.substances import Substance
from openff.evaluator.thermodynamics import ThermodynamicState


@pytest.fixture
def dummy_enthalpy_of_mixing():
    thermodynamic_state = ThermodynamicState(
        temperature=298.15 * unit.kelvin,
        pressure=101.325 * unit.kilopascal,
    )

    return EnthalpyOfMixing(
        thermodynamic_state=thermodynamic_state,
        phase=PropertyPhase.Liquid,
        value=1.0 * EnthalpyOfMixing.default_unit(),
        uncertainty=1.0 * EnthalpyOfMixing.default_unit(),
        source=MeasurementSource(doi=" "),
        substance=Substance.from_components("CCCO", "O"),
    )


@pytest.fixture
def dummy_density():
    thermodynamic_state = ThermodynamicState(
        temperature=298.15 * unit.kelvin,
        pressure=101.325 * unit.kilopascal,
    )

    return Density(
        thermodynamic_state=thermodynamic_state,
        phase=PropertyPhase.Liquid,
        value=1.0 * Density.default_unit(),
        uncertainty=1.0 * Density.default_unit(),
        source=MeasurementSource(doi=" "),
        substance=Substance.from_components("CCCO"),
    )


@pytest.fixture
def dummy_dataset(dummy_density, dummy_enthalpy_of_mixing):
    dataset = PhysicalPropertyDataSet()
    dataset.add_properties(dummy_density, dummy_enthalpy_of_mixing)

    for i, prop in enumerate(dataset.properties):
        prop.id = f"{i}"
    return dataset
