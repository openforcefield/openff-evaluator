"""
Units tests for openff.evaluator.thermodynamics
"""
import pint
import pytest
from openff.units import unit

from openff.evaluator.thermodynamics import ThermodynamicState


def test_state_equality():

    state_a = ThermodynamicState(
        temperature=1.0 * unit.kelvin, pressure=1.0 * unit.pascals
    )

    state_b = ThermodynamicState(
        temperature=1.0004 * unit.kelvin, pressure=1.0004 * unit.pascals
    )

    assert state_a == state_b

    state_c = ThermodynamicState(
        temperature=1.001 * unit.kelvin, pressure=1.001 * unit.pascals
    )

    assert state_a != state_c
    assert hash(state_a) != hash(state_c)

    state_d = ThermodynamicState(
        temperature=1.0005 * unit.kelvin, pressure=1.0005 * unit.pascals
    )

    assert state_a == state_d
    assert state_c != state_d

    state_e = ThermodynamicState(
        temperature=0.9995 * unit.kelvin, pressure=0.9995 * unit.pascals
    )

    assert state_a == state_e


@pytest.mark.parametrize(
    "state",
    [
        ThermodynamicState(temperature=1.0 * unit.kelvin),
        ThermodynamicState(temperature=1.0 * unit.kelvin, pressure=1.0 * unit.pascals),
    ],
)
def test_state_valid_checks(state):
    state.validate()


@pytest.mark.parametrize(
    "state",
    [
        ThermodynamicState(),
        ThermodynamicState(temperature=1.0 * unit.pascals),
        ThermodynamicState(temperature=1.0 * unit.pascals, pressure=1.0 * unit.kelvin),
        ThermodynamicState(temperature=-1.0 * unit.kelvin),
    ],
)
def test_state_invalid_checks(state):

    with pytest.raises((ValueError, AssertionError, pint.errors.DimensionalityError)):
        state.validate()
