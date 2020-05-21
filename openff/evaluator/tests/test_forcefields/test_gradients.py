import numpy as np
import pytest

from openff.evaluator import unit
from openff.evaluator.forcefield import ParameterGradient, ParameterGradientKey


def test_gradient_addition():

    gradient_a = ParameterGradient(
        ParameterGradientKey("vdW", "[#1:1]", "epsilon"), 1.0 * unit.kelvin
    )
    gradient_b = ParameterGradient(
        ParameterGradientKey("vdW", "[#1:1]", "epsilon"), 2.0 * unit.kelvin
    )

    result = gradient_a + gradient_b
    assert np.isclose(result.value.to(unit.kelvin).magnitude, 3.0)

    gradient_c = ParameterGradient(
        ParameterGradientKey("vdW", "[#6:1]", "epsilon"), 1.0 * unit.kelvin
    )

    with pytest.raises(ValueError):
        gradient_a + gradient_c

    with pytest.raises(ValueError):
        gradient_a + 1.0


def test_gradient_subtraction():

    gradient_a = ParameterGradient(
        ParameterGradientKey("vdW", "[#1:1]", "epsilon"), 1.0 * unit.kelvin
    )
    gradient_b = ParameterGradient(
        ParameterGradientKey("vdW", "[#1:1]", "epsilon"), 2.0 * unit.kelvin
    )

    result = gradient_a - gradient_b
    assert np.isclose(result.value.to(unit.kelvin).magnitude, -1.0)

    result = gradient_b - gradient_a
    assert np.isclose(result.value.to(unit.kelvin).magnitude, 1.0)

    gradient_c = ParameterGradient(
        ParameterGradientKey("vdW", "[#6:1]", "epsilon"), 1.0 * unit.kelvin
    )

    with pytest.raises(ValueError):
        gradient_a - gradient_c

    with pytest.raises(ValueError):
        gradient_c - gradient_a

    with pytest.raises(ValueError):
        gradient_a - 1.0


def test_gradient_multiplication():

    gradient_a = ParameterGradient(
        ParameterGradientKey("vdW", "[#1:1]", "epsilon"), 1.0 * unit.kelvin
    )

    result = gradient_a * 2.0
    assert np.isclose(result.value.to(unit.kelvin).magnitude, 2.0)

    result = 3.0 * gradient_a
    assert np.isclose(result.value.to(unit.kelvin).magnitude, 3.0)

    gradient_c = ParameterGradient(
        ParameterGradientKey("vdW", "[#1:1]", "epsilon"), 1.0 * unit.kelvin
    )

    with pytest.raises(ValueError):
        gradient_a * gradient_c


def test_gradient_division():

    gradient_a = ParameterGradient(
        ParameterGradientKey("vdW", "[#1:1]", "epsilon"), 2.0 * unit.kelvin
    )

    result = gradient_a / 2.0
    assert np.isclose(result.value.to(unit.kelvin).magnitude, 1.0)

    gradient_c = ParameterGradient(
        ParameterGradientKey("vdW", "[#1:1]", "epsilon"), 1.0 * unit.kelvin
    )

    with pytest.raises(ValueError):
        gradient_a / gradient_c
