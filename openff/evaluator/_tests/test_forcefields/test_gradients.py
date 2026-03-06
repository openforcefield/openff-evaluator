import numpy as np
import pytest
from openff.units import unit

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


def test_parameter_gradient_key_virtual_site_metadata_roundtrip():
    key_a = ParameterGradientKey(
        "VirtualSites",
        "[#1:2]-[#8X2H2+0:1]-[#1:3]",
        "distance",
        virtual_site_type="BondCharge",
        virtual_site_name="EP1",
        virtual_site_match="all_permutations",
    )

    state = key_a.__getstate__()
    key_b = ParameterGradientKey()
    key_b.__setstate__(state)

    assert key_a == key_b
    assert hash(key_a) == hash(key_b)
