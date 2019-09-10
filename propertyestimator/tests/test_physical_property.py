"""
Units tests for propertyestimator.layers.simulation
"""
import json
import numpy as np
import pytest

from propertyestimator import unit
from propertyestimator.properties import ParameterGradient, ParameterGradientKey
from propertyestimator.properties.density import Density
from propertyestimator.utils.serialization import TypedJSONEncoder

from propertyestimator.tests.utils import create_dummy_property


def test_physical_property_state_methods():

    dummy_property = create_dummy_property(Density)
    property_state = dummy_property.__getstate__()

    recreated_property = Density()
    recreated_property.__setstate__(property_state)

    recreated_state = recreated_property.__getstate__()

    original_json = json.dumps(property_state, cls=TypedJSONEncoder)
    recreated_json = json.dumps(recreated_state, cls=TypedJSONEncoder)

    assert original_json == recreated_json


def test_gradient_addition():

    gradient_a = ParameterGradient(ParameterGradientKey('vdW', '[#1:1]', 'epsilon'), 1.0 * unit.kelvin)
    gradient_b = ParameterGradient(ParameterGradientKey('vdW', '[#1:1]', 'epsilon'), 2.0 * unit.kelvin)

    result = gradient_a + gradient_b
    assert np.isclose(result.value.to(unit.kelvin).magnitude, 3.0)

    gradient_c = ParameterGradient(ParameterGradientKey('vdW', '[#6:1]', 'epsilon'), 1.0 * unit.kelvin)

    with pytest.raises(ValueError):
        gradient_a + gradient_c

    with pytest.raises(ValueError):
        gradient_a + 1.0


def test_gradient_subtraction():

    gradient_a = ParameterGradient(ParameterGradientKey('vdW', '[#1:1]', 'epsilon'), 1.0 * unit.kelvin)
    gradient_b = ParameterGradient(ParameterGradientKey('vdW', '[#1:1]', 'epsilon'), 2.0 * unit.kelvin)

    result = gradient_a - gradient_b
    assert np.isclose(result.value.to(unit.kelvin).magnitude, -1.0)

    result = gradient_b - gradient_a
    assert np.isclose(result.value.to(unit.kelvin).magnitude, 1.0)

    gradient_c = ParameterGradient(ParameterGradientKey('vdW', '[#6:1]', 'epsilon'), 1.0 * unit.kelvin)

    with pytest.raises(ValueError):
        gradient_a - gradient_c

    with pytest.raises(ValueError):
        gradient_c - gradient_a

    with pytest.raises(ValueError):
        gradient_a - 1.0


def test_gradient_multiplication():

    gradient_a = ParameterGradient(ParameterGradientKey('vdW', '[#1:1]', 'epsilon'), 1.0 * unit.kelvin)

    result = gradient_a * 2.0
    assert np.isclose(result.value.to(unit.kelvin).magnitude, 2.0)

    result = 3.0 * gradient_a
    assert np.isclose(result.value.to(unit.kelvin).magnitude, 3.0)

    gradient_c = ParameterGradient(ParameterGradientKey('vdW', '[#1:1]', 'epsilon'), 1.0 * unit.kelvin)

    with pytest.raises(ValueError):
        gradient_a * gradient_c


def test_gradient_division():

    gradient_a = ParameterGradient(ParameterGradientKey('vdW', '[#1:1]', 'epsilon'), 2.0 * unit.kelvin)

    result = gradient_a / 2.0
    assert np.isclose(result.value.to(unit.kelvin).magnitude, 1.0)

    gradient_c = ParameterGradient(ParameterGradientKey('vdW', '[#1:1]', 'epsilon'), 1.0 * unit.kelvin)

    with pytest.raises(ValueError):
        gradient_a / gradient_c
