"""
Units tests for openff.evaluator.utils.observables
"""

import json
from typing import List, Tuple, Type, Union

import numpy
import pytest
from openff.units import unit

from openff.evaluator._tests.utils import does_not_raise
from openff.evaluator.forcefield import ParameterGradient, ParameterGradientKey
from openff.evaluator.utils import get_data_filename
from openff.evaluator.utils.observables import (
    Observable,
    ObservableArray,
    ObservableFrame,
    ObservableType,
    bootstrap,
)
from openff.evaluator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder

ValueType = Union[float, int, unit.Quantity, unit.Measurement, numpy.ndarray]


def _compare_observables(
    observable_a: Union[Observable, ObservableArray],
    observable_b: Union[Observable, ObservableArray],
):
    assert isinstance(observable_a, type(observable_b))

    assert isinstance(observable_a.value.magnitude, type(observable_b.value.magnitude))
    assert numpy.allclose(observable_a.value, observable_b.value)

    if isinstance(observable_a.value.magnitude, numpy.ndarray):
        assert observable_a.value.shape == observable_b.value.shape

    if isinstance(observable_a, Observable):
        assert numpy.isclose(observable_a.error, observable_b.error)

    observable_a_gradients = {
        gradient.key: gradient for gradient in observable_a.gradients
    }
    observable_b_gradients = {
        gradient.key: gradient for gradient in observable_b.gradients
    }

    assert {*observable_a_gradients} == {*observable_b_gradients}

    for gradient_key in observable_a_gradients:
        gradient_a = observable_a_gradients[gradient_key]
        gradient_b = observable_b_gradients[gradient_key]

        assert isinstance(gradient_a.value.magnitude, type(gradient_b.value.magnitude))
        assert numpy.allclose(gradient_a.value, gradient_b.value)

        if isinstance(gradient_a.value.magnitude, numpy.ndarray):
            assert gradient_a.value.shape == gradient_b.value.shape


def _mock_observable(
    value: ValueType,
    gradient_values: List[Tuple[str, str, str, ValueType]],
    object_type: Union[Type[Observable], Type[ObservableArray]],
):
    return object_type(
        value=value,
        gradients=[
            ParameterGradient(
                key=ParameterGradientKey(tag, smirks, attribute),
                value=value * unit.kelvin,
            )
            for tag, smirks, attribute, value in gradient_values
        ],
    )


@pytest.mark.parametrize(
    "value, gradient_values, expected_value, expected_gradient_values",
    [
        (
            numpy.ones(1) * unit.kelvin,
            [numpy.ones(1) * unit.kelvin],
            numpy.ones((1, 1)) * unit.kelvin,
            [numpy.ones((1, 1)) * unit.kelvin],
        ),
        (
            1.0 * unit.kelvin,
            [numpy.ones(1) * unit.kelvin],
            numpy.ones((1, 1)) * unit.kelvin,
            [numpy.ones((1, 1)) * unit.kelvin],
        ),
        (
            numpy.ones(1) * unit.kelvin,
            [1.0 * unit.kelvin],
            numpy.ones((1, 1)) * unit.kelvin,
            [numpy.ones((1, 1)) * unit.kelvin],
        ),
        (
            numpy.ones(3) * unit.kelvin,
            [numpy.ones((3, 1)) * unit.kelvin],
            numpy.ones((3, 1)) * unit.kelvin,
            [numpy.ones((3, 1)) * unit.kelvin],
        ),
        (
            numpy.ones((3, 1)) * unit.kelvin,
            [numpy.ones(3) * unit.kelvin],
            numpy.ones((3, 1)) * unit.kelvin,
            [numpy.ones((3, 1)) * unit.kelvin],
        ),
        (
            numpy.ones((2, 3)) * unit.kelvin,
            [numpy.ones((2, 3)) * unit.kelvin],
            numpy.ones((2, 3)) * unit.kelvin,
            [numpy.ones((2, 3)) * unit.kelvin],
        ),
    ],
)
def test_observable_array_valid_initializer(
    value: unit.Quantity,
    gradient_values: List[unit.Quantity],
    expected_value: unit.Quantity,
    expected_gradient_values: List[unit.Quantity],
):
    observable = ObservableArray(
        value,
        [
            ParameterGradient(
                key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                value=gradient_value,
            )
            for gradient_value in gradient_values
        ],
    )

    # noinspection PyUnresolvedReferences
    assert observable.value.shape == expected_value.shape
    assert numpy.allclose(observable.value, expected_value)

    assert all(
        observable.gradients[i].value.shape == expected_gradient_values[i].shape
        for i in range(len(expected_gradient_values))
    )
    assert all(
        numpy.allclose(observable.gradients[i].value, expected_gradient_values[i])
        for i in range(len(expected_gradient_values))
    )


@pytest.mark.parametrize(
    "value, gradients, expected_raises, expected_message",
    [
        (
            numpy.ones(1),
            [],
            pytest.raises(TypeError),
            "The value must be a unit-wrapped integer, float or numpy array.",
        ),
        (
            "str" * unit.kelvin,
            [],
            pytest.raises(TypeError),
            "The value must be a unit-wrapped integer, float or numpy array.",
        ),
        (
            numpy.ones((2, 2, 2)) * unit.kelvin,
            [],
            pytest.raises(ValueError),
            "The wrapped array must not contain more than two dimensions.",
        ),
        (
            None,
            [
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                    value=numpy.ones((2, 2)) * unit.kelvin,
                ),
            ],
            pytest.raises(ValueError),
            "A valid value must be provided.",
        ),
        (
            1.0 * unit.kelvin,
            [
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                    value=numpy.ones(1),
                ),
            ],
            pytest.raises(TypeError),
            "The gradient values must be unit-wrapped integers, floats or numpy arrays.",
        ),
        (
            1.0 * unit.kelvin,
            [
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                    value="str" * unit.kelvin,
                ),
            ],
            pytest.raises(TypeError),
            "The gradient values must be unit-wrapped integers, floats or numpy arrays.",
        ),
        (
            1.0 * unit.kelvin,
            [
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                    value=numpy.ones((2, 2, 2)) * unit.kelvin,
                ),
            ],
            pytest.raises(ValueError),
            "Gradient values must not contain more than two dimensions.",
        ),
        (
            numpy.ones((2, 1)) * unit.kelvin,
            [
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                    value=numpy.ones((1, 2)) * unit.kelvin,
                ),
            ],
            pytest.raises(ValueError),
            "Gradient values should be 1-dimensional to match the dimensionality of the "
            "value.",
        ),
        (
            numpy.ones((1, 2)) * unit.kelvin,
            [
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                    value=numpy.ones((2, 1)) * unit.kelvin,
                ),
            ],
            pytest.raises(ValueError),
            "Gradient values should be 2-dimensional to match the dimensionality of the "
            "value.",
        ),
        (
            numpy.ones((3, 2)) * unit.kelvin,
            [
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                    value=numpy.ones((2, 2)) * unit.kelvin,
                ),
            ],
            pytest.raises(ValueError),
            "Gradient values should have a length of 3 to match the length of the "
            "value.",
        ),
    ],
)
def test_observable_array_invalid_initializer(
    value, gradients, expected_raises, expected_message
):
    with expected_raises as error_info:
        ObservableArray(value, gradients)

    assert expected_message in str(error_info.value)


@pytest.mark.parametrize("value", [0.1, numpy.ones(1)])
def test_observable_array_round_trip(value):
    observable = ObservableArray(
        value=value * unit.kelvin,
        gradients=[
            ParameterGradient(
                key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                value=value * 2.0 * unit.kelvin,
            )
        ],
    )

    round_tripped: ObservableArray = json.loads(
        json.dumps(observable, cls=TypedJSONEncoder), cls=TypedJSONDecoder
    )

    assert isinstance(round_tripped, ObservableArray)

    assert numpy.isclose(observable.value, round_tripped.value)

    assert len(observable.gradients) == len(round_tripped.gradients)
    assert observable.gradients[0] == round_tripped.gradients[0]


def test_observable_array_subset():
    observable = ObservableArray(
        value=numpy.arange(4) * unit.kelvin,
        gradients=[
            ParameterGradient(
                key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                value=numpy.arange(4) * unit.kelvin,
            )
        ],
    )

    subset = observable.subset([1, 3])
    assert len(subset) == 2

    assert numpy.allclose(subset.value, numpy.array([[1.0], [3.0]]) * unit.kelvin)
    assert numpy.allclose(
        subset.gradients[0].value, numpy.array([[1.0], [3.0]]) * unit.kelvin
    )


def test_observable_array_join():
    gradient_unit = unit.mole / unit.kilojoule

    observables = [
        ObservableArray(
            value=(numpy.arange(2) + i * 2) * unit.kelvin,
            gradients=[
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                    value=(numpy.arange(2) + i * 2) * unit.kelvin * gradient_unit,
                )
            ],
        )
        for i in range(2)
    ]

    joined = ObservableArray.join(*observables)
    assert len(joined) == 4

    assert numpy.allclose(joined.value, numpy.arange(4).reshape(-1, 1) * unit.kelvin)
    assert numpy.allclose(
        joined.gradients[0].value,
        numpy.arange(4).reshape(-1, 1) * unit.kelvin * gradient_unit,
    )


def test_observable_array_join_single():
    gradient_unit = unit.mole / unit.kilojoule

    joined = ObservableArray.join(
        ObservableArray(
            value=(numpy.arange(2)) * unit.kelvin,
            gradients=[
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                    value=(numpy.arange(2)) * unit.kelvin * gradient_unit,
                )
            ],
        )
    )
    assert len(joined) == 2


def test_observable_array_len():
    assert len(ObservableArray(value=numpy.arange(5) * unit.kelvin)) == 5


@pytest.mark.parametrize(
    "observables, expected_raises, expected_message",
    [
        (
            [],
            pytest.raises(ValueError),
            "At least one observable must be provided.",
        ),
        (
            [
                ObservableArray(value=numpy.ones(1) * unit.kelvin),
                ObservableArray(value=numpy.ones(1) * unit.pascal),
            ],
            pytest.raises(ValueError),
            "The observables must all have compatible units.",
        ),
        (
            [
                ObservableArray(
                    value=numpy.ones(2) * unit.kelvin,
                    gradients=[
                        ParameterGradient(
                            key=ParameterGradientKey("vdW", "[#1:1]", "sigma"),
                            value=numpy.ones(2) * unit.kelvin / unit.angstrom,
                        )
                    ],
                ),
                ObservableArray(
                    value=numpy.ones(2) * unit.kelvin,
                    gradients=[
                        ParameterGradient(
                            key=ParameterGradientKey("vdW", "[#6:1]", "sigma"),
                            value=numpy.ones(2) * unit.kelvin / unit.angstrom,
                        )
                    ],
                ),
            ],
            pytest.raises(ValueError),
            "The observables must contain gradient information for the same "
            "parameters.",
        ),
        (
            [
                ObservableArray(
                    value=numpy.ones(2) * unit.kelvin,
                    gradients=[
                        ParameterGradient(
                            key=ParameterGradientKey("vdW", "[#6:1]", "sigma"),
                            value=numpy.ones(2) * unit.kelvin / unit.angstrom,
                        )
                    ],
                ),
                ObservableArray(
                    value=numpy.ones(2) * unit.kelvin,
                    gradients=[
                        ParameterGradient(
                            key=ParameterGradientKey("vdW", "[#6:1]", "sigma"),
                            value=numpy.ones(2) * unit.kelvin / unit.meter,
                        )
                    ],
                ),
            ],
            pytest.raises(ValueError),
            "The gradients of each of the observables must have the same units.",
        ),
    ],
)
def test_observables_join_fail(observables, expected_raises, expected_message):
    with expected_raises as error_info:
        ObservableArray.join(*observables)

    assert (
        expected_message is None
        and error_info is None
        or expected_message in str(error_info.value)
    )


@pytest.mark.parametrize(
    "value, gradients, expected_raises, expected_message",
    [
        (
            0.1 * unit.kelvin,
            [
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                    value=0.1 * unit.kelvin,
                )
            ],
            does_not_raise(),
            None,
        ),
        (
            (0.1 * unit.kelvin).plus_minus(0.2 * unit.kelvin),
            [
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                    value=0.1 * unit.kelvin,
                )
            ],
            does_not_raise(),
            None,
        ),
        (
            0.1,
            [],
            pytest.raises(TypeError),
            "The value must be either an `openff.evaluator.unit.Measurement` or "
            "an `openff.evaluator.unit.Quantity`.",
        ),
        (
            numpy.ones(3) * unit.kelvin,
            [],
            pytest.raises(TypeError),
            "The value must be a unit-wrapped integer or float.",
        ),
        (
            None,
            [
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"), value=0.1
                )
            ],
            pytest.raises(ValueError),
            "A valid value must be provided.",
        ),
        (
            (0.1 * unit.kelvin).plus_minus(0.2 * unit.kelvin),
            [
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"), value=0.1
                )
            ],
            pytest.raises(TypeError),
            "The gradient values must be unit-wrapped integers or floats.",
        ),
        (
            (0.1 * unit.kelvin).plus_minus(0.2 * unit.kelvin),
            [
                ParameterGradient(
                    key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                    value="str" * unit.kelvin,
                )
            ],
            pytest.raises(TypeError),
            "The gradient values must be unit-wrapped integers or floats.",
        ),
    ],
)
def test_observable_initializer(value, gradients, expected_raises, expected_message):
    with expected_raises as error_info:
        Observable(value, gradients)

    if expected_message is not None:
        assert expected_message in str(error_info.value)


def test_observable_round_trip():
    observable = Observable(
        value=(0.1 * unit.kelvin).plus_minus(0.2 * unit.kelvin),
        gradients=[
            ParameterGradient(
                key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                value=0.2 * unit.kelvin,
            )
        ],
    )

    round_tripped: Observable = json.loads(
        json.dumps(observable, cls=TypedJSONEncoder), cls=TypedJSONDecoder
    )

    assert isinstance(round_tripped, Observable)

    assert numpy.isclose(observable.value, round_tripped.value)
    assert numpy.isclose(observable.error, round_tripped.error)

    assert len(observable.gradients) == len(round_tripped.gradients)
    assert observable.gradients[0] == round_tripped.gradients[0]


@pytest.mark.parametrize(
    "value_a, value_b, expected_value",
    [
        observable_tuple
        for object_type in [Observable, ObservableArray]
        for observable_tuple in [
            (
                _mock_observable(
                    2.0 * unit.kelvin,
                    [
                        ("vdW", "[#6:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#1:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    4.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    6.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 6.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 6.0 * unit.kelvin),
                    ],
                    object_type,
                ),
            ),
            (
                2.0 * unit.kelvin,
                _mock_observable(
                    4.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    6.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
            ),
        ]
    ],
)
def test_add_observables(value_a, value_b, expected_value):
    _compare_observables(value_a + value_b, expected_value)
    _compare_observables(value_b + value_a, expected_value)


@pytest.mark.parametrize(
    "value_a, value_b, expected_value",
    [
        observable_tuple
        for object_type in [Observable, ObservableArray]
        for observable_tuple in [
            (
                _mock_observable(
                    2.0 * unit.kelvin,
                    [
                        ("vdW", "[#6:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#1:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    4.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    2.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", -2.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 2.0 * unit.kelvin),
                    ],
                    object_type,
                ),
            ),
            (
                _mock_observable(
                    2.0 * unit.kelvin,
                    [
                        ("vdW", "[#6:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#1:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                2.0 * unit.kelvin,
                _mock_observable(
                    0.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", -4.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", -2.0 * unit.kelvin),
                    ],
                    object_type,
                ),
            ),
            (
                2.0 * unit.kelvin,
                _mock_observable(
                    2.0 * unit.kelvin,
                    [
                        ("vdW", "[#6:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#1:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    0.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 4.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 2.0 * unit.kelvin),
                    ],
                    object_type,
                ),
            ),
        ]
    ],
)
def test_subtract_observables(value_a, value_b, expected_value):
    _compare_observables(value_b - value_a, expected_value)


@pytest.mark.parametrize(
    "value_a, value_b, expected_value",
    [
        observable_tuple
        for object_type in [Observable, ObservableArray]
        for observable_tuple in [
            (
                _mock_observable(
                    2.0 * unit.kelvin,
                    [
                        ("vdW", "[#6:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#1:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    4.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    8.0 * unit.kelvin**2,
                    [
                        ("vdW", "[#1:1]", "epsilon", 20.0 * unit.kelvin**2),
                        ("vdW", "[#6:1]", "epsilon", 16.0 * unit.kelvin**2),
                    ],
                    object_type,
                ),
            ),
            (
                2.0 * unit.kelvin,
                _mock_observable(
                    4.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    8.0 * unit.kelvin**2,
                    [
                        ("vdW", "[#1:1]", "epsilon", 4.0 * unit.kelvin**2),
                        ("vdW", "[#6:1]", "epsilon", 8.0 * unit.kelvin**2),
                    ],
                    object_type,
                ),
            ),
            (
                2.0,
                _mock_observable(
                    4.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    8.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 4.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 8.0 * unit.kelvin),
                    ],
                    object_type,
                ),
            ),
        ]
    ],
)
def test_multiply_observables(value_a, value_b, expected_value):
    _compare_observables(value_a * value_b, expected_value)
    _compare_observables(value_b * value_a, expected_value)


@pytest.mark.parametrize(
    "value_a, value_b, expected_value",
    [
        observable_tuple
        for object_type in [Observable, ObservableArray]
        for observable_tuple in [
            (
                _mock_observable(
                    4.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    2.0 * unit.kelvin,
                    [
                        ("vdW", "[#6:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#1:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    2.0 * unit.dimensionless,
                    [
                        ("vdW", "[#1:1]", "epsilon", -3.0 * unit.dimensionless),
                        ("vdW", "[#6:1]", "epsilon", 0.0 * unit.dimensionless),
                    ],
                    object_type,
                ),
            ),
            (
                _mock_observable(
                    4.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                2.0 * unit.kelvin,
                _mock_observable(
                    2.0 * unit.dimensionless,
                    [
                        ("vdW", "[#1:1]", "epsilon", 1.0 * unit.dimensionless),
                        ("vdW", "[#6:1]", "epsilon", 2.0 * unit.dimensionless),
                    ],
                    object_type,
                ),
            ),
            (
                2.0 * unit.kelvin,
                _mock_observable(
                    4.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    1.0 / 2.0 * unit.dimensionless,
                    [
                        ("vdW", "[#1:1]", "epsilon", -1.0 / 4.0 * unit.dimensionless),
                        ("vdW", "[#6:1]", "epsilon", -1.0 / 2.0 * unit.dimensionless),
                    ],
                    object_type,
                ),
            ),
            (
                _mock_observable(
                    4.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                2.0,
                _mock_observable(
                    2.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 1.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 2.0 * unit.kelvin),
                    ],
                    object_type,
                ),
            ),
            (
                2.0,
                _mock_observable(
                    4.0 * unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", 2.0 * unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", 4.0 * unit.kelvin),
                    ],
                    object_type,
                ),
                _mock_observable(
                    1.0 / 2.0 / unit.kelvin,
                    [
                        ("vdW", "[#1:1]", "epsilon", -1.0 / 4.0 / unit.kelvin),
                        ("vdW", "[#6:1]", "epsilon", -1.0 / 2.0 / unit.kelvin),
                    ],
                    object_type,
                ),
            ),
        ]
    ],
)
def test_divide_observables(value_a, value_b, expected_value):
    _compare_observables(value_a / value_b, expected_value)


@pytest.mark.parametrize(
    "observables",
    [
        {"Temperature": ObservableArray(value=numpy.ones(2) * unit.kelvin)},
        {
            ObservableType.Temperature: ObservableArray(
                value=numpy.ones(2) * unit.kelvin
            )
        },
        ObservableFrame(
            {
                ObservableType.Temperature: ObservableArray(
                    value=numpy.ones(2) * unit.kelvin
                )
            }
        ),
    ],
)
def test_frame_constructor(observables):
    observable_frame = ObservableFrame(observables)

    assert all(observable_type in observable_frame for observable_type in observables)
    assert all(
        observable_frame[observable_type] == observables[observable_type]
        for observable_type in observables
    )


def test_frame_round_trip():
    observable_frame = ObservableFrame(
        {"Temperature": ObservableArray(value=numpy.ones(2) * unit.kelvin)}
    )

    round_tripped: ObservableFrame = json.loads(
        json.dumps(observable_frame, cls=TypedJSONEncoder), cls=TypedJSONDecoder
    )

    assert isinstance(round_tripped, ObservableFrame)

    assert {*observable_frame} == {*round_tripped}
    assert len(observable_frame) == len(round_tripped)


@pytest.mark.parametrize(
    "key, expected",
    [
        *((key, key) for key in ObservableType),
        *((key.value, key) for key in ObservableType),
    ],
)
def test_frame_validate_key(key, expected):
    assert ObservableFrame._validate_key(key) == expected


@pytest.mark.parametrize("key", [ObservableType.Temperature, "Temperature"])
def test_frame_magic_functions(key):
    observable_frame = ObservableFrame()
    assert len(observable_frame) == 0

    observable_frame[key] = ObservableArray(value=numpy.ones(1) * unit.kelvin)
    assert len(observable_frame) == 1

    assert key in observable_frame
    assert {*observable_frame} == {ObservableType.Temperature}

    del observable_frame[key]

    assert len(observable_frame) == 0
    assert key not in observable_frame


@pytest.mark.parametrize(
    "observable_frame, key, value, expected_raises, expected_message",
    [
        (
            ObservableFrame(
                {"Temperature": ObservableArray(value=numpy.ones(2) * unit.kelvin)}
            ),
            "Volume",
            numpy.ones(1) * unit.nanometer**3,
            pytest.raises(ValueError),
            "The length of the data (1) must match the length of the data already in "
            "the frame (2).",
        ),
        (
            ObservableFrame(),
            "Temperature",
            numpy.ones(1) * unit.pascals,
            pytest.raises(ValueError),
            "Temperature data must have units compatible with kelvin.",
        ),
    ],
)
def test_frame_set_invalid_item(
    observable_frame, key, value, expected_raises, expected_message
):
    with expected_raises as error_info:
        observable_frame[key] = ObservableArray(value=value)

    assert (
        expected_message is None
        and error_info is None
        or expected_message in str(error_info.value)
    )


@pytest.mark.parametrize("pressure", [None, 1 * unit.atmosphere])
def test_frame_from_openmm(pressure):
    observable_frame = ObservableFrame.from_openmm(
        get_data_filename("test/statistics/openmm_statistics.csv"), pressure
    )

    expected_types = {*ObservableType} - {ObservableType.ReducedPotential}

    if pressure is None:
        expected_types -= {ObservableType.Enthalpy}

    assert {*observable_frame} == expected_types
    assert len(observable_frame) == 10

    expected_values = {
        ObservableType.PotentialEnergy: 7934.831868494968 * unit.kilojoule / unit.mole,
        ObservableType.KineticEnergy: 5939.683117957521 * unit.kilojoule / unit.mole,
        ObservableType.TotalEnergy: 13874.51498645249 * unit.kilojoule / unit.mole,
        ObservableType.Temperature: 286.38157154881503 * unit.kelvin,
        ObservableType.Volume: 26.342326662784938 * unit.nanometer**3,
        ObservableType.Density: 0.6139877476363793 * unit.gram / unit.milliliter,
    }

    for observable_type, expected_value in expected_values.items():
        assert numpy.isclose(observable_frame[observable_type].value[0], expected_value)

    if pressure is not None:
        expected_enthalpy = (
            13874.51498645249 * unit.kilojoule / unit.mole
            + pressure * 26.342326662784938 * unit.nanometer**3 * unit.avogadro_constant
        )
        assert numpy.isclose(observable_frame["Enthalpy"].value[0], expected_enthalpy)


def test_frame_subset():
    observable_frame = ObservableFrame(
        {
            "Temperature": ObservableArray(
                value=numpy.arange(4) * unit.kelvin,
                gradients=[
                    ParameterGradient(
                        key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                        value=numpy.arange(4) * unit.kelvin,
                    )
                ],
            )
        }
    )

    subset = observable_frame.subset([1, 3])
    assert len(subset) == 2

    assert numpy.allclose(
        subset["Temperature"].value, numpy.array([[1.0], [3.0]]) * unit.kelvin
    )
    assert numpy.allclose(
        subset["Temperature"].gradients[0].value,
        numpy.array([[1.0], [3.0]]) * unit.kelvin,
    )


def test_frame_join():
    gradient_unit = unit.mole / unit.kilojoule

    observable_frames = [
        ObservableFrame(
            {
                "Temperature": ObservableArray(
                    value=(numpy.arange(2) + i * 2) * unit.kelvin,
                    gradients=[
                        ParameterGradient(
                            key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                            value=(numpy.arange(2) + i * 2)
                            * unit.kelvin
                            * gradient_unit,
                        )
                    ],
                )
            }
        )
        for i in range(2)
    ]

    joined = ObservableFrame.join(*observable_frames)
    assert len(joined) == 4

    assert numpy.allclose(
        joined["Temperature"].value,
        numpy.arange(4).reshape(-1, 1) * unit.kelvin,
    )

    assert numpy.allclose(
        joined["Temperature"].gradients[0].value,
        numpy.arange(4).reshape(-1, 1) * unit.kelvin * gradient_unit,
    )


@pytest.mark.parametrize(
    "observable_frames, expected_raises, expected_message",
    [
        (
            [
                ObservableFrame(
                    {"Temperature": ObservableArray(value=numpy.ones(2) * unit.kelvin)}
                )
            ],
            pytest.raises(ValueError),
            "At least two observable frames must be provided.",
        ),
        (
            [
                ObservableFrame(
                    {"Temperature": ObservableArray(value=numpy.ones(2) * unit.kelvin)}
                ),
                ObservableFrame(
                    {"Volume": ObservableArray(value=numpy.ones(2) * unit.nanometer**3)}
                ),
            ],
            pytest.raises(ValueError),
            "The observable frames must contain the same types of observable.",
        ),
    ],
)
def test_frame_join_fail(observable_frames, expected_raises, expected_message):
    with expected_raises as error_info:
        ObservableFrame.join(*observable_frames)

    assert (
        expected_message is None
        and error_info is None
        or expected_message in str(error_info.value)
    )


@pytest.mark.parametrize(
    "data_values, expected_error, sub_counts",
    [
        (
            numpy.random.normal(0.0, 1.0, (1000,)) * unit.kelvin,
            1.0 / numpy.sqrt(1000) * unit.kelvin,
            None,
        ),
        (
            numpy.random.normal(0.0, 1.0, (1000, 1)) * unit.kelvin,
            1.0 / numpy.sqrt(1000) * unit.kelvin,
            None,
        ),
        (
            numpy.array([1, 2, 2, 3, 3, 3]) * unit.kelvin,
            None,
            [1, 2, 3],
        ),
    ],
)
def test_bootstrap(data_values, expected_error, sub_counts):
    def bootstrap_function(values: ObservableArray) -> Observable:
        return Observable(
            value=values.value.mean().plus_minus(0.0 * values.value.units),
            gradients=[
                ParameterGradient(gradient.key, numpy.mean(gradient.value))
                for gradient in values.gradients
            ],
        )

    data = ObservableArray(
        value=data_values,
        gradients=[
            ParameterGradient(
                key=ParameterGradientKey("vdW", "[#6:1]", "epsilon"),
                value=data_values,
            )
        ],
    )

    average = bootstrap(bootstrap_function, 1000, 1.0, sub_counts, values=data)

    assert numpy.isclose(average.value, data.value.mean())
    assert numpy.isclose(average.gradients[0].value, data.value.mean())

    if expected_error is not None:
        assert numpy.isclose(average.error, expected_error, rtol=0.1)
