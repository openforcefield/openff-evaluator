"""
Units tests for propertyestimator.attributes.typing
"""

import typing
from random import random

import pint
import pytest

from propertyestimator import unit
from propertyestimator.attributes.typing import (
    is_instance_of_type,
    is_type_subclass_of_type,
)


@pytest.mark.parametrize(
    "type_a, type_b",
    [
        (int, int),
        (int, float),
        (float, float),
        (int, typing.Union[int, float, pint.Quantity]),
        (float, typing.Union[int, float, pint.Quantity]),
        (pint.Quantity, typing.Union[int, float, pint.Quantity]),
        (typing.Union[int, float, pint.Quantity], int),
        (typing.Union[int, float, pint.Quantity], float),
        (typing.Union[int, float, pint.Quantity], pint.Quantity),
    ],
)
def test_positive_is_type_subclass_of_type(type_a, type_b):
    """A positive test of the `is_type_subclass_of_type` method."""
    assert is_type_subclass_of_type(type_a, type_b)


@pytest.mark.parametrize(
    "type_a, type_b",
    [
        (float, int),
        (int, typing.Union[pint.Quantity]),
        (float, typing.Union[int, pint.Quantity]),
        (pint.Quantity, typing.Union[int, float]),
        (typing.Union[float, pint.Quantity], int),
        (typing.Union[pint.Quantity], float),
        (typing.Union[int, float], pint.Quantity),
    ],
)
def test_negative_is_type_subclass_of_type(type_a, type_b):
    """A negative test of the `is_type_subclass_of_type` method."""
    assert not is_type_subclass_of_type(type_a, type_b)


@pytest.mark.parametrize(
    "object_a, type_a",
    [
        (int(random()), int),
        (int(random()), float),
        (random(), float),
        (int(random()), typing.Union[int, float, pint.Quantity]),
        (random(), typing.Union[int, float, pint.Quantity]),
        (random() * unit.kelvin, typing.Union[int, float, pint.Quantity]),
    ],
)
def test_positive_is_instance_of_type(object_a, type_a):
    """A positive test of the `is_instance_of_type` method."""
    assert is_instance_of_type(object_a, type_a)


@pytest.mark.parametrize(
    "object_a, type_a",
    [
        (random() + 0.0001, int),
        (int(random()), typing.Union[pint.Quantity]),
        (random(), typing.Union[int, pint.Quantity]),
        (random() * unit.kelvin, typing.Union[int, float]),
    ],
)
def test_negative_is_instance_of_type(object_a, type_a):
    """A negative test of the `is_instance_of_type` method."""
    assert not is_instance_of_type(object_a, type_a)
