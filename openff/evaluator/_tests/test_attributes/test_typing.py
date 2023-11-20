"""
Units tests for openff.evaluator.attributes.typing
"""

import typing
from random import random

import pytest
from openff.units import unit

from openff.evaluator.attributes.typing import (
    is_instance_of_type,
    is_type_subclass_of_type,
)


@pytest.mark.parametrize(
    "type_a, type_b",
    [
        (int, int),
        (int, float),
        (float, float),
        (int, typing.Union[int, float, unit.Quantity]),
        (float, typing.Union[int, float, unit.Quantity]),
        (unit.Quantity, typing.Union[int, float, unit.Quantity]),
        (typing.Union[int, float, unit.Quantity], int),
        (typing.Union[int, float, unit.Quantity], float),
        (typing.Union[int, float, unit.Quantity], unit.Quantity),
    ],
)
def test_positive_is_type_subclass_of_type(type_a, type_b):
    """A positive test of the `is_type_subclass_of_type` method."""
    assert is_type_subclass_of_type(type_a, type_b)


@pytest.mark.parametrize(
    "type_a, type_b",
    [
        (float, int),
        (int, typing.Union[unit.Quantity]),
        (float, typing.Union[int, unit.Quantity]),
        (unit.Quantity, typing.Union[int, float]),
        (typing.Union[float, unit.Quantity], int),
        (typing.Union[unit.Quantity], float),
        (typing.Union[int, float], unit.Quantity),
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
        (int(random()), typing.Union[int, float, unit.Quantity]),
        (random(), typing.Union[int, float, unit.Quantity]),
        (random() * unit.kelvin, typing.Union[int, float, unit.Quantity]),
    ],
)
def test_positive_is_instance_of_type(object_a, type_a):
    """A positive test of the `is_instance_of_type` method."""
    assert is_instance_of_type(object_a, type_a)


@pytest.mark.parametrize(
    "object_a, type_a",
    [
        (random() + 0.0001, int),
        (int(random()), typing.Union[unit.Quantity]),
        (random(), typing.Union[int, unit.Quantity]),
        (random() * unit.kelvin, typing.Union[int, float]),
    ],
)
def test_negative_is_instance_of_type(object_a, type_a):
    """A negative test of the `is_instance_of_type` method."""
    assert not is_instance_of_type(object_a, type_a)
