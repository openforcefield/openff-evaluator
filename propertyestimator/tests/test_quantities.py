"""A set of examples highlighting a number of possible implementations
for the EstimatedQuantity class.
"""

import pytest
import numpy as np

from propertyestimator.utils import quantities
from simtk import unit


def test_vanilla_estimated_quantity():
    """Unit tests for the quantities.VanillaEstimatedQuantity class."""

    a = 5 * unit.angstrom
    delta_a = 0.03 * unit.angstrom

    b = 10 * unit.angstrom
    delta_b = 0.04 * unit.angstrom

    quantity_a = quantities.VanillaEstimatedQuantity(a, delta_a)
    quantity_b = quantities.VanillaEstimatedQuantity(b, delta_b)

    # Addition of quantities:
    quantity_addition = quantity_a + quantity_b

    assert abs(quantity_addition.value - 15 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_addition.uncertainty - 0.05 * unit.angstrom) < 0.00001 * unit.angstrom

    # Subtraction of quantities:
    quantity_subtraction = quantity_b - quantity_a

    assert abs(quantity_subtraction.value - 5 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_subtraction.uncertainty - 0.05 * unit.angstrom) < 0.00001 * unit.angstrom

    # Scalar multiplication:
    quantity_scalar_multiply = quantity_a * 2.0

    assert abs(quantity_scalar_multiply.value - 10 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_multiply.uncertainty - 0.06 * unit.angstrom) < 0.00001 * unit.angstrom

    quantity_scalar_reverse_multiply = 2.0 * quantity_a

    assert abs(quantity_scalar_reverse_multiply.value - 10 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_reverse_multiply.uncertainty - 0.06 * unit.angstrom) < 0.00001 * unit.angstrom

    # Division by a scalar:
    quantity_scalar_divide = quantity_b / 2.0

    assert abs(quantity_scalar_divide.value - 5 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_divide.uncertainty - 0.02 * unit.angstrom) < 0.00001 * unit.angstrom

    # Less than testing:
    assert quantity_a < quantity_b

    # Greater than testing:
    assert quantity_b > quantity_a

    # Equal to testing:
    quantity_c = quantities.VanillaEstimatedQuantity(a, delta_a)

    assert quantity_c == quantity_a
    assert quantity_b != quantity_a

    # Less than equal testing
    assert quantity_c <= quantity_a

    # Greater than equal testing
    assert quantity_c >= quantity_a


def test_tagged_estimated_quantity():
    """Unit tests for the quantities.TaggedEstimatedQuantity class."""

    a = 5 * unit.angstrom
    delta_a = 0.03 * unit.angstrom

    b = 10 * unit.angstrom
    delta_b = 0.04 * unit.angstrom

    quantity_a = quantities.TaggedEstimatedQuantity(a, delta_a, '325262315:npt_production')
    quantity_b = quantities.TaggedEstimatedQuantity(b, delta_b, '238957238:npt_production')

    # The below should not raise an exception.
    uncorrelated_addition = quantity_a + quantity_b
    uncorrelated_subtraction = quantity_b - quantity_a

    quantity_c = quantities.TaggedEstimatedQuantity(a, delta_a, '325262315:npt_production')

    with pytest.raises(quantities.DependantValuesException):
        correlated_addition = quantity_a + quantity_c

    with pytest.raises(quantities.DependantValuesException):
        correlated_subtraction = quantity_a - quantity_c


def test_explicit_estimated_quantity():
    """Unit tests for the quantities.ExplicitEstimatedQuantity class."""

    a = 5 * unit.angstrom
    delta_a = 0.03 * unit.angstrom

    b = 10 * unit.angstrom
    delta_b = 0.04 * unit.angstrom

    quantity_a = quantities.ExplicitEstimatedQuantity(a, delta_a)
    quantity_b = quantities.ExplicitEstimatedQuantity(b, delta_b)

    # Addition of quantities:
    quantity_addition = quantity_a.add_independent_quantity(quantity_b)

    assert abs(quantity_addition.value - 15 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_addition.uncertainty - 0.05 * unit.angstrom) < 0.00001 * unit.angstrom

    # Subtraction of quantities:
    quantity_subtraction = quantity_b.sub_independent_quantity(quantity_a)

    assert abs(quantity_subtraction.value - 5 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_subtraction.uncertainty - 0.05 * unit.angstrom) < 0.00001 * unit.angstrom

    # Scalar multiplication:
    quantity_scalar_multiply = quantity_a.multiply_by_scalar(2.0)

    assert abs(quantity_scalar_multiply.value - 10 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_multiply.uncertainty - 0.06 * unit.angstrom) < 0.00001 * unit.angstrom

    # Division by a scalar:
    quantity_scalar_divide = quantity_b.divide_by_scalar(2.0)

    assert abs(quantity_scalar_divide.value - 5 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_divide.uncertainty - 0.02 * unit.angstrom) < 0.00001 * unit.angstrom

    # Less than testing:
    assert quantity_a < quantity_b

    # Greater than testing:
    assert quantity_b > quantity_a

    # Equal to testing:
    quantity_c = quantities.ExplicitEstimatedQuantity(a, delta_a)

    assert quantity_c == quantity_a
    assert quantity_b != quantity_a

    # Less than equal testing
    assert quantity_c <= quantity_a

    # Greater than equal testing
    assert quantity_c >= quantity_a


def test_bootstrapped_estimated_quantity():
    """Unit tests for the quantities.VanillaEstimatedQuantity class."""

    a = 5 * unit.angstrom
    a_bootstraped = np.random.normal(5, 0.3, 1000) * unit.angstrom

    b = 10 * unit.angstrom
    b_bootstraped = np.random.normal(10, 0.4, 1000) * unit.angstrom

    quantity_a = quantities.BootstrappedEstimatedQuantity(a, a_bootstraped)
    quantity_b = quantities.BootstrappedEstimatedQuantity(b, b_bootstraped)

    # Addition of quantities:
    quantity_addition = quantity_a + quantity_b

    assert abs(quantity_addition.value - 15 * unit.angstrom) < 0.001 * unit.angstrom
    # assert abs(quantity_addition.uncertainty - 0.05 * unit.angstrom) < 0.00001 * unit.angstrom

    # Subtraction of quantities:
    quantity_subtraction = quantity_b - quantity_a

    assert abs(quantity_subtraction.value - 5 * unit.angstrom) < 0.001 * unit.angstrom
    # assert abs(quantity_subtraction.uncertainty - 0.05 * unit.angstrom) < 0.00001 * unit.angstrom

    # Scalar multiplication:
    quantity_scalar_multiply = quantity_a * 2.0

    assert abs(quantity_scalar_multiply.value - 10 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_multiply.uncertainty - 2.0 * a_bootstraped.std()) < 0.00001 * unit.angstrom

    quantity_scalar_reverse_multiply = 2.0 * quantity_a

    assert abs(quantity_scalar_reverse_multiply.value - 10 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_reverse_multiply.uncertainty - 2.0 * a_bootstraped.std()) < 0.00001 * unit.angstrom

    # Division by a scalar:
    quantity_scalar_divide = quantity_b / 2.0

    assert abs(quantity_scalar_divide.value - 5 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_divide.uncertainty - b_bootstraped.std() / 2.0) < 0.00001 * unit.angstrom

    # Less than testing:
    assert quantity_a < quantity_b

    # Greater than testing:
    assert quantity_b > quantity_a

    # Equal to testing:
    quantity_c = quantities.BootstrappedEstimatedQuantity(a, a_bootstraped)

    assert quantity_c == quantity_a
    assert quantity_b != quantity_a

    # Less than equal testing
    assert quantity_c <= quantity_a

    # Greater than equal testing
    assert quantity_c >= quantity_a
