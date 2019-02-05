"""A set of examples highlighting a number of possible implementations
for the EstimatedQuantity class.
"""

import pytest
import numpy as np

from propertyestimator.utils import quantities, statistics

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

    def bootstrap_function(array):
        return array.mean()

    sample_size = 2000

    original_data_a = np.random.normal(5, 0.3, sample_size) * unit.angstrom

    a_bootstrapped = statistics.perform_bootstrapping(bootstrap_function, 1.0,
                                                      sample_size, original_data_a)

    original_data_b = np.random.normal(10, 0.4, sample_size) * unit.angstrom

    b_bootstrapped = statistics.perform_bootstrapping(bootstrap_function, 1.0,
                                                      sample_size, original_data_b)

    quantity_a = quantities.BootstrappedEstimatedQuantity(original_data_a.mean(), a_bootstrapped)
    quantity_b = quantities.BootstrappedEstimatedQuantity(original_data_b.mean(), b_bootstrapped)

    # Addition of quantities:
    quantity_addition = quantity_a + quantity_b
    expected_uncertainty = (quantity_a.uncertainty**2 + quantity_b.uncertainty**2).sqrt()

    assert abs(quantity_addition.value - (quantity_a.value + quantity_b.value)) < 0.001 * unit.angstrom
    # assert abs(quantity_addition.uncertainty - 0.05 * unit.angstrom) < 0.00001 * unit.angstrom

    # Subtraction of quantities:
    quantity_subtraction = quantity_b - quantity_a

    assert abs(quantity_subtraction.value - (quantity_b.value - quantity_a.value)) < 0.001 * unit.angstrom
    # assert abs(quantity_subtraction.uncertainty - 0.05 * unit.angstrom) < 0.00001 * unit.angstrom

    # Scalar multiplication:
    quantity_scalar_multiply = quantity_a * 2.0

    assert abs(quantity_scalar_multiply.value - 2.0 * quantity_a.value) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_multiply.uncertainty - 2.0 * a_bootstrapped.std()) < 0.00001 * unit.angstrom

    quantity_scalar_reverse_multiply = 2.0 * quantity_a

    assert abs(quantity_scalar_reverse_multiply.value - 2.0 * quantity_a.value) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_reverse_multiply.uncertainty - 2.0 * a_bootstrapped.std()) < 0.00001 * unit.angstrom

    # Division by a scalar:
    quantity_scalar_divide = quantity_b / 2.0

    assert abs(quantity_scalar_divide.value - quantity_b.value / 2.0) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_divide.uncertainty - b_bootstrapped.std() / 2.0) < 0.00001 * unit.angstrom

    # Less than testing:
    assert quantity_a < quantity_b

    # Greater than testing:
    assert quantity_b > quantity_a

    # Equal to testing:
    quantity_c = quantities.BootstrappedEstimatedQuantity(original_data_a.mean(), a_bootstrapped)

    assert quantity_c == quantity_a
    assert quantity_b != quantity_a

    # Less than equal testing
    assert quantity_c <= quantity_a

    # Greater than equal testing
    assert quantity_c >= quantity_a
