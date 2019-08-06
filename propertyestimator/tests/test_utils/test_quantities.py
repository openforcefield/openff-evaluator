"""A set of examples highlighting a number of possible implementations
for the EstimatedQuantity class.
"""

import pytest

from propertyestimator import unit
from propertyestimator.utils import quantities


def test_add_estimated_quantity():
    """Tests adding together two independent EstimatedQuantity."""

    a = 5 * unit.angstrom
    delta_a = 0.03 * unit.angstrom

    b = 10 * unit.angstrom
    delta_b = 0.04 * unit.angstrom

    quantity_a = quantities.EstimatedQuantity(a, delta_a, '325262315:npt_production')
    quantity_b = quantities.EstimatedQuantity(b, delta_b, '893487693:npt_production')

    # Addition of quantities:
    quantity_addition = quantity_a + quantity_b

    assert abs(quantity_addition.value - 15 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_addition.uncertainty - 0.05 * unit.angstrom) < 0.00001 * unit.angstrom


def test_subtract_estimated_quantity():
    """Tests subtracting two independent EstimatedQuantity."""

    a = 5 * unit.angstrom
    delta_a = 0.03 * unit.angstrom

    b = 10 * unit.angstrom
    delta_b = 0.04 * unit.angstrom

    quantity_a = quantities.EstimatedQuantity(a, delta_a, '325262315:npt_production')
    quantity_b = quantities.EstimatedQuantity(b, delta_b, '893487693:npt_production')

    quantity_subtraction = quantity_b - quantity_a

    assert abs(quantity_subtraction.value - 5 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_subtraction.uncertainty - 0.05 * unit.angstrom) < 0.00001 * unit.angstrom


def test_scalar_multiply_estimated_quantity():
    """Tests multiplying an EstimatedQuantity by a scalar."""

    a = 5 * unit.angstrom
    delta_a = 0.03 * unit.angstrom

    quantity_a = quantities.EstimatedQuantity(a, delta_a, '325262315:npt_production')

    quantity_scalar_multiply = quantity_a * 2.0

    assert abs(quantity_scalar_multiply.value - 10 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_multiply.uncertainty - 0.06 * unit.angstrom) < 0.00001 * unit.angstrom

    quantity_scalar_reverse_multiply = 2.0 * quantity_a

    assert abs(quantity_scalar_reverse_multiply.value - 10 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_reverse_multiply.uncertainty - 0.06 * unit.angstrom) < 0.00001 * unit.angstrom


def test_scalar_divide_estimated_quantity():
    """Tests dividing an EstimatedQuantity by a scalar."""

    b = 10 * unit.angstrom
    delta_b = 0.04 * unit.angstrom

    quantity_b = quantities.EstimatedQuantity(b, delta_b, '893487693:npt_production')

    quantity_scalar_divide = quantity_b / 2.0

    assert abs(quantity_scalar_divide.value - 5 * unit.angstrom) < 0.001 * unit.angstrom
    assert abs(quantity_scalar_divide.uncertainty - 0.02 * unit.angstrom) < 0.00001 * unit.angstrom


def test_estimated_quantity_correlated_exception():
    """Tests that adding / subtracting dependent estimated quantities raise an
    exception."""

    a = 5 * unit.angstrom
    delta_a = 0.03 * unit.angstrom

    quantity_a = quantities.EstimatedQuantity(a, delta_a, '325262315:npt_production')
    quantity_c = quantities.EstimatedQuantity(a, delta_a, '325262315:npt_production')

    with pytest.raises(quantities.DependantValuesException):
        _ = quantity_a + quantity_c

    with pytest.raises(quantities.DependantValuesException):
        _ = quantity_a - quantity_c


def test_estimated_quantity_serialization():
    """Tests the (de)serialization of an EstimatedQuantity"""

    a = 5 * unit.angstrom
    delta_a = 0.03 * unit.angstrom

    quantity_a = quantities.EstimatedQuantity(a, delta_a, '325262315:npt_production')

    state_a = quantity_a.__getstate__()

    quantity_b = quantities.EstimatedQuantity(0 * unit.kelvin, 0 * unit.kelvin, '685862315:npt_production')
    quantity_b.__setstate__(state_a)

    assert quantity_a.value == quantity_b.value
    assert quantity_a.uncertainty == quantity_b.uncertainty

    assert set(quantity_a.sources) == set(quantity_b.sources)
