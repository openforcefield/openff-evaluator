"""Contains a set of classes for storing and manipulating estimated quantities and
their uncertainties
"""

from collections import Sized
from collections.abc import Iterable

import numpy as np
from simtk import unit
from uncertainties import ufloat

from propertyestimator.utils import statistics


class VanillaEstimatedQuantity:
    """A 'bare minimum' implementation of an `EstimatedQuantity` class, which employs
    the uncertainty package to linearly propagate uncertainties through simple arithmetic
    and scalar multiplication / division operations.

    Examples
    --------
    To add two **independent** quantities together:

    >>> from simtk import unit
    >>>
    >>> a_value = 5 * unit.angstrom
    >>> a_uncertainty = 0.03 * unit.angstrom
    >>>
    >>> b_value = 10 * unit.angstrom
    >>> b_uncertainty = 0.04 * unit.angstrom
    >>>
    >>> quantity_a = VanillaEstimatedQuantity(a_value, a_uncertainty)
    >>> quantity_b = VanillaEstimatedQuantity(b_value, b_uncertainty)
    >>>
    >>> quantity_addition = quantity_a + quantity_b

    To subtract one **independent** quantity from another:

    >>> quantity_subtraction = quantity_b - quantity_a

    To multiply by a scalar:

    >>> quantity_scalar_multiply = quantity_a * 2.0

    To divide by a scalar:

    >>> quantity_scalar_divide = quantity_a / 2.0
    """

    @property
    def value(self):
        return self._value

    @property
    def uncertainty(self):
        return self._uncertainty

    def __init__(self, value, uncertainty):
        """Constructs a new EstimatedQuantity object.

        Parameters
        ----------
        value: unit.Quantity
            The value of the estimated quantity.
        uncertainty: unit.Quantity
            The uncertainty in the value of the estimated quantity.
        """

        assert value is not None and uncertainty is not None

        assert isinstance(value, unit.Quantity)
        assert isinstance(uncertainty, unit.Quantity)

        assert value.unit.is_compatible(uncertainty.unit)

        self._value = value
        self._uncertainty = uncertainty

    def __add__(self, other):

        assert isinstance(other, VanillaEstimatedQuantity)

        self_ufloat, self_unit = VanillaEstimatedQuantity._get_uncertainty_object(self)
        other_ufloat, other_unit = VanillaEstimatedQuantity._get_uncertainty_object(other)

        assert self_unit == other_unit

        result_ufloat = self_ufloat + other_ufloat

        result_value = result_ufloat.nominal_value * self_unit
        result_uncertainty = result_ufloat.std_dev * self_unit

        return VanillaEstimatedQuantity(result_value, result_uncertainty)

    def __sub__(self, other):

        assert isinstance(other, VanillaEstimatedQuantity)

        self_ufloat, self_unit = VanillaEstimatedQuantity._get_uncertainty_object(self)
        other_ufloat, other_unit = VanillaEstimatedQuantity._get_uncertainty_object(other)

        assert self_unit == other_unit

        result_ufloat = self_ufloat - other_ufloat

        result_value = result_ufloat.nominal_value * self_unit
        result_uncertainty = result_ufloat.std_dev * self_unit

        return VanillaEstimatedQuantity(result_value, result_uncertainty)

    def __mul__(self, other):

        # We only support multiplication by a scalar here.
        assert np.issubdtype(type(other), float) or np.issubdtype(type(other), int)

        self_ufloat, self_unit = VanillaEstimatedQuantity._get_uncertainty_object(self)

        result_ufloat = self_ufloat * other

        result_value = result_ufloat.nominal_value * self_unit
        result_uncertainty = result_ufloat.std_dev * self_unit

        return VanillaEstimatedQuantity(result_value, result_uncertainty)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):

        self_ufloat, self_unit = VanillaEstimatedQuantity._get_uncertainty_object(self)

        result_ufloat = self_ufloat / other

        result_value = result_ufloat.nominal_value * self_unit
        result_uncertainty = result_ufloat.std_dev * self_unit

        return VanillaEstimatedQuantity(result_value, result_uncertainty)

    def __eq__(self, other):
        if not isinstance(other, VanillaEstimatedQuantity):
            return False

        return self.value == other.value and self.uncertainty == other.uncertainty

    def __ne__(self, other):
        return not self.__eq__(other)

    def __ge__(self, other):
        assert isinstance(other, VanillaEstimatedQuantity)
        return self._value >= other.value

    def __gt__(self, other):
        assert isinstance(other, VanillaEstimatedQuantity)
        return self._value > other.value

    def __le__(self, other):
        assert isinstance(other, VanillaEstimatedQuantity)
        return self._value <= other.value

    def __lt__(self, other):
        assert isinstance(other, VanillaEstimatedQuantity)
        return self._value < other.value

    @staticmethod
    def _get_uncertainty_object(estimated_quantity):
        """Converts a `VanillaEstimatedQuantity` object into an uncertainties
        `ufloat` representation.

        Parameters
        ----------
        estimated_quantity: VanillaEstimatedQuantity
            The quantity to create the uncertainties object for.

        Returns
        -------
        ufloat
            The ufloat representation of the estimated quantity object.
        unit.Unit
            The unit of the values encoded in the ufloat object.
        """

        value_in_default_unit_system = estimated_quantity.value.in_unit_system(unit.md_unit_system)
        uncertainty_in_default_unit_system = estimated_quantity.uncertainty.in_unit_system(unit.md_unit_system)

        value_unit = value_in_default_unit_system.unit
        unitless_value = estimated_quantity.value.value_in_unit(value_unit)

        uncertainty_unit = uncertainty_in_default_unit_system.unit
        unitless_uncertainty = estimated_quantity.uncertainty.value_in_unit(uncertainty_unit)

        assert value_unit == uncertainty_unit

        return ufloat(unitless_value, unitless_uncertainty), value_unit


class DependantValuesException(ValueError):
    """An exception which is raised when arithmetic operations are applied
    to two quantities which are not independent."""

    def __init__(self):

        super().__init__('The two quantities came from the same source, and so'
                         'cannot be treated as independent. The propagation /'
                         'calculation of uncertainties must be handled by'
                         'more sophisticated methods than this class employs.')


class TaggedEstimatedQuantity(VanillaEstimatedQuantity):
    """A slightly more sophisticated version of the `EstimatedQuantity` which inherits
    from the bare minimum `VanillaEstimatedQuantity` implementation, and extends it
    by tracking the source of the data stored within.

    Notes
    -----
    Care must be taken that the source string is standardised. As an example, this
    design would not work if comparing a quantity whose source is a protocol id, and
    another whose source is a trajectory file name.

    Examples
    --------
    To add two **independent** quantities together:

    >>> from simtk import unit
    >>>
    >>> a_value = 5 * unit.angstrom
    >>> a_uncertainty = 0.03 * unit.angstrom
    >>>
    >>> b_value = 10 * unit.angstrom
    >>> b_uncertainty = 0.04 * unit.angstrom
    >>>
    >>> quantity_a = TaggedEstimatedQuantity(a_value, a_uncertainty, 'calc_325262315:npt_production')
    >>> quantity_b = TaggedEstimatedQuantity(b_value, b_uncertainty, 'calc_987234582:npt_production')
    >>>
    >>> quantity_addition = quantity_a + quantity_b

    To subtract one **independent** quantity from another:

    >>> quantity_subtraction = quantity_b - quantity_a

    Attempting to add quantities from the same source (i.e probably correlated)
    will raise a `DependantValuesException`.

    >>> c_value = 5.04 * unit.angstrom
    >>> c_uncertainty = 0.029 * unit.angstrom
    >>>
    >>> quantity_c = TaggedEstimatedQuantity(c_value, c_uncertainty, 'calc_325262315:npt_production')
    >>>
    >>> # The below raised a DependantValuesException.
    >>> quantity_addition = quantity_a + quantity_c
    """

    @property
    def source(self):
        return self._source

    def __init__(self, value, uncertainty, source):
        """Constructs a new TaggedEstimatedQuantity object.

        Parameters
        ----------
        value: unit.Quantity
            The value of the estimated quantity.
        uncertainty: unit.Quantity
            The uncertainty in the value of the estimated quantity.
        source: str
            A string representation of where this value came from. This
            value is employed whenever this object is involved in any
            mathematical operations to ensure values from the same source
            are not accidentally combined.

            An example of this may be the file path of the trajectory from
            which this value was derived, or the name of the workflow protocol
            which calculated it.
        """
        super().__init__(value, uncertainty)

        self._source = source

    def __add__(self, other):

        assert isinstance(other, TaggedEstimatedQuantity)

        if self.source == other.source:
            raise DependantValuesException()

        return super(TaggedEstimatedQuantity, self).__add__(other)

    def __sub__(self, other):

        assert isinstance(other, TaggedEstimatedQuantity)

        if self.source == other.source:
            raise DependantValuesException()

        return super(TaggedEstimatedQuantity, self).__sub__(other)


class ExplicitEstimatedQuantity:
    """An implementation of an `EstimatedQuantity` class, which is almost identical
    to the VanillaEstimatedQuantity implementation, except now all operators have been
    replaced with explicit method calls.

    Examples
    --------
    To add two **independent** quantities together:

    >>> from simtk import unit
    >>>
    >>> a_value = 5 * unit.angstrom
    >>> a_uncertainty = 0.03 * unit.angstrom
    >>>
    >>> b_value = 10 * unit.angstrom
    >>> b_uncertainty = 0.04 * unit.angstrom
    >>>
    >>> quantity_a = ExplicitEstimatedQuantity(a_value, a_uncertainty)
    >>> quantity_b = ExplicitEstimatedQuantity(b_value, b_uncertainty)
    >>>
    >>> quantity_addition = quantity_a.add_independent_quantity(quantity_b)

    To subtract one **independent** quantity from another:

    >>> quantity_subtraction = quantity_b.sub_independent_quantity(quantity_a)

    To multiply by a scalar:

    >>> quantity_scalar_multiply = quantity_a.multiply_by_scalar(2.0)

    To divide by a scalar:

    >>> quantity_scalar_divide = quantity_a.divide_by_scalar(2.0)
    """

    @property
    def value(self):
        return self._value

    @property
    def uncertainty(self):
        return self._uncertainty

    def __init__(self, value, uncertainty):
        """Constructs a new EstimatedQuantity object.

        Parameters
        ----------
        value: unit.Quantity
            The value of the estimated quantity.
        uncertainty: unit.Quantity
            The uncertainty in the value of the estimated quantity.
        """

        assert value is not None and uncertainty is not None

        assert isinstance(value, unit.Quantity)
        assert isinstance(uncertainty, unit.Quantity)

        assert value.unit.is_compatible(uncertainty.unit)

        self._value = value
        self._uncertainty = uncertainty

    def add_independent_quantity(self, other):

        assert isinstance(other, ExplicitEstimatedQuantity)

        self_ufloat, self_unit = ExplicitEstimatedQuantity._get_uncertainty_object(self)
        other_ufloat, other_unit = ExplicitEstimatedQuantity._get_uncertainty_object(other)

        assert self_unit == other_unit

        result_ufloat = self_ufloat + other_ufloat

        result_value = result_ufloat.nominal_value * self_unit
        result_uncertainty = result_ufloat.std_dev * self_unit

        return ExplicitEstimatedQuantity(result_value, result_uncertainty)

    def sub_independent_quantity(self, other):

        assert isinstance(other, ExplicitEstimatedQuantity)

        self_ufloat, self_unit = ExplicitEstimatedQuantity._get_uncertainty_object(self)
        other_ufloat, other_unit = ExplicitEstimatedQuantity._get_uncertainty_object(other)

        assert self_unit == other_unit

        result_ufloat = self_ufloat - other_ufloat

        result_value = result_ufloat.nominal_value * self_unit
        result_uncertainty = result_ufloat.std_dev * self_unit

        return ExplicitEstimatedQuantity(result_value, result_uncertainty)

    def multiply_by_scalar(self, other):

        # We only support multiplication by a scalar here.
        assert np.issubdtype(type(other), float) or np.issubdtype(type(other), int)

        self_ufloat, self_unit = ExplicitEstimatedQuantity._get_uncertainty_object(self)

        result_ufloat = self_ufloat * other

        result_value = result_ufloat.nominal_value * self_unit
        result_uncertainty = result_ufloat.std_dev * self_unit

        return ExplicitEstimatedQuantity(result_value, result_uncertainty)

    def divide_by_scalar(self, other):

        self_ufloat, self_unit = ExplicitEstimatedQuantity._get_uncertainty_object(self)

        result_ufloat = self_ufloat / other

        result_value = result_ufloat.nominal_value * self_unit
        result_uncertainty = result_ufloat.std_dev * self_unit

        return ExplicitEstimatedQuantity(result_value, result_uncertainty)

    def __eq__(self, other):
        if not isinstance(other, ExplicitEstimatedQuantity):
            return False

        return self.value == other.value and self.uncertainty == other.uncertainty

    def __ne__(self, other):
        return not self.__eq__(other)

    def __ge__(self, other):
        assert isinstance(other, ExplicitEstimatedQuantity)
        return self._value >= other.value

    def __gt__(self, other):
        assert isinstance(other, ExplicitEstimatedQuantity)
        return self._value > other.value

    def __le__(self, other):
        assert isinstance(other, ExplicitEstimatedQuantity)
        return self._value <= other.value

    def __lt__(self, other):
        assert isinstance(other, ExplicitEstimatedQuantity)
        return self._value < other.value

    @staticmethod
    def _get_uncertainty_object(estimated_quantity):
        """Converts a `ExplicitEstimatedQuantity` object into an uncertainties
        `ufloat` representation.

        Parameters
        ----------
        estimated_quantity: ExplicitEstimatedQuantity
            The quantity to create the uncertainties object for.

        Returns
        -------
        ufloat
            The ufloat representation of the estimated quantity object.
        unit.Unit
            The unit of the values encoded in the ufloat object.
        """

        value_in_default_unit_system = estimated_quantity.value.in_unit_system(unit.md_unit_system)
        uncertainty_in_default_unit_system = estimated_quantity.uncertainty.in_unit_system(unit.md_unit_system)

        value_unit = value_in_default_unit_system.unit
        unitless_value = estimated_quantity.value.value_in_unit(value_unit)

        uncertainty_unit = uncertainty_in_default_unit_system.unit
        unitless_uncertainty = estimated_quantity.uncertainty.value_in_unit(uncertainty_unit)

        assert value_unit == uncertainty_unit

        return ufloat(unitless_value, unitless_uncertainty), value_unit


class BootstrappedEstimatedQuantity:
    """A 'bootstrap everything' implementation of an `EstimatedQuantity` class.

    Examples
    --------
    To create an estimated quantity:

    >>> from simtk import unit
    >>>
    >>> def bootstrap_function(bootstrap_array):
    >>>     return bootstrap_array.mean()
    >>>
    >>> sample_size = 2000
    >>>
    >>> original_data_a = np.random.normal(5, 0.3, sample_size) * unit.angstrom
    >>>
    >>> a_bootstrapped = statistics.perform_bootstrapping(bootstrap_function, 1.0,
    >>>                                                   sample_size, original_data_a)
    >>>
    >>> original_data_b = np.random.normal(10, 0.4, sample_size) * unit.angstrom
    >>>
    >>> b_bootstrapped = statistics.perform_bootstrapping(bootstrap_function, 1.0,
    >>>                                                   sample_size, original_data_b)
    >>>
    >>> quantity_a = BootstrappedEstimatedQuantity(original_data_a.mean(), a_bootstrapped)
    >>> quantity_b = BootstrappedEstimatedQuantity(original_data_b.mean(), b_bootstrapped)

    To add two quantities together:

    >>> quantity_addition = quantity_a + quantity_b

    To subtract one quantity from another:

    >>> quantity_subtraction = quantity_b - quantity_a

    To multiply by a scalar:

    >>> quantity_scalar_multiply = quantity_a * 2.0

    To divide by a scalar:

    >>> quantity_scalar_divide = quantity_a / 2.0
    """

    @property
    def value(self):
        return self._value

    @property
    def uncertainty(self):
        return self._bootstrapped_values.std()

    def __init__(self, value, bootstrapped_values):
        """Constructs a new EstimatedQuantity object.

        Parameters
        ----------
        value: unit.Quantity
            The value of the estimated quantity.
        bootstrapped_values: np.ndarray of unit.Quantity
            An array of values which were obtained by bootstrapping
            the original data set from which `value` was calculated.
        """

        assert value is not None and bootstrapped_values is not None

        assert isinstance(value, unit.Quantity)
        assert isinstance(bootstrapped_values, unit.Quantity)

        assert (isinstance(bootstrapped_values, Iterable) and
                isinstance(bootstrapped_values, Sized) and
                len(bootstrapped_values) > 0)

        assert value.unit.is_compatible(bootstrapped_values.unit)

        self._value = value
        self._bootstrapped_values = bootstrapped_values

    def __add__(self, other):

        assert isinstance(other, BootstrappedEstimatedQuantity)
        assert len(self._bootstrapped_values) == len(other._bootstrapped_values)

        def bootstrap_function(*sample_data_sets):

            data_set_a = sample_data_sets[0]
            data_set_b = sample_data_sets[1]

            result_array = data_set_a + data_set_b
            return result_array.mean()

        bootstrap_results = statistics.perform_bootstrapping(bootstrap_function,
                                                             1.0,
                                                             len(self._bootstrapped_values),
                                                             self._bootstrapped_values,
                                                             other._bootstrapped_values)

        result_value = self.value + other.value

        return BootstrappedEstimatedQuantity(result_value, bootstrap_results)

    def __sub__(self, other):

        assert isinstance(other, BootstrappedEstimatedQuantity)
        assert len(self._bootstrapped_values) == len(other._bootstrapped_values)

        def bootstrap_function(*sample_data_sets):

            data_set_a = sample_data_sets[0]
            data_set_b = sample_data_sets[1]

            result_array = data_set_a - data_set_b
            return result_array.mean()

        bootstrap_results = statistics.perform_bootstrapping(bootstrap_function,
                                                             1.0,
                                                             len(self._bootstrapped_values),
                                                             self._bootstrapped_values,
                                                             other._bootstrapped_values)

        result_value = self.value - other.value

        return BootstrappedEstimatedQuantity(result_value, bootstrap_results)

    def __mul__(self, other):
        # We only support multiplication by a scalar here.
        assert np.issubdtype(type(other), float) or np.issubdtype(type(other), int)

        multiplied_bootstraps, bootstrap_unit = statistics.get_unitless_array(
            self._bootstrapped_values)

        multiplied_bootstraps *= other
        multiplied_bootstraps = multiplied_bootstraps * bootstrap_unit

        return BootstrappedEstimatedQuantity(self._value * other, multiplied_bootstraps)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        assert np.issubdtype(type(other), float) or np.issubdtype(type(other), int)

        multiplied_bootstraps, bootstrap_unit = statistics.get_unitless_array(
            self._bootstrapped_values)

        multiplied_bootstraps /= other
        multiplied_bootstraps = multiplied_bootstraps * bootstrap_unit

        return BootstrappedEstimatedQuantity(self._value / other, multiplied_bootstraps)

    def __eq__(self, other):
        if not isinstance(other, BootstrappedEstimatedQuantity):
            return False

        for self_value, other_value in zip(self._bootstrapped_values, other._bootstrapped_values):

            if self_value == other_value:
                continue

            return False

        return self.value == other.value

    def __ne__(self, other):
        return not self.__eq__(other)

    def __ge__(self, other):
        assert isinstance(other, BootstrappedEstimatedQuantity)
        return self._value >= other.value

    def __gt__(self, other):
        assert isinstance(other, BootstrappedEstimatedQuantity)
        return self._value > other.value

    def __le__(self, other):
        assert isinstance(other, BootstrappedEstimatedQuantity)
        return self._value <= other.value

    def __lt__(self, other):
        assert isinstance(other, BootstrappedEstimatedQuantity)
        return self._value < other.value
