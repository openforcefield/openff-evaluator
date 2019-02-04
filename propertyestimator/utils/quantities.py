"""Contains a set of classes for storing and manipulating estimated quantities and
their uncertainties
"""

import math
from collections import Sized
from collections.abc import Iterable

import numpy as np
from simtk import unit
from uncertainties import ufloat


class VanillaEstimatedQuantity:
    """A 'bare minimum' implementation of an `EstimatedQuantity` class, which employs
    the uncertainty package to linearly propagate uncertainties through simple arithmetic
    and scalar multiplication / division operations.
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
        """
        """
        if not isinstance(other, VanillaEstimatedQuantity):
            return False

        return self.value == other.value and self.uncertainty == other.uncertainty

    def __ne__(self, other):
        """
        """
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
        """
        """
        if not isinstance(other, ExplicitEstimatedQuantity):
            return False

        return self.value == other.value and self.uncertainty == other.uncertainty

    def __ne__(self, other):
        """
        """
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
    """
    """

    @property
    def value(self):
        return self._value

    @property
    def uncertainty(self):
        return self._bootstrapped_values.std()

    def __init__(self, value, bootstrapped_values, relative_sample_size=1.0):
        """Constructs a new EstimatedQuantity object.

        Parameters
        ----------
        value: unit.Quantity
            The value of the estimated quantity.
        bootstrapped_values: np.ndarray of unit.Quantity
            An array of values which were obtained by bootstrapping
            the original data set from which `value` was calculated.
        relative_sample_size: The relative size of the data set to use when bootstrapping
            when computing uncertainties.
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

        self._bootstrap_relative_sample_size = relative_sample_size

    @staticmethod
    def _get_unitless_array(data_set):

        array_in_default_unit_system = data_set.in_unit_system(unit.md_unit_system)

        array_unit = array_in_default_unit_system.unit
        unitless_array = data_set.value_in_unit(array_unit)

        return unitless_array, array_unit

    @staticmethod
    def _perform_bootstrapping(bootstrap_function, relative_sample_size, *data_sets):

        """Performs bootstrapping on a data set to calculate the
        average value, and the standard error in the average,
        bootstrapping.

        Parameters
        ----------
        data_set: np.ndarray of unit.Quantity
            The data set to perform bootstrapping on.
        bootstrap_function: function
            The function to apply to the bootstrapped data

        Returns
        -------
        np.ndarray of unit.Quantity
            The bootstrapped data.
        """

        assert len(data_sets) > 0

        # Make a copy of the data so we don't accidentally destroy anything.
        data_to_bootstrap = []

        data_set_size = len(data_sets[0])
        data_set_unit = None

        for data_set in data_sets:

            assert data_set_size == len(data_set)

            unitless_array, stripped_unit = BootstrappedEstimatedQuantity._get_unitless_array(data_set)

            if data_set_unit is not None:
                assert data_set_unit == stripped_unit

            data_set_unit = stripped_unit
            data_to_bootstrap.append(unitless_array)

        # Choose the sample size as a percentage of the full data set.
        sample_size = min(math.floor(data_set_size * relative_sample_size), data_set_size)

        evaluated_values = np.zeros(data_set_size)

        for bootstrap_iteration in range(data_set_size):

            sample_datasets = []

            for data_set in data_to_bootstrap:

                sample_indices = np.random.choice(data_set_size, sample_size)
                sample_datasets.append(data_set[sample_indices])

            evaluated_values[bootstrap_iteration] = bootstrap_function(*sample_datasets)

        return evaluated_values * data_set_unit

    def __add__(self, other):

        assert isinstance(other, BootstrappedEstimatedQuantity)

        def bootstrap_function(*sample_data_sets):

            data_set_a = sample_data_sets[0]
            data_set_b = sample_data_sets[1]

            result_array = data_set_a + data_set_b
            return result_array.mean()

        bootstrap_results = self._perform_bootstrapping(bootstrap_function,
                                                        self._bootstrap_relative_sample_size,
                                                        self._bootstrapped_values,
                                                        other._bootstrapped_values)

        result_value = self.value + other.value

        return BootstrappedEstimatedQuantity(result_value, bootstrap_results)

    def __sub__(self, other):

        assert isinstance(other, BootstrappedEstimatedQuantity)

        def bootstrap_function(*sample_data_sets):

            data_set_a = sample_data_sets[0]
            data_set_b = sample_data_sets[1]

            result_array = data_set_a - data_set_b
            return result_array.mean()

        bootstrap_results = self._perform_bootstrapping(bootstrap_function,
                                                        self._bootstrap_relative_sample_size,
                                                        self._bootstrapped_values,
                                                        other._bootstrapped_values)

        result_value = self.value - other.value

        return BootstrappedEstimatedQuantity(result_value, bootstrap_results)

    def __mul__(self, other):
        # We only support multiplication by a scalar here.
        assert np.issubdtype(type(other), float) or np.issubdtype(type(other), int)

        multiplied_bootstraps, bootstrap_unit = BootstrappedEstimatedQuantity._get_unitless_array(
            self._bootstrapped_values)

        multiplied_bootstraps *= other
        multiplied_bootstraps = multiplied_bootstraps * bootstrap_unit

        return BootstrappedEstimatedQuantity(self._value * other, multiplied_bootstraps)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        assert np.issubdtype(type(other), float) or np.issubdtype(type(other), int)

        multiplied_bootstraps, bootstrap_unit = BootstrappedEstimatedQuantity._get_unitless_array(
            self._bootstrapped_values)

        multiplied_bootstraps /= other
        multiplied_bootstraps = multiplied_bootstraps * bootstrap_unit

        return BootstrappedEstimatedQuantity(self._value / other, multiplied_bootstraps)

    def __eq__(self, other):
        """
        """
        if not isinstance(other, BootstrappedEstimatedQuantity):
            return False

        for self_value, other_value in zip(self._bootstrapped_values, other._bootstrapped_values):

            if self_value == other_value:
                continue

            return False

        return self.value == other.value

    def __ne__(self, other):
        """
        """
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
