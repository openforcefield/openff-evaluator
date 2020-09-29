"""
A collection of classes for representing, storing, and manipulating common observables
which are collected over the course of a molecular simulation.
"""
import abc
import copy
import math
import operator
import re
from enum import Enum
from io import StringIO
from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy
import pandas
import pint.compat

from openff.evaluator import unit
from openff.evaluator.forcefield import ParameterGradient, ParameterGradientKey

# noinspection PyTypeChecker
T = TypeVar("T", bound="_Observable")


class _Observable(abc.ABC):
    """A common base class for observables."""

    @property
    def gradients(self) -> List[ParameterGradient]:
        return [*self._gradients]

    def __init__(self, value=None, gradients=None):

        self._value = None
        self._gradients = None

        self._initialize(value, gradients)

    @abc.abstractmethod
    def _initialize(self, value, gradients):
        """A initializer which is common to both ``__init__`` and ``__setstate__``."""
        self._value = value
        self._gradients = [] if gradients is None else gradients

    def __str__(self):
        return str(self._value)

    def __repr__(self):
        return repr(self._value)

    def __setstate__(self, state):
        self._initialize(state["value"], state["gradients"])

    def __getstate__(self):
        return {"value": self._value, "gradients": self._gradients}


class Observable(_Observable):
    """A class which stores the mean value of an observable as well as optionally
    the standard error in the mean and derivatives of the mean with respect to certain
    force field parameters.
    """

    @property
    def value(self) -> unit.Quantity:
        return None if self._value is None else self._value.value

    @property
    def error(self) -> unit.Quantity:
        return None if self._value is None else self._value.error

    def __init__(
        self,
        value: Union[unit.Measurement, unit.Quantity] = None,
        gradients: List[ParameterGradient] = None,
    ):
        super(Observable, self).__init__(value, gradients)

    def _initialize(
        self,
        value: Union[unit.Measurement, unit.Quantity],
        gradients: List[ParameterGradient],
    ):

        if value is not None and not isinstance(
            value, (unit.Quantity, unit.Measurement)
        ):

            raise TypeError(
                "The value must be either an `openff.evaluator.unit.Measurement` or "
                "an `openff.evaluator.unit.Quantity`."
            )

        if value is not None and not isinstance(value, unit.Measurement):

            if value is not None and not isinstance(value.magnitude, (int, float)):
                raise TypeError("The value must be a unit-wrapped integer or float.")

            value = value.plus_minus(0.0 * value.units)

        if gradients is not None:

            if value is None:
                raise ValueError("A valid value must be provided.")

            if not all(
                isinstance(gradient.value, unit.Quantity)
                and isinstance(gradient.value.magnitude, (int, float))
                for gradient in gradients
            ):

                raise TypeError(
                    "The gradient values must be unit-wrapped integers or floats."
                )

        super(Observable, self)._initialize(value, gradients)

    def _compatible_gradients(
        self, other: T
    ) -> Tuple[
        Dict[ParameterGradientKey, ParameterGradient],
        Dict[ParameterGradientKey, ParameterGradient],
    ]:
        """A common function for ensuring that two observables contain derivatives
        with respected to the same force field parameters, and refactors these
        derivatives into more easily manipulable dictionaries.

        Parameters
        ----------
        other
            The other observable object.

        Returns
        -------
            This object's and the other object's derivatives re-shaped into
            dictionaries.
        """

        self_gradients = {gradient.key: gradient for gradient in self._gradients}
        other_gradients = {gradient.key: gradient for gradient in other._gradients}

        if {*self_gradients} != {*other_gradients}:
            raise ValueError(
                "Two observables can only be summed if they contain gradients with "
                "respect to the same force field parameters."
            )

        return self_gradients, other_gradients

    def _add_sub(self, other: T, operator_function) -> T:
        """A common function for adding or subtracting two observables"""

        if type(self) != type(other):
            raise NotImplementedError()

        self_gradients, other_gradients = self._compatible_gradients(other)

        return self.__class__(
            value=operator_function(self._value, other._value),
            gradients=[
                operator_function(self_gradients[key], other_gradients[key])
                for key in self_gradients
            ],
        )

    def __add__(self, other: T) -> T:
        return self._add_sub(other, operator.add)

    def __sub__(self, other: T) -> T:
        return self._add_sub(other, operator.sub)

    def __mul__(self, other: Union[float, int, unit.Quantity, T]) -> T:

        if (
            isinstance(other, float)
            or isinstance(other, int)
            or isinstance(other, unit.Quantity)
        ):

            if (
                isinstance(other, unit.Quantity)
                and isinstance(other.magnitude, numpy.ndarray)
                and other.magnitude.ndim < 2
            ):
                other = other.reshape(-1, 1)

            return self.__class__(
                value=self._value * other,
                gradients=[gradient * other for gradient in self._gradients],
            )

        elif not isinstance(other, self.__class__):
            raise NotImplementedError()

        self_gradients, other_gradients = self._compatible_gradients(other)

        val = self.__class__(
            value=self._value * other._value,
            gradients=[
                (
                    other_gradients[key] * self._value.value
                    + self_gradients[key] * other._value.value
                )
                for key in self_gradients
            ],
        )

        return val

    def __rmul__(self, other: Union[float, int, unit.Quantity, T]) -> T:
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, int, unit.Quantity, T]):

        if not isinstance(other, self.__class__):
            return self * (1.0 / other)

        self_gradients, other_gradients = self._compatible_gradients(other)

        return self.__class__(
            value=self._value / other._value,
            gradients=[
                (
                    self_gradients[key] * other._value.value
                    - self._value.value * other_gradients[key]
                )
                / (other._value.value * other._value.value)
                for key in self_gradients
            ],
        )

    def __rtruediv__(self, other: Union[float, int, unit.Quantity, T]):

        if isinstance(other, self.__class__):
            return self.__truediv__(other)

        if not isinstance(other, (float, int, unit.Quantity)):
            raise NotImplementedError()

        if (
            isinstance(other, unit.Quantity)
            and isinstance(other.magnitude, numpy.ndarray)
            and other.magnitude.ndim < 2
        ):
            other = other.reshape(-1, 1)

        value = other / self._value

        return self.__class__(
            value=unit.Measurement(
                value.magnitude.nominal_value * value.units,
                value.magnitude.std_dev * value.units,
            ),
            gradients=[
                -other * gradient / (self._value.value * self._value.value)
                for gradient in self.gradients
            ],
        )


class ObservableArray(_Observable):
    """A class which stores the value(s) of an observable obtained via molecule
    simulation (or simulation data) as well as optionally the derivatives of the value
    with respect to certain force field parameters.
    """

    @property
    def value(self) -> unit.Quantity:
        """The value(s) of the observable."""
        return self._value

    def __init__(
        self, value: unit.Quantity = None, gradients: List[ParameterGradient] = None
    ):
        super(ObservableArray, self).__init__(value, gradients)

    def _initialize(self, value: unit.Quantity, gradients: List[ParameterGradient]):

        expected_types = (int, float, numpy.ndarray)

        if value is not None:

            if not isinstance(value, unit.Quantity) or not isinstance(
                value.magnitude, expected_types
            ):
                raise TypeError(
                    "The value must be a unit-wrapped integer, float or numpy array."
                )

            if not issubclass(type(value.magnitude), numpy.ndarray):
                value = numpy.array([value.magnitude]) * value.units

            # Ensure the inner array has a uniform shape.
            if value.magnitude.ndim > 2:

                raise ValueError(
                    "The wrapped array must not contain more than two dimensions."
                )

            if value.magnitude.ndim < 2:
                value = value.reshape(-1, 1)

        reshaped_gradients = []

        if gradients is not None:

            if value is None:
                raise ValueError("A valid value must be provided.")

            # Make sure the value and gradients have the same wrapped type.
            if not all(
                isinstance(gradient.value, unit.Quantity)
                and isinstance(gradient.value.magnitude, expected_types)
                for gradient in gradients
            ):

                raise TypeError(
                    "The gradient values must be unit-wrapped integers, floats "
                    "or numpy arrays."
                )

            # Make sure the gradient values are all numpy arrays and make sure the each
            # have the same shape as the value.
            for gradient in gradients:

                gradient_value = gradient.value.magnitude

                if not isinstance(gradient.value.magnitude, numpy.ndarray):
                    gradient_value = numpy.array([gradient_value])

                if gradient_value.ndim < 2:
                    gradient_value = gradient_value.reshape(-1, 1)

                if gradient_value.ndim > 2:

                    raise ValueError(
                        "Gradient values must not contain more than two dimensions."
                    )

                if value.magnitude.shape[1] != gradient_value.shape[1]:

                    raise ValueError(
                        f"Gradient values should be {value.magnitude.shape[1]}-"
                        f"dimensional to match the dimensionality of the value."
                    )

                if gradient_value.shape[0] != value.magnitude.shape[0]:

                    raise ValueError(
                        f"Gradient values should have a length of "
                        f"{value.magnitude.shape[0]} to match the length of the value."
                    )

                reshaped_gradients.append(
                    ParameterGradient(
                        key=gradient.key,
                        value=unit.Quantity(gradient_value, gradient.value.units),
                    )
                )

        super(ObservableArray, self)._initialize(value, reshaped_gradients)

    def subset(self, indices: Iterable[int]) -> "ObservableArray":
        """Extracts the subset of the values stored for this observable at the
        specified indices.

        Parameters
        ----------
        indices
            The indices of the entries to extract.

        Returns
        -------
            The subset of the observable values.
        """

        return self.__class__(
            value=self._value[indices],
            gradients=[
                ParameterGradient(key=gradient.key, value=gradient.value[indices])
                for gradient in self._gradients
            ],
        )

    @classmethod
    def join(cls, *observables: "ObservableArray") -> "ObservableArray":
        """Concatenates multiple observables together in the order that they appear in
        the args list.

        Parameters
        ----------
        observables
            The observables to join.

        Returns
        -------
            The concatenated observable object.
        """
        if len(observables) < 2:
            raise ValueError("At least two observables must be provided.")

        expected_gradients = {gradient.key for gradient in observables[0].gradients}
        expected_gradient_units = {
            gradient.key: gradient.value.units for gradient in observables[0].gradients
        }

        # Ensure the arrays contain the same observables.
        if not all(
            observable.value.dimensionality == observables[0].value.dimensionality
            for observable in observables
        ):
            raise ValueError("The observables must all have compatible units.")

        # Ensure the arrays contain gradients for the same FF parameters.
        if not all(
            {gradient.key for gradient in observable.gradients} == expected_gradients
            for observable in observables
        ):
            raise ValueError(
                "The observables must contain gradient information for the same "
                "parameters."
            )

        # Ensure the gradients are all in the same units.
        if not all(
            {gradient.key: gradient.value.units for gradient in observable.gradients}
            == expected_gradient_units
            for observable in observables
        ):
            raise ValueError(
                "The gradients of each of the observables must have the same units."
            )

        return ObservableArray(
            value=numpy.concatenate(
                [
                    observable.value.to(observables[0].value.units).magnitude
                    for observable in observables
                ]
            )
            * observables[0].value.units,
            gradients=[
                ParameterGradient(
                    key=gradient_key,
                    value=numpy.concatenate(
                        [
                            next(
                                x for x in observable.gradients if x.key == gradient_key
                            )
                            .value.to(expected_gradient_units[gradient_key])
                            .magnitude
                            for observable in observables
                        ]
                    )
                    * expected_gradient_units[gradient_key],
                )
                for gradient_key in expected_gradients
            ],
        )

    def __len__(self):
        return 0 if self._value is None else len(self._value)


class ObservableType(Enum):
    """An enumeration of the common observables which may be extracted from
    molecular simulations (or simulation data) and stored in an ``ObservableFrame``.
    """

    PotentialEnergy = "PotentialEnergy"
    KineticEnergy = "KineticEnergy"
    TotalEnergy = "TotalEnergy"
    Temperature = "Temperature"
    Volume = "Volume"
    Density = "Density"
    Enthalpy = "Enthalpy"
    ReducedPotential = "ReducedPotential"


class ObservableFrame(MutableMapping[Union[str, ObservableType], ObservableArray]):
    """A data object for storing and retrieving frames of the thermodynamic observables
    enumerated by the ``ObservableType`` enum.
    """

    _units: Dict[ObservableType, pint.Unit] = {
        ObservableType.PotentialEnergy: unit.kilojoules / unit.mole,
        ObservableType.KineticEnergy: unit.kilojoules / unit.mole,
        ObservableType.TotalEnergy: unit.kilojoules / unit.mole,
        ObservableType.Temperature: unit.kelvin,
        ObservableType.Volume: unit.nanometer ** 3,
        ObservableType.Density: unit.gram / unit.milliliter,
        ObservableType.Enthalpy: unit.kilojoules / unit.mole,
        ObservableType.ReducedPotential: unit.dimensionless,
    }

    def __init__(
        self, observables: Dict[Union[str, ObservableType], ObservableArray] = None
    ):
        self._observables: Dict[ObservableType, ObservableArray] = {}

        observables = {} if observables is None else observables

        for key, value in observables.items():
            self[key] = value

    @staticmethod
    def _validate_key(key: Union[str, ObservableType]) -> ObservableType:
        """Validates whether a key is either an `ObservableType` or a string
        representation of an `ObservableType`.

        A `KeyError` is raised if any other types are passed as an key,
        or if the `str` cannot be converted to an `ObservableType`

        Parameters
        ----------
        key
            The key to validate.

        Returns
        -------
            The validated key.
        """
        key_error_message = (
            "The key must either be an `ObservableType` object or a "
            "string representation of an `ObservableType` object."
        )

        if isinstance(key, str):

            try:
                key = ObservableType(key)
            except ValueError:
                raise KeyError(key_error_message)

        elif not isinstance(key, ObservableType):
            raise KeyError(key_error_message)

        return key

    def __getitem__(self, key: Union[str, ObservableType]) -> ObservableArray:
        return self._observables[self._validate_key(key)]

    def __setitem__(self, key: Union[str, ObservableType], value: ObservableArray):
        key = self._validate_key(key)

        if value.value is None or not isinstance(value.value.magnitude, numpy.ndarray):

            raise ValueError(
                "The value of the observable must be a unit-wrapped numpy array with"
                "shape=(n_measurements,) or shape=(n_measurements, 1)."
            )

        if not len(value) == len(self) and len(self) > 0:

            raise ValueError(
                f"The length of the data ({len(value)}) must match the "
                f"length of the data already in the frame ({len(self)})."
            )

        if value.value.dimensionality != self._units[key].dimensionality:

            raise ValueError(
                f"{key.value} data must have units compatible with {self._units[key]}."
            )

        self._observables[key] = value

    def __delitem__(self, key: Union[str, ObservableType]):
        del self._observables[self._validate_key(key)]

    def __iter__(self):
        return iter(self._observables)

    def __contains__(self, key: Union[str, ObservableType]) -> bool:
        return self._validate_key(key) in self._observables

    def __len__(self) -> int:
        observable = next(iter(self._observables.values()), None)
        return 0 if observable is None else len(observable)

    @classmethod
    def from_openmm(
        cls, file_path: str, pressure: unit.Quantity = None
    ) -> "ObservableFrame":
        """Creates an observable frame from the CSV output of an OpenMM simulation.

        Parameters
        ----------
        file_path
            The file path to the CSV file.
        pressure
            The pressure at which the observables in the csv file were collected.

        Returns
        -------
            The imported observables.
        """
        with open(file_path, "r") as file:

            file_contents = file.read()

            if len(file_contents) < 1:
                return cls()

            file_contents = file_contents[1:]
            file_contents = re.sub("#.*\n", "", file_contents)

            string_object = StringIO(file_contents)
            data_array = pandas.read_csv(string_object)

        observable_to_openmm_header = {
            ObservableType.PotentialEnergy: "Potential Energy (kJ/mole)",
            ObservableType.KineticEnergy: "Kinetic Energy (kJ/mole)",
            ObservableType.TotalEnergy: "Total Energy (kJ/mole)",
            ObservableType.Temperature: "Temperature (K)",
            ObservableType.Volume: "Box Volume (nm^3)",
            ObservableType.Density: "Density (g/mL)",
        }
        openmm_header_to_unit = {
            "Potential Energy (kJ/mole)": unit.kilojoules / unit.mole,
            "Kinetic Energy (kJ/mole)": unit.kilojoules / unit.mole,
            "Total Energy (kJ/mole)": unit.kilojoules / unit.mole,
            "Temperature (K)": unit.kelvin,
            "Box Volume (nm^3)": unit.nanometer ** 3,
            "Density (g/mL)": unit.gram / unit.milliliter,
        }

        observables = {
            observable_type: ObservableArray(
                value=numpy.array(data_array[header]) * openmm_header_to_unit[header]
            )
            for observable_type, header in observable_to_openmm_header.items()
            if header in data_array
        }

        if pressure is not None:

            observables[ObservableType.Enthalpy] = ObservableArray(
                value=(
                    observables[ObservableType.TotalEnergy].value
                    + observables[ObservableType.Volume].value
                    * pressure
                    * unit.avogadro_constant
                )
            )

        return cls(observables)

    def subset(self, indices: Iterable[int]) -> "ObservableFrame":
        """Extracts the subset of the the array which is located at the
        specified indices.

        Parameters
        ----------
        indices
            The indices of the entries to extract.

        Returns
        -------
            The subset of data.
        """

        return self.__class__(
            {
                observable_type: self[observable_type].subset(indices)
                for observable_type in self
            }
        )

    @classmethod
    def join(cls, *observable_frames: "ObservableFrame") -> "ObservableFrame":
        """Joins multiple observable frames together in the order that they appear in
        the args list.

        Parameters
        ----------
        observable_frames
            The observable frames to join.

        Returns
        -------
            The joined observable frame.
        """

        if len(observable_frames) < 2:
            raise ValueError("At least two observable frames must be provided.")

        expected_observables: Set[ObservableType] = {*observable_frames[0]}

        # Ensure the observable frames contain the same observables.
        if not all(
            {*observable_frame} == expected_observables
            for observable_frame in observable_frames
        ):

            raise ValueError(
                "The observable frames must contain the same types of observable."
            )

        joined_observables = {
            observable_type: ObservableArray.join(
                *(
                    observable_frame[observable_type]
                    for observable_frame in observable_frames
                )
            )
            for observable_type in expected_observables
        }

        return cls(joined_observables)

    def __setstate__(self, state):

        for key, value in state["observables"].items():
            self[key] = value

    def __getstate__(self):
        return {
            "observables": {
                key.value: value for key, value in self._observables.items()
            }
        }


def bootstrap(
    bootstrap_function: Callable,
    iterations: int = 200,
    relative_sample_size: float = 1.0,
    sub_counts: Iterable[int] = None,
    **observables: ObservableArray,
) -> Observable:
    """Bootstrapping a set of observables to compute the average value of the
    observables as well as the the standard error in the average.

    Parameters
    ----------
    bootstrap_function
        The function to evaluate at each bootstrap iteration.
    iterations
        The number of bootstrap iterations to perform.
    relative_sample_size
        The percentage sample size to bootstrap over, relative to the
        size of the full data set.
    sub_counts
        If the data being bootstrapped contains arrays of concatenated sub data
        (such as when reweighting), this variable can be used to specify the number
        of items which belong to each subset. Data is then sampled with replacement
        so that the bootstrap sample contains the correct proportion of data from
        each subset.

        If the data to bootstrap is of the form [x0, x1, x2, y0, y1] for example,
        then `data_sub_counts=[3, 2]` and a possible sample may look like
        [x0, x0, x2, y0, y0], but never [x0, x1, y0, y1, y1].

        The sub-counts must sum up to the total length of the data provided to
        ``observables``.
    observables
        The observables which will be passed to the bootstrap function. All observables
        must have the same length.

    Returns
    -------
        The average of the data and the uncertainty in the average.
    """

    if len(observables) == 0:
        raise ValueError("There are no observables to bootstrap")

    # Ensure that the observables are all compatible.
    data_size = len(observables[next(iter(observables))])

    assert all(
        isinstance(data_value, ObservableArray) for data_value in observables.values()
    )
    assert all(len(observables[key]) == data_size for key in observables)

    # Make a copy of the observables so we don't accidentally destroy anything.
    observables = copy.deepcopy(observables)

    if sub_counts is None:
        sub_counts = numpy.array([data_size])

    assert numpy.sum(sub_counts) == data_size

    # Compute the mean value (and gradients if present).
    mean_observable = bootstrap_function(**observables)

    # Bootstrap to compute the uncertainties
    bootstrapped_values = numpy.zeros(iterations)

    for bootstrap_iteration in range(iterations):

        sample_observables: Dict[str, ObservableArray] = {
            key: ObservableArray(
                value=(numpy.zeros(observables[key].value.magnitude.shape))
                * observables[key].value.units,
            )
            for key in observables
        }

        start_index = 0

        for sub_count in sub_counts:

            # Choose the sample size as a percentage of the full data set.
            sample_size = min(math.floor(sub_count * relative_sample_size), sub_count)
            sample_indices = numpy.random.choice(sub_count, sample_size)

            for key in observables:

                sub_data = observables[key].subset(
                    range(start_index, start_index + sub_count)
                )

                sample_observables[key].value[
                    start_index : start_index + sub_count
                ] = sub_data.value[sample_indices]

            start_index += sub_count

        bootstrapped_values[bootstrap_iteration] = (
            bootstrap_function(**sample_observables)
            .value.to(mean_observable.value.units)
            .magnitude
        )

    std_error = bootstrapped_values.std() * mean_observable.value.units

    return Observable(
        value=mean_observable.value.plus_minus(std_error),
        gradients=mean_observable.gradients,
    )


pint.compat.upcast_types.append(Observable)
pint.compat.upcast_types.append(ObservableArray)
