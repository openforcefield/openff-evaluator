"""
A collection of classes for loading and manipulating statistics data files.
"""
import copy
import math
import re
from enum import Enum
from io import StringIO

import numpy as np
import pandas as pd

from propertyestimator import unit


class ObservableType(Enum):
    """The supported statistics which may be extracted / stored
    in statistics data files.
    """

    PotentialEnergy = 'PotentialEnergy'
    KineticEnergy = 'KineticEnergy'
    TotalEnergy = 'TotalEnergy'
    Temperature = 'Temperature'
    Volume = 'Volume'
    Density = 'Density'
    Enthalpy = 'Enthalpy'
    ReducedPotential = 'ReducedPotential'


class StatisticsArray:
    """
    A data object for storing and retrieving the thermodynamic statistics
    described by the `ObservableType` enum.
    """

    _observable_units = {
        ObservableType.PotentialEnergy: unit.kilojoules / unit.mole,
        ObservableType.KineticEnergy: unit.kilojoules / unit.mole,
        ObservableType.TotalEnergy: unit.kilojoules / unit.mole,
        ObservableType.Temperature: unit.kelvin,
        ObservableType.Volume: unit.nanometer**3,
        ObservableType.Density: unit.gram / unit.milliliter,
        ObservableType.Enthalpy: unit.kilojoules / unit.mole,
        ObservableType.ReducedPotential: unit.dimensionless
    }

    def __init__(self):
        """Constructs a new StatisticsArray object."""
        self._internal_data = {}

    @staticmethod
    def _validate_key(key):
        """Validates whether an key is either an `ObservableType` or a
        string representation of an `ObservableType`.

        A `KeyError` is raised if any other types are passed as an key,
        or if the `str` cannot be converted to an `ObservableType`

        Parameters
        ----------
        key: str or ObservableType
            The key to validate.

        Returns
        -------
        ObservableType
            The validated key
        """
        key_error_message = 'The key must either be an ObservableType or a ' \
                            'string representation of an ObservableType'

        if isinstance(key, str):

            try:
                key = ObservableType(key)
            except ValueError:
                raise KeyError(key_error_message)

        elif not isinstance(key, ObservableType):
            raise KeyError(key_error_message)

        return key

    def __getitem__(self, key):
        """Return the data for a given observable.

        Parameters
        ----------
        key: ObservableType or str
            The type of observable to retrieve.

        Returns
        -------
        unit.Quantity
            The unit wrapped data (shape=(len(self)) dtype=float) of the
            given type, or `None` if not present in the array.
        """

        key = self._validate_key(key)
        return self._internal_data[key]

    def __setitem__(self, key, value):
        """Sets the data for a given observable.

        Parameters
        ----------
        key: ObservableType or str
            The type of observable to set.
        value: unit.Quantity
            The unit wrapped data to set with shape=len(self) and dtype=float
        """

        key = self._validate_key(key)

        if not len(value) == len(self) and len(self) > 0:

            raise ValueError(f'The length of the data ({len(value)}) must match the'
                             f'shape of the array ({len(self)})')

        if not isinstance(value, unit.Quantity):
            raise ValueError('The data must be a unit bearing `propertyestimator.unit.Quantity`')

        if unit.get_base_units(value.units)[-1] != unit.get_base_units(StatisticsArray._observable_units[key])[-1]:

            raise ValueError(f'{key} must have units compatible with '
                             f'{StatisticsArray._observable_units[key]}')

        self._internal_data[key] = value

    def __contains__(self, key):
        """Queries whether the array contains data for a given observable.

        Parameters
        ----------
        key: ObservableType or str
            The type of observable to query for.

        Returns
        -------
        bool
            True if data for the `key` is available.
        """
        key = self._validate_key(key)
        return key in self._internal_data and len(self._internal_data[key]) > 0

    def __len__(self):
        """Get the number of data items in the array.

        Returns
        -------
        int
            The number of data items in the array.
        """
        if len(self._internal_data) == 0:
            return 0

        return len(next(iter(self._internal_data.values())))

    def to_pandas_csv(self, file_path):
        """Saves the `StatisticsArray` to a pandas csv file.

        Parameters
        ----------
        file_path: str
            The file path to save the csv file to.
        """

        if len(self._internal_data) == 0:

            # Handle the case where there is no data in the array.
            with open(file_path, 'w') as file:
                file.write('')

            return

        data_list = []
        units_list = {}

        for observable_type in ObservableType:

            if observable_type not in self:
                continue

            data = self._internal_data[observable_type]
            unit_type = StatisticsArray._observable_units[observable_type]

            data_list.append(data.to(unit_type).magnitude)
            units_list[observable_type] = f'{unit_type:~}'

        data_array = np.array(data_list).transpose()

        columns_names = [f'{data_type.value} ({units_list[data_type]})' for data_type in units_list]

        data_frame = pd.DataFrame(data=data_array, columns=columns_names)
        data_frame.to_csv(file_path)

    @classmethod
    def from_openmm_csv(cls, file_path, pressure=None):
        """Creates a new `StatisticsArray` object from an openmm csv file.

        Parameters
        ----------
        file_path: str
            The file path to the csv file.
        pressure: unit.Quantity, optional
            The pressure at which the statistics in the csv file were collected.

        Returns
        -------
        StatisticsArray
            The loaded statistics array.
        """
        with open(file_path, 'r') as file:

            file_contents = file.read()

            if len(file_contents) < 1:
                return cls()

            file_contents = file_contents[1:]
            file_contents = re.sub('#.*\n', '', file_contents)

            string_object = StringIO(file_contents)
            data_array = pd.read_csv(string_object)

        values = {}

        observable_to_openmm_header = {
            ObservableType.PotentialEnergy: 'Potential Energy (kJ/mole)',
            ObservableType.KineticEnergy: 'Kinetic Energy (kJ/mole)',
            ObservableType.TotalEnergy: 'Total Energy (kJ/mole)',
            ObservableType.Temperature: 'Temperature (K)',
            ObservableType.Volume: 'Box Volume (nm^3)',
            ObservableType.Density: 'Density (g/mL)'
        }

        openmm_header_to_unit = {
            'Potential Energy (kJ/mole)': unit.kilojoules / unit.mole,
            'Kinetic Energy (kJ/mole)': unit.kilojoules / unit.mole,
            'Total Energy (kJ/mole)': unit.kilojoules / unit.mole,
            'Temperature (K)': unit.kelvin,
            'Box Volume (nm^3)': unit.nanometer ** 3,
            'Density (g/mL)': unit.gram / unit.milliliter
        }

        for observable_type, header_name in observable_to_openmm_header.items():

            if header_name not in data_array:
                continue

            values[observable_type] = np.array(data_array[header_name]) * \
                                      openmm_header_to_unit[header_name]

        if pressure is not None:

            values[ObservableType.Enthalpy] = values[ObservableType.TotalEnergy] + \
                                              values[ObservableType.Volume] * pressure * unit.avogadro_number

        return_object = cls()
        return_object._internal_data = values

        return return_object

    @classmethod
    def from_pandas_csv(cls, file_path):
        """Creates a new `StatisticsArray` object from an pandas csv file.

        Parameters
        ----------
        file_path: str
            The file path to the csv file.
        """
        data_array = None

        with open(file_path, 'r') as file:

            file_contents = file.read()

            if len(file_contents) < 1:
                raise ValueError('The statistics file is empty.')

            string_object = StringIO(file_contents)
            data_array = pd.read_csv(string_object)

        values = {}

        for header_name in data_array:

            header_split = header_name.split()

            observable_name = header_split[0]

            try:
                observable_type = ObservableType(observable_name)
            except ValueError:
                continue

            unit_type = StatisticsArray._observable_units[observable_type]
            values[observable_type] = np.array(data_array[header_name]) * unit_type

        return_object = cls()
        return_object._internal_data = values

        return return_object

    @classmethod
    def from_existing(cls, existing_instance, data_indices=None):
        """Creates a new `StatisticsArray` from an existing array. If
        a set of data indices are provided, only a subset of data will
        be copied across from the existing instance.

        Parameters
        ----------
        existing_instance: StatisticsArray
            The existing array to clone
        data_indices: list of int, optional
            A set of indices which indicate which data points to copy
            from the original object. If None, all data points will be
            copied.

        Returns
        -------
        StatisticsArray
            The created array object.
        """

        new_values = {}

        for observable_type in ObservableType:

            if observable_type not in existing_instance:
                continue

            copied_values = copy.deepcopy(existing_instance[observable_type])
            new_values[observable_type] = copied_values[data_indices]

        return_object = cls()
        return_object._internal_data = new_values

        return return_object

    @classmethod
    def join(cls, *existing_instances):
        """Joins multiple statistics arrays together in the order
        that they appear in the args list.

        Parameters
        ----------
        existing_instances: StatisticsArray
            The existing arrays to join together.

        Returns
        -------
        StatisticsArray
            The created array object.
        """

        number_of_arrays = sum([1 for instance in existing_instances])

        if number_of_arrays < 2:
            raise ValueError('At least two arrays must be passed.')

        new_values = {}

        observable_types = [observable_type for observable_type in ObservableType if
                            observable_type in existing_instances[0]]

        for observable_type in observable_types:

            new_length = 0

            for existing_instance in existing_instances:

                if observable_type not in existing_instance:

                    raise ValueError('The arrays must contain the same'
                                     'types of observable.')

                new_length += len(existing_instance)

            new_array = np.zeros(new_length)
            expected_unit = StatisticsArray._observable_units[observable_type]

            counter = 0

            for existing_instance in existing_instances:

                for value in existing_instance[observable_type]:
                    new_array[counter] = value.to(expected_unit).magnitude
                    counter += 1

            new_values[observable_type] = new_array * expected_unit

        return_object = cls()
        return_object._internal_data = new_values

        return return_object


def bootstrap(bootstrap_function, iterations=200, relative_sample_size=1.0, data_sub_counts=None, **data_kwargs):
    """Performs bootstrapping on a data set to calculate the
    average value, and the standard error in the average,
    bootstrapping.

    Parameters
    ----------
    bootstrap_function: function
        The function to evaluate at each bootstrap iteration. The function
        should take a kwargs array as input, and return a float.
    iterations: int
        The number of bootstrap iterations to perform.
    relative_sample_size: float
        The percentage sample size to bootstrap over, relative to the
        size of the full data set.
    data_sub_counts: np.ndarray, optional
        If the data being bootstrapped contains arrays of concatenated sub data
        (such as when reweighting), this variable can be used to the number of
        items which belong to each subset. Data is then sampled with replacement
        so that the bootstrap sample contains the correct proportion of data from
        each subset.

        If the data to bootstrap is of the form [x0, x1, x2, y0, y1] for example,
        then `data_sub_counts=[3, 2]` and a possible sample may look like
        [x0, x0, x2, y0, y0], but never [x0, x1, y0, y1, y1].
    data_kwargs: np.ndarray, shape=(num_frames, num_dimensions), dtype=float
        A key words dictionary of the data which will be passed to the
         bootstrap function. Each kwargs argument should be a numpy array.

    Returns
    -------
    float
        The average of the data.
    float
        The uncertainty in the average.
    """

    if len(data_kwargs) is 0:
        raise ValueError('There is no data to bootstrap')

    # Make a copy of the data so we don't accidentally destroy anything.
    data_to_bootstrap = {}
    data_size = None

    for keyword in data_kwargs:

        assert isinstance(data_kwargs[keyword], np.ndarray)
        data_to_bootstrap[keyword] = np.array(data_kwargs[keyword])

        if data_size is None:
            data_size = len(data_kwargs[keyword])
        else:
            assert data_size == len(data_kwargs[keyword])

    if data_sub_counts is None:
        data_sub_counts = np.array([data_size])

    assert data_sub_counts.sum() == data_size

    average_values = np.zeros(iterations)

    for bootstrap_iteration in range(iterations):

        sample_data = {}

        for keyword in data_to_bootstrap:
            sample_data[keyword] = np.zeros(data_to_bootstrap[keyword].shape)

        start_index = 0

        for sub_count in data_sub_counts:

            # Choose the sample size as a percentage of the full data set.
            sample_size = min(math.floor(sub_count * relative_sample_size), sub_count)
            sample_indices = np.random.choice(sub_count, sample_size)

            for keyword in data_to_bootstrap:

                sub_data = data_to_bootstrap[keyword][start_index: start_index + sub_count]

                for index in range(sub_count):
                    sample_data[keyword][index + start_index] = sub_data[sample_indices][index]

            start_index += sub_count

        average_values[bootstrap_iteration] = bootstrap_function(**sample_data)

    average_value = bootstrap_function(**data_to_bootstrap)
    uncertainty = average_values.std()

    if isinstance(average_value, np.float32) or isinstance(average_value, np.float64):
        average_value = average_value.item()

    if isinstance(uncertainty, np.float32) or isinstance(uncertainty, np.float64):
        uncertainty = uncertainty.item()

    return average_value, uncertainty
