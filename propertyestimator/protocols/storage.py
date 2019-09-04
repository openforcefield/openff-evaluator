"""
A collection of protocols for loading cached data off of the disk.
"""

import json
from os import path

from propertyestimator.storage.dataclasses import StoredDataCollection
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class UnpackStoredDataCollection(BaseProtocol):
    """Loads a `StoredDataCollection` object from disk,
    and makes its inner data objects easily accessible to other protocols.
    """

    @protocol_input(tuple)
    def input_data_path(self):
        """A tuple which contains both the path to the simulation data object,
        it's ancillary data directory, and the force field which was used to
        generate the stored data."""
        pass

    @protocol_output(dict)
    def collection_data_paths(self):
        """A dictionary of data object path, data directory path and force field
        path tuples partitioned by the unique collection keys."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new UnpackStoredDataCollection object."""
        super().__init__(protocol_id)

        self._input_data_path = None
        self._collection_data_paths = None

    def execute(self, directory, available_resources):

        if len(self._input_data_path) != 3:

            return PropertyEstimatorException(directory=directory,
                                              message='The input data path should be a tuple '
                                                      'of a path to the data object, directory, and a path '
                                                      'to the force field used to generate it.')

        data_object_path = self._input_data_path[0]
        data_directory = self._input_data_path[1]
        force_field_path = self._input_data_path[2]

        if not path.isfile(data_object_path):

            return PropertyEstimatorException(directory=directory,
                                              message='The path to the data object'
                                                      'is invalid: {}'.format(data_object_path))

        if not path.isdir(data_directory):

            return PropertyEstimatorException(directory=directory,
                                              message='The path to the data directory'
                                                      'is invalid: {}'.format(data_directory))

        if not path.isfile(force_field_path):

            return PropertyEstimatorException(directory=directory,
                                              message='The path to the force field'
                                                      'is invalid: {}'.format(force_field_path))

        with open(data_object_path, 'r') as file:
            data_object = json.load(file, cls=TypedJSONDecoder)

        if not isinstance(data_object, StoredDataCollection):

            return PropertyEstimatorException(directory=directory,
                                              message=f'The data object must be a `StoredDataCollection` '
                                                      f'and not a {type(data_object)}')

        self._collection_data_paths = {}

        for data_key, inner_data_object in data_object.data.items():

            inner_object_path = path.join(directory, f'{data_key}.json')
            inner_directory_path = path.join(data_directory, data_key)

            with open(inner_object_path, 'w') as file:
                json.dump(inner_data_object, file, cls=TypedJSONEncoder)

            self._collection_data_paths[data_key] = (inner_object_path,
                                                     inner_directory_path,
                                                     force_field_path)

        return self._get_output_dictionary()


@register_calculation_protocol()
class UnpackStoredSimulationData(BaseProtocol):
    """Loads a `StoredSimulationData` object from disk,
    and makes its attributes easily accessible to other protocols.
    """

    @protocol_input(tuple)
    def simulation_data_path(self):
        """A tuple which contains both the path to the simulation data object,
        it's ancillary data directory, and the force field which was used to
        generate the stored data."""
        pass

    @protocol_output(Substance)
    def substance(self):
        """The substance which was stored."""
        pass

    @protocol_output(int)
    def total_number_of_molecules(self):
        """The total number of molecules in the stored system."""
        pass

    @protocol_output(ThermodynamicState)
    def thermodynamic_state(self):
        """The thermodynamic state which was stored."""
        pass

    @protocol_output(float)
    def statistical_inefficiency(self):
        """The statistical inefficiency of the stored data."""
        pass

    @protocol_output(str)
    def coordinate_file_path(self):
        """A path to the stored simulation trajectory."""
        pass

    @protocol_output(str)
    def trajectory_file_path(self):
        """A path to the stored simulation trajectory."""
        pass

    @protocol_output(str)
    def statistics_file_path(self):
        """A path to the stored simulation statistics array."""
        pass

    @protocol_output(str)
    def force_field_path(self):
        """A path to the force field parameters used to generate
        the stored data."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new UnpackStoredSimulationData object."""
        super().__init__(protocol_id)

        self._simulation_data_path = None

        self._substance = None
        self._total_number_of_molecules = None

        self._thermodynamic_state = None

        self._statistical_inefficiency = None

        self._coordinate_file_path = None
        self._trajectory_file_path = None

        self._statistics_file_path = None

        self._force_field_path = None

    def execute(self, directory, available_resources):

        if len(self._simulation_data_path) != 3:

            return PropertyEstimatorException(directory=directory,
                                              message='The simulation data path should be a tuple '
                                                      'of a path to the data object, directory, and a path '
                                                      'to the force field used to generate it.')

        data_object_path = self._simulation_data_path[0]
        data_directory = self._simulation_data_path[1]
        force_field_path = self._simulation_data_path[2]

        if not path.isdir(data_directory):

            return PropertyEstimatorException(directory=directory,
                                              message='The path to the data directory'
                                                      'is invalid: {}'.format(data_directory))

        if not path.isfile(force_field_path):

            return PropertyEstimatorException(directory=directory,
                                              message='The path to the force field'
                                                      'is invalid: {}'.format(force_field_path))

        with open(data_object_path, 'r') as file:
            data_object = json.load(file, cls=TypedJSONDecoder)

        self._substance = data_object.substance
        self._total_number_of_molecules = data_object.total_number_of_molecules

        self._thermodynamic_state = data_object.thermodynamic_state

        self._statistical_inefficiency = data_object.statistical_inefficiency

        self._coordinate_file_path = path.join(data_directory, data_object.coordinate_file_name)
        self._trajectory_file_path = path.join(data_directory, data_object.trajectory_file_name)

        self._statistics_file_path = path.join(data_directory, data_object.statistics_file_name)

        self._force_field_path = force_field_path

        return self._get_output_dictionary()
