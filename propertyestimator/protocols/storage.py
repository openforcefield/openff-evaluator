"""
A collection of protocols for loading cached data off of the disk.
"""

import json
from os import path
from typing import Union

from propertyestimator.storage.dataclasses import StoredDataCollection
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder
from propertyestimator.workflow.decorators import (
    UNDEFINED,
    protocol_input,
    protocol_output,
)
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class UnpackStoredDataCollection(BaseProtocol):
    """Loads a `StoredDataCollection` object from disk,
    and makes its inner data objects easily accessible to other protocols.
    """

    input_data_path = protocol_input(
        docstring="A tuple which contains both the path to the simulation data object, "
        "it's ancillary data directory, and the force field which was used "
        "to generate the stored data.",
        type_hint=Union[list, tuple],
        default_value=UNDEFINED,
    )

    collection_data_paths = protocol_output(
        docstring="A dictionary of data object path, data directory path and "
        "force field path tuples partitioned by the unique collection "
        "keys.",
        type_hint=dict,
    )

    def execute(self, directory, available_resources):

        if len(self.input_data_path) != 3:

            return PropertyEstimatorException(
                directory=directory,
                message="The input data path should be a tuple "
                "of a path to the data object, directory, and a path "
                "to the force field used to generate it.",
            )

        data_object_path = self.input_data_path[0]
        data_directory = self.input_data_path[1]
        force_field_path = self.input_data_path[2]

        if not path.isfile(data_object_path):

            return PropertyEstimatorException(
                directory=directory,
                message="The path to the data object"
                "is invalid: {}".format(data_object_path),
            )

        if not path.isdir(data_directory):

            return PropertyEstimatorException(
                directory=directory,
                message="The path to the data directory"
                "is invalid: {}".format(data_directory),
            )

        if not path.isfile(force_field_path):

            return PropertyEstimatorException(
                directory=directory,
                message="The path to the force field"
                "is invalid: {}".format(force_field_path),
            )

        with open(data_object_path, "r") as file:
            data_object = json.load(file, cls=TypedJSONDecoder)

        if not isinstance(data_object, StoredDataCollection):

            return PropertyEstimatorException(
                directory=directory,
                message=f"The data object must be a `StoredDataCollection` "
                f"and not a {type(data_object)}",
            )

        self.collection_data_paths = {}

        for data_key, inner_data_object in data_object.data.items():

            inner_object_path = path.join(directory, f"{data_key}.json")
            inner_directory_path = path.join(data_directory, data_key)

            with open(inner_object_path, "w") as file:
                json.dump(inner_data_object, file, cls=TypedJSONEncoder)

            self.collection_data_paths[data_key] = (
                inner_object_path,
                inner_directory_path,
                force_field_path,
            )

        return self._get_output_dictionary()


@register_calculation_protocol()
class UnpackStoredSimulationData(BaseProtocol):
    """Loads a `StoredSimulationData` object from disk,
    and makes its attributes easily accessible to other protocols.
    """

    simulation_data_path = protocol_input(
        docstring="A list / tuple which contains both the path to the simulation data "
        "object, it's ancillary data directory, and the force field which "
        "was used to generate the stored data.",
        type_hint=Union[list, tuple],
        default_value=UNDEFINED,
    )

    substance = protocol_output(
        docstring="The substance which was stored.", type_hint=Substance
    )

    total_number_of_molecules = protocol_output(
        docstring="The total number of molecules in the stored system.", type_hint=int
    )

    thermodynamic_state = protocol_output(
        docstring="The thermodynamic state which was stored.",
        type_hint=ThermodynamicState,
    )

    statistical_inefficiency = protocol_output(
        docstring="The statistical inefficiency of the stored data.", type_hint=float
    )

    coordinate_file_path = protocol_output(
        docstring="A path to the stored simulation output coordinates.", type_hint=str
    )
    trajectory_file_path = protocol_output(
        docstring="A path to the stored simulation trajectory.", type_hint=str
    )
    statistics_file_path = protocol_output(
        docstring="A path to the stored simulation statistics array.", type_hint=str
    )

    force_field_path = protocol_output(
        docstring="A path to the force field parameters used to generate the stored data.",
        type_hint=str,
    )

    def execute(self, directory, available_resources):

        if len(self.simulation_data_path) != 3:

            return PropertyEstimatorException(
                directory=directory,
                message="The simulation data path should be a tuple "
                "of a path to the data object, directory, and a path "
                "to the force field used to generate it.",
            )

        data_object_path = self.simulation_data_path[0]
        data_directory = self.simulation_data_path[1]
        force_field_path = self.simulation_data_path[2]

        if not path.isdir(data_directory):

            return PropertyEstimatorException(
                directory=directory,
                message="The path to the data directory"
                "is invalid: {}".format(data_directory),
            )

        if not path.isfile(force_field_path):

            return PropertyEstimatorException(
                directory=directory,
                message="The path to the force field"
                "is invalid: {}".format(force_field_path),
            )

        with open(data_object_path, "r") as file:
            data_object = json.load(file, cls=TypedJSONDecoder)

        self.substance = data_object.substance
        self.total_number_of_molecules = data_object.total_number_of_molecules

        self.thermodynamic_state = data_object.thermodynamic_state

        self.statistical_inefficiency = data_object.statistical_inefficiency

        self.coordinate_file_path = path.join(
            data_directory, data_object.coordinate_file_name
        )
        self.trajectory_file_path = path.join(
            data_directory, data_object.trajectory_file_name
        )

        self.statistics_file_path = path.join(
            data_directory, data_object.statistics_file_name
        )

        self.force_field_path = force_field_path

        return self._get_output_dictionary()
