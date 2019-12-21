"""
A collection of protocols for loading cached data off of the disk.
"""

import json
from os import path
from typing import Union

from propertyestimator.attributes import UNDEFINED
from propertyestimator.storage.data import StoredSimulationData
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.serialization import TypedJSONDecoder
from propertyestimator.workflow.attributes import InputAttribute, OutputAttribute
from propertyestimator.workflow.plugins import workflow_protocol
from propertyestimator.workflow.protocols import Protocol


@workflow_protocol()
class UnpackStoredSimulationData(Protocol):
    """Loads a `StoredSimulationData` object from disk,
    and makes its attributes easily accessible to other protocols.
    """

    simulation_data_path = InputAttribute(
        docstring="A list / tuple which contains both the path to the simulation data "
        "object, it's ancillary data directory, and the force field which "
        "was used to generate the stored data.",
        type_hint=Union[list, tuple],
        default_value=UNDEFINED,
    )

    substance = OutputAttribute(
        docstring="The substance which was stored.", type_hint=Substance
    )

    total_number_of_molecules = OutputAttribute(
        docstring="The total number of molecules in the stored system.", type_hint=int
    )

    thermodynamic_state = OutputAttribute(
        docstring="The thermodynamic state which was stored.",
        type_hint=ThermodynamicState,
    )

    statistical_inefficiency = OutputAttribute(
        docstring="The statistical inefficiency of the stored data.", type_hint=float
    )

    coordinate_file_path = OutputAttribute(
        docstring="A path to the stored simulation output coordinates.", type_hint=str
    )
    trajectory_file_path = OutputAttribute(
        docstring="A path to the stored simulation trajectory.", type_hint=str
    )
    statistics_file_path = OutputAttribute(
        docstring="A path to the stored simulation statistics array.", type_hint=str
    )

    force_field_path = OutputAttribute(
        docstring="A path to the force field parameters used to generate the stored data.",
        type_hint=str,
    )

    def execute(self, directory, available_resources):

        if len(self.simulation_data_path) != 3:

            raise ValueError(
                "The simulation data path should be a tuple of a path to the data "
                "object, directory, and a path to the force field used to generate it."
            )

        data_object_path = self.simulation_data_path[0]
        data_directory = self.simulation_data_path[1]
        force_field_path = self.simulation_data_path[2]

        if not path.isdir(data_directory):

            raise ValueError(
                f"The path to the data directory is invalid: {data_directory}"
            )

        if not path.isfile(force_field_path):

            raise ValueError(
                f"The path to the force field is invalid: {force_field_path}"
            )

        with open(data_object_path, "r") as file:
            data_object = json.load(file, cls=TypedJSONDecoder)

        if not isinstance(data_object, StoredSimulationData):

            raise ValueError(
                f"The data path must point to a `StoredSimulationData` "
                f"object, and not a {data_object.__class__.__name__}",
            )

        self.substance = data_object.substance
        self.total_number_of_molecules = data_object.number_of_molecules

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
