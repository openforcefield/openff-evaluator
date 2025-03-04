"""
A collection of protocols for loading cached data off of the disk.
"""

from os import path
from typing import Union

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.storage.data import StoredEquilibrationData, StoredSimulationData
from openff.evaluator.substances import Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.exceptions import EquilibrationDataExistsException
from openff.evaluator.utils.observables import ObservableFrame
from openff.evaluator.workflow import Protocol, workflow_protocol
from openff.evaluator.workflow.attributes import InputAttribute, OutputAttribute


@workflow_protocol()
class CheckStoredEquilibrationData(Protocol):
    """Checks if a `StoredEquilibrationData` object exists on disk."""

    simulation_data_path = InputAttribute(
        docstring="A path to the simulation data object.",
        type_hint=Union[list, tuple],
        default_value=UNDEFINED,
    )

    data_missing = OutputAttribute(
        docstring="Whether the data object is missing on disk.",
        type_hint=bool,
    )

    def _execute(self, directory, available_resources):
        self.data_missing = True
        if len(self.simulation_data_path) == 3:
            data_object_path = self.simulation_data_path[0]
            self.data_missing = not path.isfile(data_object_path)

        # short-circuit rest of graph -- if the data exists, we don't need to re-run
        if not self.data_missing:
            raise EquilibrationDataExistsException(
                "The equilibration data already exists on disk."
            )


@workflow_protocol()
class UnpackStoredEquilibrationData(Protocol):
    """Loads a `StoredEquilibrationData` object from disk,
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

    coordinate_file_path = OutputAttribute(
        docstring="A path to the stored simulation output coordinates.", type_hint=str
    )

    force_field_path = OutputAttribute(
        docstring="A path to the force field parameters used to generate the stored "
        "data.",
        type_hint=str,
    )

    def _execute(self, directory, available_resources):
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

        data_object = StoredEquilibrationData.from_json(data_object_path)

        if not isinstance(data_object, StoredEquilibrationData):
            raise ValueError(
                f"The data path must point to a `StoredEquilibrationData` "
                f"object, and not a {data_object.__class__.__name__}",
            )

        self.substance = data_object.substance
        self.total_number_of_molecules = data_object.number_of_molecules

        self.thermodynamic_state = data_object.thermodynamic_state

        self.coordinate_file_path = path.join(
            data_directory, data_object.coordinate_file_name
        )

        self.force_field_path = force_field_path


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

    observables = OutputAttribute(
        docstring="The stored observables frame.", type_hint=ObservableFrame
    )

    coordinate_file_path = OutputAttribute(
        docstring="A path to the stored simulation output coordinates.", type_hint=str
    )
    trajectory_file_path = OutputAttribute(
        docstring="A path to the stored simulation trajectory.", type_hint=str
    )

    force_field_path = OutputAttribute(
        docstring="A path to the force field parameters used to generate the stored "
        "data.",
        type_hint=str,
    )

    def _execute(self, directory, available_resources):
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

        data_object = StoredSimulationData.from_json(data_object_path)

        if not isinstance(data_object, StoredSimulationData):
            raise ValueError(
                f"The data path must point to a `StoredSimulationData` "
                f"object, and not a {data_object.__class__.__name__}",
            )

        self.substance = data_object.substance
        self.total_number_of_molecules = data_object.number_of_molecules

        self.thermodynamic_state = data_object.thermodynamic_state

        self.observables = data_object.observables

        self.coordinate_file_path = path.join(
            data_directory, data_object.coordinate_file_name
        )
        self.trajectory_file_path = path.join(
            data_directory, data_object.trajectory_file_name
        )

        self.force_field_path = force_field_path
