import abc
import json
import os

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.forcefield.system import ParameterizedSystem
from openff.evaluator.workflow import Protocol, workflow_protocol
from openff.evaluator.workflow.attributes import InputAttribute, OutputAttribute


class _GenerateRestraints(Protocol, abc.ABC):
    """The base class which will generate a set of restraint values from their
    respective schemas and for a specific APR phase.
    """

    restraint_schemas = InputAttribute(
        docstring="The full set of restraint schemas.",
        type_hint=dict,
        default_value=UNDEFINED,
    )

    restraints_path = OutputAttribute(
        docstring="The file path to the `paprika` generated restraints JSON file.",
        type_hint=str,
    )

    @classmethod
    def _restraints_to_dict(cls, restraints):
        """Converts a list of ``paprika`` restraint objects to
        a list of JSON compatible dictionary representations
        """
        from paprika.io import PaprikaEncoder

        return [
            json.loads(json.dumps(restraint.__dict__, cls=PaprikaEncoder))
            for restraint in restraints
        ]

    def _save_restraints(
        self,
        directory: str,
        static_restraints,
        conformational_restraints=None,
        symmetry_restraints=None,
        wall_restraints=None,
        guest_restraints=None,
    ):
        """Saves the restraints to a convenient JSON file."""

        conformational_restraints = (
            [] if conformational_restraints is None else conformational_restraints
        )
        symmetry_restraints = [] if symmetry_restraints is None else symmetry_restraints
        wall_restraints = [] if wall_restraints is None else wall_restraints
        guest_restraints = [] if guest_restraints is None else guest_restraints

        restraints_dictionary = {
            "static": self._restraints_to_dict(static_restraints),
            "conformational": self._restraints_to_dict(conformational_restraints),
            "symmetry": self._restraints_to_dict(symmetry_restraints),
            "wall": self._restraints_to_dict(wall_restraints),
            "guest": self._restraints_to_dict(guest_restraints),
        }

        self.restraints_path = os.path.join(directory, "restraints.json")

        with open(self.restraints_path, "w") as file:
            json.dump(restraints_dictionary, file)


@workflow_protocol()
class GenerateAttachRestraints(_GenerateRestraints):
    """Generates the restraint values to apply during the 'attach' phase from a set
    of restraint schema definitions and makes them easily accessible for the protocols
    which will apply them to the parameterized system."""

    complex_coordinate_path = InputAttribute(
        docstring="The file path to a coordinate file which contains the solvated"
        "host-guest complex and has the anchor dummy atoms added.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    attach_lambdas = InputAttribute(
        docstring="The values of lambda to use for the attach phase. These must"
        "start from 0.0 and increase monotonically to and include 1.0.",
        type_hint=list,
        default_value=UNDEFINED,
    )

    def _execute(self, directory, available_resources):
        from paprika.evaluator import Setup

        # Construct the restraints to keep the host in place and
        # with an open cavity.
        static_restraints = Setup.build_static_restraints(
            self.complex_coordinate_path,
            len(self.attach_lambdas),
            None,
            None,
            self.restraint_schemas.get("static", []),
        )
        conformational_restraints = Setup.build_conformational_restraints(
            self.complex_coordinate_path,
            self.attach_lambdas,
            None,
            None,
            self.restraint_schemas.get("conformational", []),
        )

        # Construct the restraints to keep the guest at the correct
        # distance and orientation relative to the host.
        symmetry_restraints = Setup.build_symmetry_restraints(
            self.complex_coordinate_path,
            len(self.attach_lambdas),
            self.restraint_schemas.get("symmetry", []),
        )
        wall_restraints = Setup.build_wall_restraints(
            self.complex_coordinate_path,
            len(self.attach_lambdas),
            self.restraint_schemas.get("wall", []),
        )
        guest_restraints = Setup.build_guest_restraints(
            self.complex_coordinate_path,
            self.attach_lambdas,
            None,
            self.restraint_schemas.get("guest", []),
        )

        self._save_restraints(
            directory,
            static_restraints,
            conformational_restraints,
            symmetry_restraints,
            wall_restraints,
            guest_restraints,
        )


@workflow_protocol()
class GeneratePullRestraints(GenerateAttachRestraints):
    """Generates the restraint values to apply during the 'pull' phase from a set
    of restraint schema definitions and makes them easily accessible for the protocols
    which will apply them to the parameterized system."""

    n_pull_windows = InputAttribute(
        docstring="The number of lambda to use for the pull phase.",
        type_hint=int,
        default_value=UNDEFINED,
    )

    def _execute(self, directory, available_resources):
        from paprika.evaluator import Setup

        # Construct the restraints to keep the host in place and
        # with an open cavity.
        static_restraints = Setup.build_static_restraints(
            self.complex_coordinate_path,
            len(self.attach_lambdas),
            self.n_pull_windows,
            None,
            self.restraint_schemas.get("static", []),
        )
        conformational_restraints = Setup.build_conformational_restraints(
            self.complex_coordinate_path,
            self.attach_lambdas,
            self.n_pull_windows,
            None,
            self.restraint_schemas.get("conformational", []),
        )

        # Construct the restraints to keep the guest at the correct
        # distance to the host.
        guest_restraints = Setup.build_guest_restraints(
            self.complex_coordinate_path,
            self.attach_lambdas,
            self.n_pull_windows,
            self.restraint_schemas.get("guest", []),
        )

        # Remove the attach phases from the restraints as these restraints are
        # only being used for the pull phase.
        for restraint in (
            static_restraints + conformational_restraints + guest_restraints
        ):
            for key in restraint.phase["attach"]:
                restraint.phase["attach"][key] = None

        self._save_restraints(
            directory,
            static_restraints,
            conformational_restraints,
            None,
            None,
            guest_restraints,
        )


@workflow_protocol()
class GenerateReleaseRestraints(_GenerateRestraints):
    """Generates the restraint values to apply during the 'release' phase from a set
    of restraint schema definitions and makes them easily accessible for the protocols
    which will apply them to the parameterized system."""

    host_coordinate_path = InputAttribute(
        docstring="The file path to a coordinate file which contains the solvated"
        "host molecule and has the anchor dummy atoms added.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    release_lambdas = InputAttribute(
        docstring="The values of lambda to use for the release phase. These must"
        "start from 1.0 and decrease monotonically to and include 0.0.",
        type_hint=list,
        default_value=UNDEFINED,
    )

    def _execute(self, directory, available_resources):
        from paprika.evaluator import Setup

        # Construct the restraints to keep the host in place and
        # with an open cavity.
        static_restraints = Setup.build_static_restraints(
            self.host_coordinate_path,
            None,
            None,
            len(self.release_lambdas),
            self.restraint_schemas.get("static", []),
        )
        conformational_restraints = Setup.build_conformational_restraints(
            self.host_coordinate_path,
            None,
            None,
            self.release_lambdas,
            self.restraint_schemas.get("conformational", []),
        )

        self._save_restraints(
            directory,
            static_restraints,
            conformational_restraints,
        )


@workflow_protocol()
class GenerateBoundRestraints(GenerateAttachRestraints):
    """Generates the restraint values to apply to the bound host-guest system for the gradient
    calculation from a set of restraint schema definitions and makes them easily accessible
    for the protocols which will apply them to the parameterized system."""

    def _execute(self, directory, available_resources):
        from paprika.evaluator import Setup

        # Construct the restraints to keep the host in place
        static_restraints = Setup.build_static_restraints(
            self.complex_coordinate_path,
            len(self.attach_lambdas),
            None,
            None,
            self.restraint_schemas.get("static", []),
        )

        # Construct the restraints to keep the guest at the correct
        # distance and orientation relative to the host.
        symmetry_restraints = Setup.build_symmetry_restraints(
            self.complex_coordinate_path,
            len(self.attach_lambdas),
            self.restraint_schemas.get("symmetry", []),
        )
        wall_restraints = Setup.build_wall_restraints(
            self.complex_coordinate_path,
            len(self.attach_lambdas),
            self.restraint_schemas.get("wall", []),
        )

        self._save_restraints(
            directory,
            static_restraints,
            None,
            symmetry_restraints,
            wall_restraints,
            None,
        )


@workflow_protocol()
class GenerateUnboundRestraints(GeneratePullRestraints):
    """Generates the restraint values to apply to the unbound host-guest system for the gradient
    calculation from a set of restraint schema definitions and makes them easily accessible
    for the protocols which will apply them to the parameterized system."""

    def _execute(self, directory, available_resources):
        from paprika.evaluator import Setup

        # Construct the restraints to keep the host in place
        static_restraints = Setup.build_static_restraints(
            self.complex_coordinate_path,
            len(self.attach_lambdas),
            self.n_pull_windows,
            None,
            self.restraint_schemas.get("static", []),
        )

        # Construct the restraints to keep the guest at the correct
        # distance and orientation relative to the host.
        guest_restraints = Setup.build_guest_restraints(
            self.complex_coordinate_path,
            self.attach_lambdas,
            self.n_pull_windows,
            self.restraint_schemas.get("guest", []),
        )

        # Remove the attach phases from the restraints as these restraints are
        # only being used for the pull phase.
        for restraint in static_restraints + guest_restraints:
            for key in restraint.phase["attach"]:
                restraint.phase["attach"][key] = None

        self._save_restraints(
            directory,
            static_restraints,
            None,
            None,
            None,
            guest_restraints,
        )


@workflow_protocol()
class ApplyRestraints(Protocol):
    """A protocol which will apply the restraints defined in a restraints JSON
    file to a specified system.
    """

    restraints_path = InputAttribute(
        docstring="The file path to the JSON file which contains the restraint "
        "definitions. This will usually have been generated by a "
        "`GenerateXXXRestraints` protocol.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    phase = InputAttribute(
        docstring="The APR phase to take the restraints from.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    window_index = InputAttribute(
        docstring="The index of the window to take the restraints from.",
        type_hint=int,
        default_value=UNDEFINED,
    )

    input_system = InputAttribute(
        docstring="The parameterized system which the restraints should be added "
        "to.",
        type_hint=ParameterizedSystem,
        default_value=UNDEFINED,
    )

    output_system = OutputAttribute(
        docstring="The parameterized system which now includes the added restraints.",
        type_hint=ParameterizedSystem,
    )

    @classmethod
    def _parse_restraints(cls, restraint_dictionaries):
        """Parses the dictionary representations of a list of `paprika` restraint
        objects into a list of full restraint objects."""

        from paprika.restraints import DAT_restraint

        restraints = []

        for restraint_dictionary in restraint_dictionaries:
            restraint = DAT_restraint()
            restraint.__dict__ = restraint_dictionary

            properties = [
                "mask1",
                "mask2",
                "mask3",
                "mask4",
                "topology",
                "instances",
                "custom_restraint_values",
                "auto_apr",
                "continuous_apr",
                "attach",
                "pull",
                "release",
                "amber_index",
            ]

            for class_property in properties:
                if f"_{class_property}" in restraint.__dict__.keys():
                    restraint.__dict__[class_property] = restraint.__dict__[
                        f"_{class_property}"
                    ]

            restraints.append(restraint)

        return restraints

    @classmethod
    def load_restraints(cls, file_path: str):
        """Loads a set of `paprika` restraint objects from a JSON file.

        Parameters
        ----------
        file_path
            The path to the JSON serialized restraints.

        Returns
        -------
            The loaded `paprika` restraint objects.
        """

        from paprika.io import PaprikaDecoder

        with open(file_path) as file:
            restraints_dictionary = json.load(file, cls=PaprikaDecoder)

        restraints = {
            restraint_type: cls._parse_restraints(restraints_dictionary[restraint_type])
            for restraint_type in restraints_dictionary
        }

        return restraints

    def _execute(self, directory, available_resources):
        from paprika.restraints.openmm import (
            apply_dat_restraint,
            apply_positional_restraints,
        )
        from simtk.openmm import XmlSerializer

        # Load in the system to add the restraints to.
        system = self.input_system.system

        # Define a custom force group per type of restraint to help
        # with debugging / analysis.
        force_groups = {
            "static": 10,
            "conformational": 11,
            "guest": 12,
            "symmetry": 13,
            "wall": 14,
        }

        # Apply the serialized restraints.
        restraints = self.load_restraints(self.restraints_path)

        for restraint_type in force_groups:
            if restraint_type not in restraints:
                continue

            for restraint in restraints[restraint_type]:
                apply_dat_restraint(
                    system,
                    restraint,
                    self.phase,
                    self.window_index,
                    # flat_bottom=restraint_type in ["symmetry", "wall"],
                    force_group=force_groups[restraint_type],
                )

        # Apply the positional restraints to the dummy atoms.
        apply_positional_restraints(
            self.input_system.topology_path, system, force_group=15
        )

        output_system_path = os.path.join(directory, "output.xml")

        with open(output_system_path, "w") as file:
            file.write(XmlSerializer.serialize(system))

        self.output_system = ParameterizedSystem(
            substance=self.input_system.substance,
            force_field=self.input_system.force_field,
            topology_path=self.input_system.topology_path,
            system_path=output_system_path,
        )
