import abc
import os
from collections import defaultdict
from typing import Dict, List

import numpy
from openff.units import unit

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.forcefield.system import ParameterizedSystem
from openff.evaluator.substances import Component, Substance
from openff.evaluator.workflow import Protocol, workflow_protocol
from openff.evaluator.workflow.attributes import InputAttribute, OutputAttribute


def _components_by_role(substance: Substance) -> Dict[Component.Role, List[Component]]:
    """Extract the a components from the input substance which have been
    flagged with a specific role.

    Returns
    -------
        The components in the substance split by their assigned role.
    """

    components = defaultdict(list)

    for component in substance:
        components[component.role].append(component)

    return components


def _atom_indices_by_role(
    substance: Substance, coordinate_path: str
) -> Dict[Component.Role, List[int]]:
    """Retrieve the indices of the atoms which belong to components with a
    specific role.
    """
    import mdtraj
    from openff.toolkit.topology import Molecule, Topology

    # Split the substance into the components assigned each role.
    components_by_role = _components_by_role(substance)

    # Create OFF representations of the components
    off_molecules = {
        component: Molecule.from_smiles(component.smiles)
        for components in components_by_role.values()
        for component in components
    }

    # Load in the complex structure.
    mdtraj_trajectory: mdtraj.Trajectory = mdtraj.load_pdb(coordinate_path)

    off_topology: Topology = Topology.from_mdtraj(
        mdtraj_trajectory.topology, off_molecules.values()
    )

    atom_indices = defaultdict(list)

    for component_role in components_by_role:

        for component in components_by_role[component_role]:

            # Find the indices of all instances of this component.
            off_molecule = off_molecules[component]

            for topology_molecule in off_topology.topology_molecules:

                if (
                    topology_molecule.reference_molecule.to_smiles()
                    != off_molecule.to_smiles()
                ):
                    continue

                atom_indices[component_role].extend(
                    [
                        i + topology_molecule.atom_start_topology_index
                        for i in range(topology_molecule.n_atoms)
                    ]
                )

    return atom_indices


class _PrepareAPRCoordinates(Protocol, abc.ABC):
    """The base class for protocols which will be used to prepare the coordinates
    for an APR calculation.
    """

    substance = InputAttribute(
        docstring="The substance which defines the host, guest and solvent.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )
    complex_file_path = InputAttribute(
        docstring="The path to the file which the coordinates of the guest molecule"
        "bound to the host molecule.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    output_coordinate_path = OutputAttribute(
        docstring="The file path to the system which has been correctly aligned to "
        "the z-axis.",
        type_hint=str,
    )


@workflow_protocol()
class PreparePullCoordinates(_PrepareAPRCoordinates):
    """A protocol which will align a host-guest complex to the z-axis and position
    the guest molecule at a specified point along the pull axis.
    """

    guest_orientation_mask = InputAttribute(
        docstring="The string mask which describes which guest atoms will be "
        "restrained relative to the dummy atoms to keep the molecule aligned to the "
        "z-axis. This should be of the form 'X Y' where X Y are ParmEd selectors for "
        "the first and second guest atoms.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    pull_distance = InputAttribute(
        docstring="The total distance that the guest will be pulled along the z-axis "
        "during the pull phase.",
        type_hint=unit.Quantity,
        default_value=UNDEFINED,
    )
    pull_window_index = InputAttribute(
        docstring="The index of the pull window to generate coordinates for.",
        type_hint=int,
        default_value=UNDEFINED,
    )
    n_pull_windows = InputAttribute(
        docstring="The total number of the pull windows in the calculation.",
        type_hint=int,
        default_value=UNDEFINED,
    )

    def _execute(self, directory, available_resources):

        from openmm import app
        from paprika.evaluator import Setup

        atom_indices_by_role = _atom_indices_by_role(
            self.substance, self.complex_file_path
        )

        guest_atom_indices = atom_indices_by_role[Component.Role.Ligand]

        host_structure = Setup.prepare_complex_structure(
            self.complex_file_path,
            guest_atom_indices,
            self.guest_orientation_mask,
            self.pull_distance.to(unit.angstrom).magnitude,
            self.pull_window_index,
            self.n_pull_windows,
        )

        self.output_coordinate_path = os.path.join(directory, "output.pdb")

        with open(self.output_coordinate_path, "w") as file:
            app.PDBFile.writeFile(
                host_structure.topology, host_structure.positions, file, True
            )


@workflow_protocol()
class PrepareReleaseCoordinates(_PrepareAPRCoordinates):
    """A protocol which will extract the host molecule from a file containing both
    the host and guest molecules and produce a coordinate file containing only the
    host which has been correctly aligned to the z-axis.
    """

    def _execute(self, directory, available_resources):

        import mdtraj
        from openmm import app
        from paprika.evaluator import Setup

        mdtraj_trajectory = mdtraj.load_pdb(self.complex_file_path)

        atom_indices_by_role = _atom_indices_by_role(
            self.substance, self.complex_file_path
        )
        host_atom_indices = atom_indices_by_role[Component.Role.Receptor]

        host_trajectory = mdtraj_trajectory.atom_slice(host_atom_indices)

        host_file_path = os.path.join(directory, "host_input.pdb")
        host_trajectory.save(host_file_path)

        host_structure = Setup.prepare_host_structure(host_file_path)

        self.output_coordinate_path = os.path.join(directory, "output.pdb")

        with open(self.output_coordinate_path, "w") as file:
            app.PDBFile.writeFile(
                host_structure.topology, host_structure.positions, file, True
            )


@workflow_protocol()
class AddDummyAtoms(Protocol):
    """A protocol which will add the reference 'dummy' atoms to a parameterised
    system. This protocol assumes the host / complex has already been correctly
    aligned to the z-axis and has been placed at the origin.
    """

    substance = InputAttribute(
        docstring="The substance which defines the host, guest and solvent.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )

    offset = InputAttribute(
        docstring="The distance to offset the dummy atoms from the origin (0, 0, 0) "
        "backwards along the z-axis.",
        type_hint=unit.Quantity,
        default_value=UNDEFINED,
    )

    input_coordinate_path = InputAttribute(
        docstring="The file path to the coordinates which the dummy atoms "
        "should be added to.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    input_system = InputAttribute(
        docstring="The parameterized system which the dummy atoms "
        "should be added to.",
        type_hint=ParameterizedSystem,
        default_value=UNDEFINED,
    )

    output_coordinate_path = OutputAttribute(
        docstring="The file path to the coordinates which include the added dummy "
        "atoms.",
        type_hint=str,
    )
    output_system = OutputAttribute(
        docstring="The parameterized system which include the added dummy atoms.",
        type_hint=ParameterizedSystem,
    )

    def _execute(self, directory, available_resources):

        import parmed.geometry
        from openmm import NonbondedForce, XmlSerializer, app
        from paprika.evaluator import Setup

        # Extract the host atoms to determine the offset of the dummy atoms.
        # noinspection PyTypeChecker
        input_structure: parmed.Structure = parmed.load_file(
            self.input_coordinate_path, structure=True
        )

        # Add the dummy atoms to the structure.
        offset = self.offset.to(unit.angstrom).magnitude

        Setup.add_dummy_atoms_to_structure(
            input_structure,
            [
                numpy.array([0, 0, -offset]),
                numpy.array([0, 0, -3.0 - offset]),
                numpy.array([0, 2.2, -5.2 - offset]),
            ],
            numpy.zeros(3),
        )

        # Shift the structure to avoid issues with the PBC
        input_structure.coordinates += numpy.array(
            [
                input_structure.box[0] * 0.5,
                input_structure.box[1] * 0.5,
                -input_structure.coordinates[-1, 2] + 1.0,
            ]
        )

        # Save the final coordinates.
        self.output_coordinate_path = os.path.join(directory, "output.pdb")

        with open(self.output_coordinate_path, "w") as file:
            app.PDBFile.writeFile(
                input_structure.topology, input_structure.positions, file, True
            )

        # Add the dummy atoms to the system.
        system = self.input_system.system

        for _ in range(3):
            system.addParticle(mass=207)

        for force_index in range(system.getNumForces()):

            force = system.getForce(force_index)

            if not isinstance(force, NonbondedForce):
                continue

            force.addParticle(0.0, 1.0, 0.0)
            force.addParticle(0.0, 1.0, 0.0)
            force.addParticle(0.0, 1.0, 0.0)

        output_system_path = os.path.join(directory, "output.xml")

        with open(output_system_path, "w") as file:
            file.write(XmlSerializer.serialize(system))

        self.output_system = ParameterizedSystem(
            self.input_system.substance,
            self.input_system.force_field,
            self.output_coordinate_path,
            output_system_path,
        )
