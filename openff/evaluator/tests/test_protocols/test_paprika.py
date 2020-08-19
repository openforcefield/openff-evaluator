import os

import numpy
import pytest

from openff.evaluator import unit
from openff.evaluator.protocols.paprika.coordinates import (
    AddDummyAtoms,
    PreparePullCoordinates,
    PrepareReleaseCoordinates,
    _atom_indices_by_role,
    _components_by_role,
)
from openff.evaluator.protocols.paprika.restraints import (
    ApplyRestraints,
    GenerateAttachRestraints,
    GeneratePullRestraints,
    GenerateReleaseRestraints,
)
from openff.evaluator.substances import Component, ExactAmount, Substance
from openff.evaluator.utils import get_data_filename


@pytest.fixture(scope="module")
def dummy_complex() -> Substance:

    substance = Substance()

    substance.add_component(
        Component(smiles="C", role=Component.Role.Ligand), ExactAmount(1)
    )
    substance.add_component(
        Component(smiles="CO", role=Component.Role.Receptor), ExactAmount(1)
    )

    return substance


@pytest.fixture()
def complex_file_path(tmp_path):

    import parmed.geometry
    from paprika.setup import Setup

    complex_path = get_data_filename(
        os.path.join("test", "molecules", "methanol_methane.pdb")
    )

    # noinspection PyTypeChecker
    structure: parmed.Structure = parmed.load_file(complex_path, structure=True)
    # noinspection PyTypeChecker
    center_of_mass = parmed.geometry.center_of_mass(
        structure.coordinates, masses=numpy.ones(len(structure.coordinates))
    )

    Setup.add_dummy_atoms_to_structure(
        structure,
        [
            numpy.array([0.0, 0.0, 10.0]),
            numpy.array([0.0, 0.0, 20.0]),
            numpy.array([0.0, 5.0, 25.0]),
        ],
        center_of_mass,
    )

    complex_path = os.path.join(tmp_path, "complex.pdb")
    structure.save(complex_path)

    return complex_path


@pytest.fixture(scope="module")
def restraints_schema():
    return {
        "static": [{"atoms": "@12 @1", "force_constant": 5.0}],
        "conformational": [
            {"atoms": "@1 @2 @3 @4", "force_constant": 6.0, "target": 104.3}
        ],
        "symmetry": [{"atoms": "@12 @7 @3 @4", "force_constant": 50.0, "target": 11.0}],
        "wall": [{"atoms": "@12 @7 @3 @4", "force_constant": 50.0, "target": 11.0}],
        "guest": [
            {
                "atoms": "@12 @7",
                "attach": {"force_constant": 5.0, "target": 6.0},
                "pull": {"force_constant": 5.0, "target": 24.0},
            }
        ],
    }


@pytest.fixture()
def attach_restraints_path(tmp_path, complex_file_path, restraints_schema):

    protocol = GenerateAttachRestraints("")
    protocol.complex_coordinate_path = complex_file_path
    protocol.attach_lambdas = [1.0]
    protocol.restraint_schemas = restraints_schema
    protocol.execute(str(tmp_path))

    return protocol.restraints_path


@pytest.fixture()
def pull_restraints_path(tmp_path, complex_file_path, restraints_schema):

    protocol = GeneratePullRestraints("")
    protocol.complex_coordinate_path = complex_file_path
    protocol.attach_lambdas = [0.0]
    protocol.n_pull_windows = 2
    protocol.restraint_schemas = restraints_schema
    protocol.execute(str(tmp_path))

    return protocol.restraints_path


@pytest.fixture()
def release_restraints_path(tmp_path, complex_file_path, restraints_schema):

    protocol = GenerateReleaseRestraints("")
    protocol.host_coordinate_path = complex_file_path
    protocol.release_lambdas = [1.0]
    protocol.restraint_schemas = restraints_schema
    protocol.execute(str(tmp_path))

    return protocol.restraints_path


def test_components_by_role(dummy_complex):

    components_by_role = _components_by_role(dummy_complex)

    assert len(components_by_role) == 2

    assert Component.Role.Ligand in components_by_role
    assert Component.Role.Receptor in components_by_role

    assert len(components_by_role[Component.Role.Receptor]) == 1
    assert components_by_role[Component.Role.Receptor][0].smiles == "CO"

    assert len(components_by_role[Component.Role.Ligand]) == 1
    assert components_by_role[Component.Role.Ligand][0].smiles == "C"


def test_atom_indices_by_role(dummy_complex):

    atom_indices_by_role = _atom_indices_by_role(
        dummy_complex,
        get_data_filename(os.path.join("test", "molecules", "methanol_methane.pdb")),
    )

    assert len(atom_indices_by_role) == 2

    assert Component.Role.Ligand in atom_indices_by_role
    assert Component.Role.Receptor in atom_indices_by_role

    assert len(atom_indices_by_role[Component.Role.Receptor]) == 6
    assert atom_indices_by_role[Component.Role.Receptor] == [0, 1, 2, 3, 4, 5]

    assert len(atom_indices_by_role[Component.Role.Ligand]) == 5
    assert atom_indices_by_role[Component.Role.Ligand] == [6, 7, 8, 9, 10]


def test_prepare_release_coordinates(tmp_path, dummy_complex):
    import mdtraj

    protocol = PrepareReleaseCoordinates("")

    protocol.substance = dummy_complex
    protocol.complex_file_path = get_data_filename(
        os.path.join("test", "molecules", "methanol_methane.pdb")
    )

    protocol.execute(str(tmp_path))

    assert os.path.isfile(protocol.output_coordinate_path)

    host_trajectory = mdtraj.load_pdb(protocol.output_coordinate_path)
    assert host_trajectory.topology.n_atoms == 6


@pytest.mark.parametrize("window_index,expected_z", [(0, 0.0), (1, 2.4)])
def test_prepare_pull_coordinates(tmp_path, dummy_complex, window_index, expected_z):
    import mdtraj

    protocol = PreparePullCoordinates("")
    protocol.substance = dummy_complex
    protocol.complex_file_path = get_data_filename(
        os.path.join("test", "molecules", "methanol_methane.pdb")
    )
    protocol.guest_orientation_mask = "@7 @8"
    protocol.pull_distance = 24.0 * unit.angstrom
    protocol.pull_window_index = window_index
    protocol.n_pull_windows = 2

    protocol.execute(str(tmp_path))

    assert os.path.isfile(protocol.output_coordinate_path)

    host_trajectory = mdtraj.load_pdb(protocol.output_coordinate_path)
    assert host_trajectory.topology.n_atoms == 11

    assert numpy.allclose(
        host_trajectory.xyz[0][6, :], numpy.array([0.0, 0.0, expected_z])
    )
    assert numpy.allclose(host_trajectory.xyz[0][7, :2], numpy.zeros(2))


def test_add_dummy_atoms(tmp_path, dummy_complex):

    import mdtraj
    from simtk import openmm
    from simtk import unit as simtk_unit

    # Create an empty system to add the dummy atoms to.
    system_path = os.path.join(tmp_path, "input.xml")

    system = openmm.System()
    system.addForce(openmm.NonbondedForce())

    with open(system_path, "w") as file:
        file.write(openmm.XmlSerializer.serialize(system))

    protocol = AddDummyAtoms("release_add_dummy_atoms")
    protocol.substance = dummy_complex
    protocol.input_coordinate_path = get_data_filename(
        os.path.join("test", "molecules", "methanol_methane.pdb")
    )
    protocol.input_system_path = system_path
    protocol.offset = 6.0 * unit.angstrom
    protocol.execute(str(tmp_path))

    # Validate that dummy atoms have been added to the configuration file
    # and the structure has been correctly shifted.
    trajectory = mdtraj.load_pdb(protocol.output_coordinate_path)
    assert trajectory.topology.n_atoms == 14

    assert numpy.allclose(trajectory.xyz[0][11:12, :2], 2.5)
    assert numpy.isclose(trajectory.xyz[0][11, 2], 0.62)
    assert numpy.isclose(trajectory.xyz[0][12, 2], 0.32)
    assert numpy.isclose(trajectory.xyz[0][13, 0], 2.5)
    assert numpy.isclose(trajectory.xyz[0][13, 1], 2.72)
    assert numpy.isclose(trajectory.xyz[0][13, 2], 0.1)

    # Validate the atom / residue names.
    all_atoms = [*trajectory.topology.atoms]
    dummy_atoms = all_atoms[11:14]

    assert all(atom.name == "DUM" for atom in dummy_atoms)
    assert all(dummy_atoms[i].residue.name == f"DM{i + 1}" for i in range(3))

    # Validate that the dummy atoms got added to the system
    with open(protocol.output_system_path) as file:
        system: openmm.System = openmm.XmlSerializer.deserialize(file.read())

    assert system.getNumParticles() == 3
    assert all(
        numpy.isclose(system.getParticleMass(i).value_in_unit(simtk_unit.dalton), 207.0)
        for i in range(3)
    )

    assert system.getNumForces() == 1
    assert system.getForce(0).getNumParticles() == 3


def validate_generated_restraints(restraints_path, expected_restraint_types, phase):

    restraints_dictionary = ApplyRestraints.load_restraints(restraints_path)
    restraints_dictionary = {
        restraint_type: restraints
        for restraint_type, restraints in restraints_dictionary.items()
        if len(restraints) > 0
    }

    assert {*restraints_dictionary} == expected_restraint_types

    restraints = [
        restraint
        for restraints in restraints_dictionary.values()
        for restraint in restraints
    ]

    assert all(
        "release" in restraint.phase and restraint.phase[phase] is not None
        for restraint in restraints
    )


def validate_system_file(system_path, expected_force_groups):

    from simtk import openmm

    assert os.path.isfile(system_path)

    with open(system_path) as file:
        system: openmm.System = openmm.XmlSerializer.deserialize(file.read())

    assert system.getNumForces() == len(expected_force_groups) + 3

    for force in system.getForces():
        assert force.getForceGroup() in {*expected_force_groups, 15}


def test_generate_attach_restraints(tmp_path, complex_file_path, restraints_schema):

    protocol = GenerateAttachRestraints("")
    protocol.complex_coordinate_path = complex_file_path
    protocol.attach_lambdas = [0.0, 0.5, 1.0]
    protocol.restraint_schemas = restraints_schema

    protocol.execute(str(tmp_path))

    assert os.path.isfile(protocol.restraints_path)

    validate_generated_restraints(
        protocol.restraints_path,
        {"static", "conformational", "guest", "wall", "symmetry"},
        "attach",
    )


def test_generate_pull_restraints(tmp_path, complex_file_path, restraints_schema):

    protocol = GeneratePullRestraints("")
    protocol.complex_coordinate_path = complex_file_path
    protocol.attach_lambdas = [0.0, 1.0]
    protocol.n_pull_windows = 2
    protocol.restraint_schemas = restraints_schema

    protocol.execute(str(tmp_path))

    assert os.path.isfile(protocol.restraints_path)

    validate_generated_restraints(
        protocol.restraints_path, {"static", "conformational", "guest"}, "pull"
    )


def test_generate_release_restraints(tmp_path, complex_file_path, restraints_schema):

    protocol = GenerateReleaseRestraints("")
    protocol.host_coordinate_path = complex_file_path
    protocol.release_lambdas = [1.0, 0.0]
    protocol.restraint_schemas = restraints_schema

    protocol.execute(str(tmp_path))

    assert os.path.isfile(protocol.restraints_path)

    validate_generated_restraints(
        protocol.restraints_path, {"static", "conformational"}, "release"
    )


def test_apply_attach_restraints(tmp_path, complex_file_path, attach_restraints_path):

    from simtk import openmm

    with open(os.path.join(tmp_path, "system.xml"), "w") as file:
        file.write(openmm.XmlSerializer.serialize(openmm.System()))

    protocol = ApplyRestraints("")
    protocol.restraints_path = attach_restraints_path
    protocol.input_coordinate_path = complex_file_path
    protocol.input_system_path = os.path.join(tmp_path, "system.xml")
    protocol.phase = "attach"
    protocol.window_index = 0
    protocol.execute(str(tmp_path))

    validate_system_file(protocol.output_system_path, {10, 11, 12, 13, 14})


def test_apply_pull_restraints(tmp_path, complex_file_path, pull_restraints_path):

    from simtk import openmm

    with open(os.path.join(tmp_path, "system.xml"), "w") as file:
        file.write(openmm.XmlSerializer.serialize(openmm.System()))

    protocol = ApplyRestraints("")
    protocol.restraints_path = pull_restraints_path
    protocol.input_coordinate_path = complex_file_path
    protocol.input_system_path = os.path.join(tmp_path, "system.xml")
    protocol.phase = "pull"
    protocol.window_index = 0
    protocol.execute(str(tmp_path))

    validate_system_file(protocol.output_system_path, {10, 11, 12})


def test_apply_release_restraints(tmp_path, complex_file_path, release_restraints_path):

    from simtk import openmm

    with open(os.path.join(tmp_path, "system.xml"), "w") as file:
        file.write(openmm.XmlSerializer.serialize(openmm.System()))

    protocol = ApplyRestraints("")
    protocol.restraints_path = release_restraints_path
    protocol.input_coordinate_path = complex_file_path
    protocol.input_system_path = os.path.join(tmp_path, "system.xml")
    protocol.phase = "release"
    protocol.window_index = 0
    protocol.execute(str(tmp_path))

    validate_system_file(protocol.output_system_path, {10, 11})
