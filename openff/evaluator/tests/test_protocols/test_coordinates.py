"""
Units tests for openff-evaluator.protocols.coordinates
"""
import tempfile

import pytest
from openmm.app import PDBFile

from openff.evaluator.backends import ComputeResources
from openff.evaluator.protocols.coordinates import (
    BuildCoordinatesPackmol,
    BuildDockedCoordinates,
    SolvateExistingStructure,
)
from openff.evaluator.substances import Component, ExactAmount, MoleFraction, Substance
from openff.evaluator.utils import get_data_filename, has_openeye


def _build_input_output_substances():
    """Builds sets if input and expected substances for the
    `test_build_coordinate_composition` test.

    Returns
    -------
    list of tuple of Substance and Substance
        A list of input and expected substances.
    """

    # Start with some easy cases
    substances = [
        (Substance.from_components("O"), Substance.from_components("O")),
        (Substance.from_components("O", "C"), Substance.from_components("O", "C")),
        (
            Substance.from_components("O", "C", "CO"),
            Substance.from_components("O", "C", "CO"),
        ),
    ]

    # Handle some cases where rounding will need to occur.
    input_substance = Substance()
    input_substance.add_component(Component("O"), MoleFraction(0.41))
    input_substance.add_component(Component("C"), MoleFraction(0.59))

    expected_substance = Substance()
    expected_substance.add_component(Component("O"), MoleFraction(0.4))
    expected_substance.add_component(Component("C"), MoleFraction(0.6))

    substances.append((input_substance, expected_substance))

    input_substance = Substance()
    input_substance.add_component(Component("O"), MoleFraction(0.59))
    input_substance.add_component(Component("C"), MoleFraction(0.41))

    expected_substance = Substance()
    expected_substance.add_component(Component("O"), MoleFraction(0.6))
    expected_substance.add_component(Component("C"), MoleFraction(0.4))

    substances.append((input_substance, expected_substance))

    return substances


@pytest.mark.parametrize("input_substance, expected", _build_input_output_substances())
def test_build_coordinates_packmol(input_substance, expected):
    """Tests that the build coordinate protocols correctly report
    the composition of the built system."""

    build_coordinates = BuildCoordinatesPackmol("build_coordinates")
    build_coordinates.max_molecules = 10
    build_coordinates.substance = input_substance

    with tempfile.TemporaryDirectory() as directory:
        build_coordinates.execute(directory)

    assert build_coordinates.output_substance == expected

    for component in input_substance:
        assert component.identifier in build_coordinates.assigned_residue_names

        if component.smiles == "O":
            assigned_name = build_coordinates.assigned_residue_names[
                component.identifier
            ]
            assert assigned_name[:3] == "HOH"


@pytest.mark.parametrize("count_exact_amount", [False, True])
def test_build_coordinates_packmol_exact(count_exact_amount):
    """Tests that the build coordinate protocol behaves correctly for substances
    with exact amounts."""

    import mdtraj

    substance = Substance()
    substance.add_component(Component("O"), MoleFraction(1.0))
    substance.add_component(Component("C"), ExactAmount(1))

    max_molecule = 11 if count_exact_amount else 10

    build_coordinates = BuildCoordinatesPackmol("build_coordinates")
    build_coordinates.max_molecules = max_molecule
    build_coordinates.count_exact_amount = count_exact_amount
    build_coordinates.substance = substance

    with tempfile.TemporaryDirectory() as directory:
        build_coordinates.execute(directory)
        built_system = mdtraj.load_pdb(build_coordinates.coordinate_file_path)

    assert built_system.n_residues == 11


def test_solvate_existing_structure_protocol():
    """Tests solvating a single methanol molecule in water."""

    import mdtraj

    methanol_component = Component("CO")

    methanol_substance = Substance()
    methanol_substance.add_component(methanol_component, ExactAmount(1))

    water_substance = Substance()
    water_substance.add_component(Component("O"), MoleFraction(1.0))

    with tempfile.TemporaryDirectory() as temporary_directory:
        build_methanol_coordinates = BuildCoordinatesPackmol("build_methanol")
        build_methanol_coordinates.max_molecules = 1
        build_methanol_coordinates.substance = methanol_substance
        build_methanol_coordinates.execute(temporary_directory, ComputeResources())

        methanol_residue_name = build_methanol_coordinates.assigned_residue_names[
            methanol_component.identifier
        ]

        solvate_coordinates = SolvateExistingStructure("solvate_methanol")
        solvate_coordinates.max_molecules = 9
        solvate_coordinates.substance = water_substance
        solvate_coordinates.solute_coordinate_file = (
            build_methanol_coordinates.coordinate_file_path
        )
        solvate_coordinates.execute(temporary_directory, ComputeResources())
        solvated_system = mdtraj.load_pdb(solvate_coordinates.coordinate_file_path)

        assert solvated_system.n_residues == 10
        assert solvated_system.top.residue(0).name == methanol_residue_name


def test_build_docked_coordinates_protocol():
    """Tests docking a methanol molecule into alpha-Cyclodextrin."""

    if not has_openeye():
        pytest.skip("The `BuildDockedCoordinates` protocol requires OpenEye.")

    ligand_substance = Substance()
    ligand_substance.add_component(
        Component("CO", role=Component.Role.Ligand),
        ExactAmount(1),
    )

    # TODO: This test could likely be made substantially faster
    #       by storing the binary prepared receptor. Would this
    #       be in breach of any oe license terms?
    with tempfile.TemporaryDirectory() as temporary_directory:
        build_docked_coordinates = BuildDockedCoordinates("build_methanol")
        build_docked_coordinates.ligand_substance = ligand_substance
        build_docked_coordinates.number_of_ligand_conformers = 5
        build_docked_coordinates.receptor_coordinate_file = get_data_filename(
            "test/molecules/acd.mol2"
        )
        build_docked_coordinates.execute(temporary_directory, ComputeResources())

        docked_pdb = PDBFile(build_docked_coordinates.docked_complex_coordinate_path)
        assert docked_pdb.topology.getNumResidues() == 2


if __name__ == "__main__":
    test_build_docked_coordinates_protocol()
