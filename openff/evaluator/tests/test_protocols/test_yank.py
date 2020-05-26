"""
Units tests for openff.evaluator.protocols.yank
"""
import os
import tempfile

import pytest

from openff.evaluator import unit
from openff.evaluator.backends import ComputeResources
from openff.evaluator.protocols.coordinates import BuildCoordinatesPackmol
from openff.evaluator.protocols.forcefield import BuildSmirnoffSystem
from openff.evaluator.protocols.yank import (
    LigandReceptorYankProtocol,
    SolvationYankProtocol,
)
from openff.evaluator.substances import Component, ExactAmount, MoleFraction, Substance
from openff.evaluator.tests.utils import build_tip3p_smirnoff_force_field
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.utils import temporarily_change_directory


def _setup_dummy_system(directory, substance, number_of_molecules, force_field_path):

    os.makedirs(directory, exist_ok=True)

    build_coordinates = BuildCoordinatesPackmol("coordinates")
    build_coordinates.substance = substance
    build_coordinates.max_molecules = number_of_molecules
    build_coordinates.execute(directory, None)

    assign_parameters = BuildSmirnoffSystem("assign_parameters")
    assign_parameters.force_field_path = force_field_path
    assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
    assign_parameters.substance = substance
    assign_parameters.execute(directory, None)

    return build_coordinates.coordinate_file_path, assign_parameters.system_path


@pytest.mark.xfail(reason="openmoltools is incompatible with Ambertools 20")
def test_ligand_receptor_yank_protocol():

    full_substance = Substance()

    full_substance.add_component(
        Component(smiles="c1ccccc1", role=Component.Role.Receptor), ExactAmount(1),
    )
    full_substance.add_component(
        Component(smiles="C", role=Component.Role.Ligand), ExactAmount(1),
    )
    full_substance.add_component(
        Component(smiles="O", role=Component.Role.Solvent), MoleFraction(1.0),
    )

    solute_substance = Substance()
    solute_substance.add_component(
        Component(smiles="C", role=Component.Role.Ligand), ExactAmount(1),
    )
    solute_substance.add_component(
        Component(smiles="O", role=Component.Role.Solvent), MoleFraction(1.0),
    )

    thermodynamic_state = ThermodynamicState(
        temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
    )

    with tempfile.TemporaryDirectory() as directory:

        with temporarily_change_directory(directory):

            force_field_path = "ff.json"

            with open(force_field_path, "w") as file:
                file.write(build_tip3p_smirnoff_force_field().json())

            complex_coordinate_path, complex_system_path = _setup_dummy_system(
                "full", full_substance, 3, force_field_path
            )

            ligand_coordinate_path, ligand_system_path = _setup_dummy_system(
                "ligand", solute_substance, 2, force_field_path
            )

            run_yank = LigandReceptorYankProtocol("yank")
            run_yank.substance = full_substance
            run_yank.thermodynamic_state = thermodynamic_state
            run_yank.number_of_iterations = 1
            run_yank.steps_per_iteration = 1
            run_yank.checkpoint_interval = 1
            run_yank.verbose = True
            run_yank.setup_only = True

            run_yank.ligand_residue_name = "TMP"
            run_yank.receptor_residue_name = "TMP"
            run_yank.solvated_ligand_coordinates = ligand_coordinate_path
            run_yank.solvated_ligand_system = ligand_system_path
            run_yank.solvated_complex_coordinates = complex_coordinate_path
            run_yank.solvated_complex_system = complex_system_path

            run_yank.force_field_path = force_field_path
            run_yank.execute("", ComputeResources())


@pytest.mark.parametrize("solvent_smiles", ["O", "C(Cl)Cl"])
@pytest.mark.xfail(reason="openmoltools is incompatible with Ambertools 20")
def test_solvation_yank_protocol(solvent_smiles):

    full_substance = Substance()

    full_substance.add_component(
        Component(smiles="CO", role=Component.Role.Solute), ExactAmount(1),
    )
    full_substance.add_component(
        Component(smiles=solvent_smiles, role=Component.Role.Solvent),
        MoleFraction(1.0),
    )

    solvent_substance = Substance()
    solvent_substance.add_component(
        Component(smiles=solvent_smiles, role=Component.Role.Solvent),
        MoleFraction(1.0),
    )

    solute_substance = Substance()
    solute_substance.add_component(
        Component(smiles="CO", role=Component.Role.Solute), ExactAmount(1),
    )

    thermodynamic_state = ThermodynamicState(
        temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
    )

    with tempfile.TemporaryDirectory() as directory:

        with temporarily_change_directory(directory):

            force_field_path = "ff.json"

            with open(force_field_path, "w") as file:
                file.write(build_tip3p_smirnoff_force_field().json())

            solvated_coordinate_path, solvated_system_path = _setup_dummy_system(
                "full", full_substance, 2, force_field_path
            )

            vacuum_coordinate_path, vacuum_system_path = _setup_dummy_system(
                "vacuum", solute_substance, 1, force_field_path
            )

            run_yank = SolvationYankProtocol("yank")
            run_yank.solute = solute_substance
            run_yank.solvent_1 = solvent_substance
            run_yank.solvent_2 = Substance()
            run_yank.thermodynamic_state = thermodynamic_state
            run_yank.number_of_iterations = 1
            run_yank.steps_per_iteration = 1
            run_yank.checkpoint_interval = 1
            run_yank.verbose = True
            run_yank.setup_only = True
            run_yank.solvent_1_coordinates = solvated_coordinate_path
            run_yank.solvent_1_system = solvated_system_path
            run_yank.solvent_2_coordinates = vacuum_coordinate_path
            run_yank.solvent_2_system = vacuum_system_path

            run_yank.electrostatic_lambdas_1 = [1.00]
            run_yank.steric_lambdas_1 = [1.00]
            run_yank.electrostatic_lambdas_2 = [1.00]
            run_yank.steric_lambdas_2 = [1.00]
            run_yank.execute("", ComputeResources())
