"""
Units tests for openff.evaluator.protocols.yank
"""
import os
import tempfile

import mdtraj
import numpy as np
import pytest
from openff.units import unit

from openff.evaluator.backends import ComputeResources
from openff.evaluator.forcefield import ParameterGradientKey
from openff.evaluator.protocols.coordinates import BuildCoordinatesPackmol
from openff.evaluator.protocols.forcefield import BuildSmirnoffSystem
from openff.evaluator.protocols.yank import (
    LigandReceptorYankProtocol,
    SolvationYankProtocol,
)
from openff.evaluator.substances import Component, ExactAmount, MoleFraction, Substance
from openff.evaluator.tests.utils import build_tip3p_smirnoff_force_field
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.timeseries import TimeSeriesStatistics
from openff.evaluator.utils.utils import get_data_filename, temporarily_change_directory


def _setup_dummy_system(directory, substance, number_of_molecules, force_field_path):
    os.makedirs(directory, exist_ok=True)

    build_coordinates = BuildCoordinatesPackmol("coordinates")
    build_coordinates.substance = substance
    build_coordinates.max_molecules = number_of_molecules
    build_coordinates.execute(str(directory))

    assign_parameters = BuildSmirnoffSystem("assign_parameters")
    assign_parameters.force_field_path = force_field_path
    assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
    assign_parameters.substance = substance
    assign_parameters.execute(str(directory))

    return (
        build_coordinates.coordinate_file_path,
        assign_parameters.parameterized_system,
    )


def test_ligand_receptor_yank_protocol():
    full_substance = Substance()

    full_substance.add_component(
        Component(smiles="c1ccccc1", role=Component.Role.Receptor),
        ExactAmount(1),
    )
    full_substance.add_component(
        Component(smiles="C", role=Component.Role.Ligand),
        ExactAmount(1),
    )
    full_substance.add_component(
        Component(smiles="O", role=Component.Role.Solvent),
        MoleFraction(1.0),
    )

    solute_substance = Substance()
    solute_substance.add_component(
        Component(smiles="C", role=Component.Role.Ligand),
        ExactAmount(1),
    )
    solute_substance.add_component(
        Component(smiles="O", role=Component.Role.Solvent),
        MoleFraction(1.0),
    )

    thermodynamic_state = ThermodynamicState(
        temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
    )

    with tempfile.TemporaryDirectory() as directory:
        with temporarily_change_directory(directory):
            force_field_path = "ff.json"

            with open(force_field_path, "w") as file:
                file.write(build_tip3p_smirnoff_force_field().json())

            complex_coordinate_path, complex_system = _setup_dummy_system(
                "full", full_substance, 3, force_field_path
            )

            ligand_coordinate_path, ligand_system = _setup_dummy_system(
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
            run_yank.solvated_ligand_system = ligand_system
            run_yank.solvated_complex_coordinates = complex_coordinate_path
            run_yank.solvated_complex_system = complex_system

            run_yank.force_field_path = force_field_path
            run_yank.execute("", ComputeResources())


@pytest.mark.parametrize("solvent_smiles", ["O", "C(Cl)Cl"])
def test_solvation_yank_protocol(solvent_smiles):
    full_substance = Substance()

    full_substance.add_component(
        Component(smiles="CO", role=Component.Role.Solute),
        ExactAmount(1),
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
        Component(smiles="CO", role=Component.Role.Solute),
        ExactAmount(1),
    )

    thermodynamic_state = ThermodynamicState(
        temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
    )

    with tempfile.TemporaryDirectory() as directory:
        with temporarily_change_directory(directory):
            force_field_path = "ff.json"

            with open(force_field_path, "w") as file:
                file.write(build_tip3p_smirnoff_force_field().json())

            solvated_coordinate_path, solvated_system = _setup_dummy_system(
                "full", full_substance, 2, force_field_path
            )

            vacuum_coordinate_path, vacuum_system = _setup_dummy_system(
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
            run_yank.solution_1_coordinates = solvated_coordinate_path
            run_yank.solution_1_system = solvated_system
            run_yank.solution_2_coordinates = vacuum_coordinate_path
            run_yank.solution_2_system = vacuum_system

            run_yank.electrostatic_lambdas_1 = [1.00]
            run_yank.steric_lambdas_1 = [1.00]
            run_yank.electrostatic_lambdas_2 = [1.00]
            run_yank.steric_lambdas_2 = [1.00]
            run_yank.execute("", ComputeResources())


def test_compute_state_energy_gradients(tmpdir):
    build_tip3p_smirnoff_force_field().json(os.path.join(tmpdir, "ff.json"))

    _, parameterized_system = _setup_dummy_system(
        tmpdir, Substance.from_components("O"), 10, os.path.join(tmpdir, "ff.json")
    )

    protocol = SolvationYankProtocol("")
    protocol.thermodynamic_state = ThermodynamicState(
        298.15 * unit.kelvin, 1.0 * unit.atmosphere
    )
    protocol.gradient_parameters = [
        ParameterGradientKey("vdW", "[#1]-[#8X2H2+0:1]-[#1]", "epsilon")
    ]

    gradients = protocol._compute_state_energy_gradients(
        mdtraj.load_dcd(
            get_data_filename("test/trajectories/water.dcd"),
            get_data_filename("test/trajectories/water.pdb"),
        ),
        parameterized_system.topology,
        parameterized_system.force_field.to_force_field(),
        True,
        ComputeResources(),
    )

    assert len(gradients) == 1
    assert not np.isclose(gradients[0].value, 0.0 * unit.dimensionless)


def test_analyze_phase(monkeypatch, tmpdir):
    from openmm import unit as openmm_unit

    # Generate the required inputs
    build_tip3p_smirnoff_force_field().json(os.path.join(tmpdir, "ff.json"))

    coordinate_path, parameterized_system = _setup_dummy_system(
        tmpdir, Substance.from_components("O"), 10, os.path.join(tmpdir, "ff.json")
    )
    solvent_trajectory = mdtraj.load_dcd(
        get_data_filename("test/trajectories/water.dcd"),
        get_data_filename("test/trajectories/water.pdb"),
    )

    # Mock the internally called methods.
    monkeypatch.setattr(
        SolvationYankProtocol,
        "_time_series_statistics",
        lambda *_: TimeSeriesStatistics(
            len(solvent_trajectory), len(solvent_trajectory), 1.0, 0
        ),
    )
    monkeypatch.setattr(
        SolvationYankProtocol, "_extract_trajectory", lambda *_: solvent_trajectory
    )
    monkeypatch.setattr(
        SolvationYankProtocol,
        "_extract_solvent_trajectory",
        lambda *_: solvent_trajectory,
    )
    monkeypatch.setattr(
        SolvationYankProtocol, "_compute_state_energy_gradients", lambda *_: []
    )

    # Build up the protocol.
    protocol = SolvationYankProtocol("")
    protocol.thermodynamic_state = ThermodynamicState(
        298.15 * unit.kelvin, 1.0 * unit.atmosphere
    )
    protocol.gradient_parameters = [
        ParameterGradientKey("vdW", "[#1]-[#8X2H2+0:1]-[#1]", "epsilon")
    ]
    protocol.solvent_1 = Substance.from_components("O")
    protocol._analysed_output = {
        "general": {"solvent1": {"nstates": 1}},
        "free_energy": {
            "solvent1": {
                "kT": 1.0 / openmm_unit.kilojoules_per_mole,
                "free_energy_diff": 0.0,
                "free_energy_diff_unit": 0.0 * openmm_unit.kilojoules_per_mole,
                "free_energy_diff_error": 0.0,
                "free_energy_diff_error_unit": 0.0 * openmm_unit.kilojoules_per_mole,
            }
        },
    }

    (
        free_energy,
        solution_trajectory,
        solvent_trajectory,
        solution_gradients,
        solvent_gradients,
    ) = protocol._analyze_phase(
        "", parameterized_system, "solvent1", ComputeResources()
    )
