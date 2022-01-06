"""
Units tests for openff.evaluator.protocols.openmm
"""
import json
import os
import tempfile
from os import path

import mdtraj
import numpy
import pytest
from openff.toolkit.topology import Molecule, Topology
from openff.toolkit.typing.engines.smirnoff import ForceField
from openff.units import unit

try:
    from openmm import unit as openmm_unit
except ImportError:
    from simtk.openmm import unit as openmm_unit

from smirnoff_plugins.handlers.nonbonded import DoubleExponential

from openff.evaluator.backends import ComputeResources
from openff.evaluator.forcefield import ParameterGradientKey
from openff.evaluator.protocols.coordinates import BuildCoordinatesPackmol
from openff.evaluator.protocols.forcefield import BuildSmirnoffSystem
from openff.evaluator.protocols.openmm import (
    OpenMMEnergyMinimisation,
    OpenMMEvaluateEnergies,
    OpenMMSimulation,
    _compute_gradients,
    _evaluate_energies,
)
from openff.evaluator.substances import Substance
from openff.evaluator.tests.utils import build_tip3p_smirnoff_force_field
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils import get_data_filename
from openff.evaluator.utils.observables import ObservableType
from openff.evaluator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder


def _setup_dummy_system(directory):
    """Generate a temporary parameterized system object."""

    force_field_path = path.join(directory, "ff.json")

    with open(force_field_path, "w") as file:
        file.write(build_tip3p_smirnoff_force_field().json())

    substance = Substance.from_components("O")

    build_coordinates = BuildCoordinatesPackmol("build_coordinates")
    build_coordinates.max_molecules = 10
    build_coordinates.mass_density = 0.05 * unit.grams / unit.milliliters
    build_coordinates.substance = substance
    build_coordinates.execute(directory)

    assign_parameters = BuildSmirnoffSystem("assign_parameters")
    assign_parameters.force_field_path = force_field_path
    assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
    assign_parameters.substance = substance
    assign_parameters.execute(directory)

    return (
        build_coordinates.coordinate_file_path,
        assign_parameters.parameterized_system,
    )


def test_run_energy_minimisation():

    with tempfile.TemporaryDirectory() as directory:

        coordinate_path, parameterized_system = _setup_dummy_system(directory)

        energy_minimisation = OpenMMEnergyMinimisation("energy_minimisation")
        energy_minimisation.input_coordinate_file = coordinate_path
        energy_minimisation.parameterized_system = parameterized_system
        energy_minimisation.execute(directory, ComputeResources())
        assert path.isfile(energy_minimisation.output_coordinate_file)


def test_run_openmm_simulation():

    thermodynamic_state = ThermodynamicState(298 * unit.kelvin, 1.0 * unit.atmosphere)

    with tempfile.TemporaryDirectory() as directory:

        coordinate_path, parameterized_system = _setup_dummy_system(directory)

        npt_equilibration = OpenMMSimulation("npt_equilibration")
        npt_equilibration.steps_per_iteration = 2
        npt_equilibration.output_frequency = 1
        npt_equilibration.thermodynamic_state = thermodynamic_state
        npt_equilibration.input_coordinate_file = coordinate_path
        npt_equilibration.parameterized_system = parameterized_system
        npt_equilibration.execute(directory, ComputeResources())

        assert path.isfile(npt_equilibration.output_coordinate_file)
        assert path.isfile(npt_equilibration.trajectory_file_path)
        assert len(npt_equilibration.observables) == 2


def test_run_openmm_simulation_checkpoints():

    import mdtraj

    thermodynamic_state = ThermodynamicState(298 * unit.kelvin, 1.0 * unit.atmosphere)

    with tempfile.TemporaryDirectory() as directory:

        coordinate_path, parameterized_system = _setup_dummy_system(directory)

        # Check that executing twice doesn't run the simulation twice
        npt_equilibration = OpenMMSimulation("npt_equilibration")
        npt_equilibration.total_number_of_iterations = 1
        npt_equilibration.steps_per_iteration = 4
        npt_equilibration.output_frequency = 1
        npt_equilibration.thermodynamic_state = thermodynamic_state
        npt_equilibration.input_coordinate_file = coordinate_path
        npt_equilibration.parameterized_system = parameterized_system

        npt_equilibration.execute(directory, ComputeResources())
        assert os.path.isfile(npt_equilibration._checkpoint_path)
        npt_equilibration.execute(directory, ComputeResources())

        assert len(npt_equilibration.observables) == 4
        assert (
            len(
                mdtraj.load(npt_equilibration.trajectory_file_path, top=coordinate_path)
            )
            == 4
        )

        # Make sure that the output files are correctly truncating if more frames
        # than expected are written
        with open(npt_equilibration._checkpoint_path, "r") as file:
            checkpoint = json.load(file, cls=TypedJSONDecoder)

            # Fake having saved more frames than expected
            npt_equilibration.steps_per_iteration = 8
            checkpoint.steps_per_iteration = 8
            npt_equilibration.output_frequency = 2
            checkpoint.output_frequency = 2

        with open(npt_equilibration._checkpoint_path, "w") as file:
            json.dump(checkpoint, file, cls=TypedJSONEncoder)

        npt_equilibration.execute(directory, ComputeResources())

        assert len(npt_equilibration.observables) == 4
        assert (
            len(
                mdtraj.load(npt_equilibration.trajectory_file_path, top=coordinate_path)
            )
            == 4
        )


def test_evaluate_energies_openmm():

    substance = Substance.from_components("O")
    thermodynamic_state = ThermodynamicState(298 * unit.kelvin, 1.0 * unit.atmosphere)

    with tempfile.TemporaryDirectory() as directory:

        coordinate_path, parameterized_system = _setup_dummy_system(directory)

        reduced_potentials = OpenMMEvaluateEnergies("")
        reduced_potentials.substance = substance
        reduced_potentials.thermodynamic_state = thermodynamic_state
        reduced_potentials.parameterized_system = parameterized_system
        reduced_potentials.trajectory_file_path = get_data_filename(
            "test/trajectories/water.dcd"
        )
        reduced_potentials.execute(directory, ComputeResources())

        assert ObservableType.ReducedPotential in reduced_potentials.output_observables
        assert ObservableType.PotentialEnergy in reduced_potentials.output_observables


@pytest.mark.xfail(
    reason="Broken until smirnoff_plugins is made compatible with openff.units"
)
def test_smirnoff_plugin_gradients():
    molecule = Molecule.from_smiles("C")
    molecule.generate_conformers(n_conformers=1)

    conformer = molecule.conformers[0].m_as(unit.nanometer)
    conformer = numpy.vstack([conformer, conformer + 0.5])

    topology = Topology.from_molecules([Molecule.from_smiles("C")] * 2)

    epsilon = 0.1094

    custom_handler = DoubleExponential(version="0.3")
    custom_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[#6X4:1]",
            "r_min": 1.908 * openmm_unit.angstrom,
            "epsilon": epsilon * openmm_unit.kilocalories_per_mole,
        }
    )
    custom_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[#1:1]-[#6X4]",
            "r_min": 1.487 * openmm_unit.angstrom,
            "epsilon": 0.0 * openmm_unit.kilocalories_per_mole,
        }
    )

    force_field = ForceField(load_plugins=True)
    force_field.register_parameter_handler(custom_handler)

    vdw_handler = force_field.get_parameter_handler("vdW")
    vdw_handler.add_parameter(
        parameter_kwargs={
            "smirks": "[*:1]",
            "epsilon": 0.0 * unit.kilocalories_per_mole,
            "sigma": 1.0 * unit.angstrom,
        }
    )

    trajectory = mdtraj.Trajectory(
        xyz=conformer.reshape((1, 10, 3)) * openmm_unit.nanometers,
        topology=topology.to_openmm(),
    )

    observables = _evaluate_energies(
        ThermodynamicState(
            temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
        ),
        force_field.create_openmm_system(topology),
        trajectory,
        ComputeResources(),
        enable_pbc=False,
    )
    _compute_gradients(
        [
            ParameterGradientKey(
                tag="DoubleExponential", smirks="[#6X4:1]", attribute="epsilon"
            )
        ],
        observables,
        force_field,
        ThermodynamicState(
            temperature=298.15 * unit.kelvin, pressure=1.0 * unit.atmosphere
        ),
        topology,
        trajectory,
        ComputeResources(),
        enable_pbc=False,
    )

    assert numpy.isclose(
        observables[ObservableType.PotentialEnergy].gradients[0].value,
        observables[ObservableType.PotentialEnergy].value
        / (epsilon * unit.kilocalorie / unit.mole),
    )
