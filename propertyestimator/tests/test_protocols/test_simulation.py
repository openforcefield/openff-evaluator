"""
Units tests for propertyestimator.protocols.simulation
"""
import json
import os
import tempfile
from os import path

from propertyestimator import unit
from propertyestimator.backends import ComputeResources
from propertyestimator.protocols.coordinates import BuildCoordinatesPackmol
from propertyestimator.protocols.forcefield import BuildSmirnoffSystem
from propertyestimator.protocols.simulation import RunEnergyMinimisation, RunOpenMMSimulation, OpenMMParallelTempering
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder
from propertyestimator.utils.statistics import StatisticsArray


def _setup_dummy_system(directory):

    force_field_path = path.join(directory, 'ff.json')

    with open(force_field_path, 'w') as file:
        file.write(build_tip3p_smirnoff_force_field().json())

    substance = Substance.from_components('C')

    build_coordinates = BuildCoordinatesPackmol('build_coordinates')
    build_coordinates.max_molecules = 1
    build_coordinates.mass_density = 0.001 * unit.grams / unit.milliliters
    build_coordinates.substance = substance
    build_coordinates.execute(directory, None)

    assign_parameters = BuildSmirnoffSystem(f'assign_parameters')
    assign_parameters.force_field_path = force_field_path
    assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
    assign_parameters.substance = substance
    assign_parameters.execute(directory, None)

    return build_coordinates.coordinate_file_path, assign_parameters.system_path


def test_run_energy_minimisation():

    with tempfile.TemporaryDirectory() as directory:

        coordinate_path, system_path = _setup_dummy_system(directory)

        energy_minimisation = RunEnergyMinimisation('energy_minimisation')
        energy_minimisation.input_coordinate_file = coordinate_path
        energy_minimisation.system_path = system_path

        result = energy_minimisation.execute(directory, ComputeResources())

        assert not isinstance(result, PropertyEstimatorException)
        assert path.isfile(energy_minimisation.output_coordinate_file)


def test_run_openmm_simulation():

    thermodynamic_state = ThermodynamicState(298 * unit.kelvin,
                                             1.0 * unit.atmosphere)

    with tempfile.TemporaryDirectory() as directory:

        coordinate_path, system_path = _setup_dummy_system(directory)

        npt_equilibration = RunOpenMMSimulation('npt_equilibration')
        npt_equilibration.steps_per_iteration = 2
        npt_equilibration.output_frequency = 1
        npt_equilibration.thermodynamic_state = thermodynamic_state
        npt_equilibration.input_coordinate_file = coordinate_path
        npt_equilibration.system_path = system_path

        result = npt_equilibration.execute(directory, ComputeResources())

        assert not isinstance(result, PropertyEstimatorException)
        assert path.isfile(npt_equilibration.output_coordinate_file)
        assert path.isfile(npt_equilibration.trajectory_file_path)
        assert path.isfile(npt_equilibration.statistics_file_path)


def test_run_openmm_simulation_checkpoints():

    import mdtraj

    thermodynamic_state = ThermodynamicState(298 * unit.kelvin,
                                             1.0 * unit.atmosphere)

    with tempfile.TemporaryDirectory() as directory:

        coordinate_path, system_path = _setup_dummy_system(directory)

        # Check that executing twice doesn't run the simulation twice
        npt_equilibration = RunOpenMMSimulation('npt_equilibration')
        npt_equilibration.total_number_of_iterations = 1
        npt_equilibration.steps_per_iteration = 4
        npt_equilibration.output_frequency = 1
        npt_equilibration.thermodynamic_state = thermodynamic_state
        npt_equilibration.input_coordinate_file = coordinate_path
        npt_equilibration.system_path = system_path

        npt_equilibration.execute(directory, ComputeResources())
        assert os.path.isfile(npt_equilibration._checkpoint_path)
        npt_equilibration.execute(directory, ComputeResources())

        assert len(StatisticsArray.from_pandas_csv(npt_equilibration.statistics_file_path)) == 4
        assert len(mdtraj.load(npt_equilibration.trajectory_file_path, top=coordinate_path)) == 4

        # Make sure that the output files are correctly truncating if more frames
        # than expected are written
        with open(npt_equilibration._checkpoint_path, 'r') as file:
            checkpoint = json.load(file, cls=TypedJSONDecoder)

            # Fake having saved more frames than expected
            npt_equilibration.steps_per_iteration = 8
            checkpoint.steps_per_iteration = 8
            npt_equilibration.output_frequency = 2
            checkpoint.output_frequency = 2

        with open(npt_equilibration._checkpoint_path, 'w') as file:
            json.dump(checkpoint, file, cls=TypedJSONEncoder)

        npt_equilibration.execute(directory, ComputeResources())

        assert len(StatisticsArray.from_pandas_csv(npt_equilibration.statistics_file_path)) == 4
        assert len(mdtraj.load(npt_equilibration.trajectory_file_path, top=coordinate_path)) == 4


def test_openmm_parallel_tempering():

    thermodynamic_state = ThermodynamicState(298 * unit.kelvin)

    with tempfile.TemporaryDirectory() as directory:

        coordinate_path, system_path = _setup_dummy_system(directory)

        protocol = OpenMMParallelTempering('protocol')
        protocol.total_number_of_iterations = 1
        protocol.steps_per_iteration = 1
        protocol.thermodynamic_state = thermodynamic_state
        protocol.maximum_temperature = 500 * unit.kelvin
        protocol.number_of_replicas = 1
        protocol.input_coordinate_file = coordinate_path
        protocol.system_path = system_path
        protocol.high_precision = True
        protocol.enable_pbc = False
        protocol.allow_gpu_platforms = False

        result = protocol.execute(directory, ComputeResources())
        assert not isinstance(result, PropertyEstimatorException)
