"""This file is only temporary will the effect of different default settings is
explored.
"""
import json
import logging
import os
import shutil
import time

import numpy as np

from propertyestimator import unit
from propertyestimator.protocols import analysis, simulation
from propertyestimator.thermodynamics import Ensemble, ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.serialization import TypedJSONDecoder
from propertyestimator.utils.statistics import ObservableType
from propertyestimator.utils.utils import temporarily_change_directory


def _run_regular_simulation(temperature, total_number_of_iterations, steps_per_iteration, timestep,
                            number_of_molecules, coordinate_path, system_path, compute_resources):
    """Perform the simulation using the
    RunOpenMMSimulation protocol.
    """

    if os.path.isdir('run_regular'):
        shutil.rmtree('run_regular')

    os.makedirs('run_regular', exist_ok=True)

    protocol = simulation.RunOpenMMSimulation('protocol')
    protocol.total_number_of_iterations = total_number_of_iterations
    protocol.steps_per_iteration = steps_per_iteration
    protocol.output_frequency = steps_per_iteration
    protocol.input_coordinate_file = coordinate_path
    protocol.thermodynamic_state = ThermodynamicState(temperature)
    protocol.system_path = system_path
    protocol.timestep = timestep
    protocol.ensemble = Ensemble.NVT
    protocol.enable_pbc = False
    protocol.allow_gpu_platforms = number_of_molecules != 1
    protocol.high_precision = number_of_molecules == 1

    result = protocol.execute('run_regular', compute_resources)
    assert not isinstance(result, PropertyEstimatorException)

    extract_energy = analysis.ExtractAverageStatistic('extract')
    extract_energy.statistics_type = ObservableType.PotentialEnergy
    extract_energy.statistics_path = protocol.statistics_file_path

    result = extract_energy.execute('run_regular', compute_resources)
    assert not isinstance(result, PropertyEstimatorException)

    return extract_energy.value


def _run_parallel_tempering(temperatures, total_number_of_iterations, steps_per_iteration, timestep, integrator,
                            number_of_molecules, coordinate_path, system_path, compute_resources):
    """Perform the parallel tempering simulation using the
    OpenMMParallelTempering protocol.
    """

    if os.path.isdir('run_parallel'):
        shutil.rmtree('run_parallel')

    os.makedirs('run_parallel', exist_ok=True)

    protocol = simulation.OpenMMParallelTempering('protocol')
    protocol.total_number_of_iterations = total_number_of_iterations
    protocol.steps_per_iteration = steps_per_iteration
    protocol.input_coordinate_file = coordinate_path
    protocol.system_path = system_path
    protocol.timestep = timestep
    protocol.integrator = integrator
    protocol.ensemble = Ensemble.NVT
    protocol.enable_pbc = False
    protocol.output_frequency = 2
    protocol.allow_gpu_platforms = number_of_molecules != 1
    protocol.high_precision = number_of_molecules == 1

    protocol.thermodynamic_state = ThermodynamicState(temperatures[0])
    protocol.maximum_temperature = temperatures[-1]
    protocol.replica_temperatures = temperatures[1:-1]

    result = protocol.execute('run_parallel', compute_resources)
    assert not isinstance(result, PropertyEstimatorException)

    extract_energy = analysis.ExtractAverageStatistic('extract')
    extract_energy.statistics_type = ObservableType.PotentialEnergy
    extract_energy.statistics_path = protocol.statistics_file_path

    result = extract_energy.execute('run_parallel', compute_resources)
    assert not isinstance(result, PropertyEstimatorException)

    return extract_energy.value


def _mean_std(list_of_quantity):

    array = np.zeros(len(list_of_quantity))

    base_unit = list_of_quantity[0].units

    for i, quantity in enumerate(list_of_quantity):
        array[i] = quantity.to(base_unit).magnitude

    mean = array.mean() * base_unit
    std = array.std() * base_unit

    return mean, std


def test_run(input_json, available_resources):

    """A convenience function for running the full parallel tempering workflow
    with a given set of parameters.
    """

    input_dictionary = json.loads(input_json, cls=TypedJSONDecoder)

    root_directory = input_dictionary['root_directory']
    original_coordinate_path = input_dictionary['coordinate_path']
    original_system_path = input_dictionary['system_path']
    replica_temperatures = input_dictionary['replica_temperatures']
    number_of_molecules = input_dictionary['number_of_molecules']
    total_number_of_iterations = input_dictionary['total_number_of_iterations']
    steps_per_iteration = input_dictionary['steps_per_iteration']
    timestep = input_dictionary['timestep']
    integrator = input_dictionary['integrator']
    replicates = input_dictionary['replicates']

    simulation_means = []
    parallel_means = []

    simulation_stds = []
    parallel_stds = []

    for replicate_index in range(replicates):

        directory = os.path.join(root_directory, f'{replicate_index}')
        os.makedirs(directory, exist_ok=True)

        coordinate_path = os.path.join(directory, 'input.pdb')
        system_path = os.path.join(directory, 'system.xml')

        shutil.copyfile(original_coordinate_path, coordinate_path)
        shutil.copyfile(original_system_path, system_path)

        coordinate_path = 'input.pdb'
        system_path = 'system.xml'

        with temporarily_change_directory(directory):

            # Run the parallel tempering
            parallel_tempering_start_time = time.perf_counter()

            parallel_average_energy = _run_parallel_tempering(replica_temperatures,
                                                              total_number_of_iterations,
                                                              steps_per_iteration,
                                                              timestep,
                                                              integrator,
                                                              number_of_molecules,
                                                              coordinate_path,
                                                              system_path,
                                                              available_resources)

            parallel_tempering_end_time = time.perf_counter()

            parallel_means.append(parallel_average_energy.value)
            parallel_stds.append(parallel_average_energy.uncertainty)

            # Run a regular simulation for comparison
            simulation_start_time = time.perf_counter()

            average_energy = _run_regular_simulation(replica_temperatures[0],
                                                     total_number_of_iterations,
                                                     steps_per_iteration,
                                                     timestep,
                                                     number_of_molecules,
                                                     coordinate_path,
                                                     system_path,
                                                     available_resources)

            simulation_end_time = time.perf_counter()

            simulation_means.append(average_energy.value)
            simulation_stds.append(average_energy.uncertainty)

            logging.info(f'Regular_Time {(simulation_end_time - simulation_start_time) * 1000} '
                         f'Parallel_Time {(parallel_tempering_end_time - parallel_tempering_start_time) * 1000}')

    return (*_mean_std(parallel_means),
            *_mean_std(parallel_stds),
            *_mean_std(simulation_means),
            *_mean_std(simulation_stds))
