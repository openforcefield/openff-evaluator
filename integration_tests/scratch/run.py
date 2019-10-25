import argparse
import logging
import os
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
from openforcefield.topology import Molecule, Topology
from scipy.optimize import newton
from scipy.stats import norm
from simtk import openmm, unit as simtk_unit
from simtk.openmm import app

from propertyestimator import unit
from propertyestimator.backends import ComputeResources
from propertyestimator.protocols import analysis, coordinates, forcefield, reweighting, simulation
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.thermodynamics import Ensemble, ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import disable_pbc
from propertyestimator.utils.statistics import StatisticsArray, ObservableType
from propertyestimator.utils.utils import temporarily_change_directory, setup_timestamp_logging

logger = logging.getLogger()

# OpenMM constant for Coulomb interactions (openmm/platforms/reference/
# include/SimTKOpenMMRealType.h) in OpenMM units
ONE_4PI_EPS0 = 138.935456


def _build_system(smiles, number_of_molecules, directory, coordinate_file_name, system_file_name):
    """Set up the coordinates / system object for a given smiles and
    containing a specified number of molecules.
    """

    import parmed as pmd

    force_field_path = os.path.join(directory, 'ff.json')

    with open(force_field_path, 'w') as file:
        file.write(build_tip3p_smirnoff_force_field().json())

    substance = Substance.from_components(smiles)

    build_coordinates = coordinates.BuildCoordinatesPackmol('build_coordinates')
    build_coordinates.max_molecules = 1
    build_coordinates.substance = substance
    build_coordinates.execute(directory, None)

    original_structure = pmd.load_file(os.path.join(directory, 'output.pdb'))
    new_structure = pmd.Structure()

    shifted_positions = []

    for index in range(number_of_molecules):

        for position in original_structure.positions:

            shifted_position = position + (0.01 * index,
                                           0.01 * index,
                                           0.01 * index) * simtk_unit.angstrom

            shifted_positions.append(shifted_position.value_in_unit(simtk_unit.angstrom))

        new_structure += original_structure

    with open(coordinate_file_name, 'w') as file:
        app.PDBFile.writeFile(new_structure.topology, shifted_positions * simtk_unit.angstrom, file)

    assign_parameters = forcefield.BuildSmirnoffSystem(f'assign_parameters')
    assign_parameters.force_field_path = force_field_path
    assign_parameters.coordinate_file_path = coordinate_file_name
    assign_parameters.substance = substance
    assign_parameters.execute(directory, None)

    # Force disable any pbc
    with open(assign_parameters.system_path) as file:
        system = openmm.XmlSerializer.deserialize(file.read())

    disable_pbc(system)

    with open(system_file_name, 'w') as file:
        file.write(openmm.XmlSerializer.serialize(system))


def _convert_nonbonded_to_custom(original_force):
    """Convert an openmm NonbondedForce into a CustomNonbondedForce
    """

    energy_expression = (f'4*epsilon*((sigma/r)^12-(sigma/r)^6)+'
                         f'ONE_4PI_EPS0*chargeprod/r; '
                          
                         # Steric mixing rules
                         f'sigma=sqrt(sigma1*sigma2); '
                         f'epsilon=sqrt(epsilon1*epsilon2); '
                          
                         # Electrostatic mixing rules
                         f'chargeprod=charge1*charge2; '
                         f'ONE_4PI_EPS0={ONE_4PI_EPS0:f};')

    custom_force = openmm.CustomNonbondedForce(energy_expression)

    custom_force.setNonbondedMethod(openmm.CustomNonbondedForce.NoCutoff)

    custom_force.addPerParticleParameter("charge")
    custom_force.addPerParticleParameter("sigma")
    custom_force.addPerParticleParameter("epsilon")

    custom_force.setUseSwitchingFunction(original_force.getUseSwitchingFunction())
    custom_force.setCutoffDistance(original_force.getCutoffDistance())
    custom_force.setSwitchingDistance(original_force.getSwitchingDistance())
    custom_force.setUseLongRangeCorrection(original_force.getUseDispersionCorrection())

    for index in range(original_force.getNumParticles()):

        charge, sigma, epsilon = original_force.getParticleParameters(index)

        custom_force.addParticle([charge, sigma, epsilon])
        original_force.setParticleParameters(index, 0.0, 0.0, 0.0)

        custom_charge, custom_sigma, custom_epsilon = custom_force.getParticleParameters(index)

        assert (np.isclose(charge.value_in_unit(simtk_unit.elementary_charge), custom_charge) and
                np.isclose(sigma.value_in_unit(simtk_unit.nanometer), custom_sigma) and
                np.isclose(epsilon.value_in_unit(simtk_unit.kilojoule_per_mole), custom_epsilon))

    for index in range(original_force.getNumExceptions()):

        atom_i, atom_j, *original_parameters = original_force.getExceptionParameters(index)
        custom_force.addExclusion(atom_i, atom_j)

    return custom_force


def _make_system_ideal(smiles, coordinate_file_path,
                       system_file_path, new_system_file_path=None):
    """Converts a system of interacting molecules into one of
    ideal ones.
    """

    substance = Substance.from_components(smiles)

    unique_molecules = [Molecule.from_smiles(component.smiles) for
                        component in substance.components]

    openmm_topology = app.PDBFile(coordinate_file_path).topology
    topology = Topology.from_openmm(openmm_topology, unique_molecules)

    with open(system_file_path) as file:
        system = openmm.XmlSerializer.deserialize(file.read())

    if topology.n_topology_molecules > 1:

        # Set up a custom ideal force in the case of multiple molecules.
        nonbonded_force_indices = [index for index in range(system.getNumForces()) if
                                   isinstance(system.getForce(index), openmm.NonbondedForce)]

        assert len(nonbonded_force_indices) == 1
        nonbonded_force_index = nonbonded_force_indices[0]

        # Convert the original force into a custom one on which we
        # can define interaction groups.
        custom_force = _convert_nonbonded_to_custom(system.getForce(nonbonded_force_index))

        # Setup the custom interaction groups.
        for molecule in topology.topology_molecules:

            atom_indices = [atom.topology_atom_index for atom in molecule.atoms]
            custom_force.addInteractionGroup(atom_indices, atom_indices)

        # system.removeForce(nonbonded_force_index)
        system.addForce(custom_force)

    if new_system_file_path is None:
        new_system_file_path = system_file_path

    with open(new_system_file_path, 'w') as file:
        file.write(openmm.XmlSerializer.serialize(system))


def _determine_temperatures(minimum_temperature, maximum_temperature, number_of_intermediates,
                            number_of_molecules, desired_probability, coordinate_path, system_path,
                            compute_resources, plot=False):
    """Determine which temperatures to use for the replica exchange by
    running a short set of simulations over the temperature range of interest,
    fitting the potential energy as a function of temperature, then iteratively
    selecting the temperatures to theoretically achieve a specific exchange
    probability.
    """

    assert 0 < desired_probability <= 1.0

    temperatures = minimum_temperature + np.linspace(0.0, 1.0, number_of_intermediates + 2) * (maximum_temperature -
                                                                                               minimum_temperature)

    mu = np.zeros(len(temperatures))
    # sigma = np.zeros(len(temperatures))

    # Determine the energy as a function of T
    for index, temperature in enumerate(temperatures):

        os.makedirs(f'setup_{temperature}', exist_ok=True)

        protocol = simulation.RunOpenMMSimulation('protocol')
        protocol.input_coordinate_file = coordinate_path
        protocol.system_path = system_path
        protocol.steps_per_iteration = 200000
        protocol.output_frequency = 500
        protocol.ensemble = Ensemble.NVT
        protocol.enable_pbc = False
        protocol.allow_gpu_platforms = number_of_molecules != 1
        protocol.high_precision = number_of_molecules == 1
        protocol.thermodynamic_state = ThermodynamicState(temperature)

        result = protocol.execute(f'setup_{temperature}', compute_resources)
        assert not isinstance(result, PropertyEstimatorException)

        statistics_array = StatisticsArray.from_pandas_csv(protocol.statistics_file_path)
        potentials = statistics_array[ObservableType.PotentialEnergy].to(unit.kilojoule / unit.mole).magnitude

        mu[index] = np.mean(potentials)
        sigma = np.std(potentials)

        if plot:
            x_values = np.linspace(mu[index] - 3 * sigma, mu[index] + 3 * sigma, 100)
            plt.plot(x_values, norm.pdf(x_values, mu[index], sigma))

    # Fit a polynomial
    coefficients = np.polyfit(temperatures.to(unit.kelvin).magnitude, mu, 2)

    if plot:

        plt.show()

        fitted_x_values = np.linspace(temperatures[0].to(unit.kelvin).magnitude,
                                      temperatures[-1].to(unit.kelvin).magnitude, 50)

        fitted_y_values = (coefficients[0] * fitted_x_values ** 2 +
                           coefficients[1] * fitted_x_values +
                           coefficients[2])

        plt.plot(temperatures.to(unit.kelvin).magnitude, mu)
        plt.plot(fitted_x_values, fitted_y_values)
        plt.show()

    current_temperature = temperatures[0]
    replica_temperatures = [current_temperature]

    while current_temperature < maximum_temperature:

        def evaluate_energy(x):

            return (coefficients[0] * x ** 2 +
                    coefficients[1] * x +
                    coefficients[2]) * unit.kilojoule / unit.mole

        def evaluate_energy_gradient(x):

            return (coefficients[0] * x * 2 +
                    coefficients[1]) * unit.kilojoule / unit.mole / unit.kelvin

        def evaluate_root(x):

            beta_1 = 1.0 / (current_temperature * unit.molar_gas_constant)
            beta_2 = 1.0 / (x * unit.kelvin * unit.molar_gas_constant)

            energy_1 = evaluate_energy(current_temperature.to(unit.kelvin).magnitude)
            energy_2 = evaluate_energy(x)

            exponent = ((beta_1 - beta_2) * (energy_1 - energy_2)).to(unit.dimensionless).magnitude

            return exponent - np.log(desired_probability)

        def evaluate_root_gradient(x):

            beta_1 = 1.0 / (current_temperature * unit.molar_gas_constant)
            beta_2 = 1.0 / (x * unit.kelvin * unit.molar_gas_constant)

            energy_1 = evaluate_energy(current_temperature.to(unit.kelvin).magnitude)
            energy_2 = evaluate_energy(x)

            energy_gradient_2 = evaluate_energy_gradient(x)

            return (beta_2 * energy_gradient_2 -
                    beta_1 * energy_gradient_2 +
                    beta_2 * (energy_1 - energy_2) / (x * unit.kelvin)).to(unit.Unit('1/kelvin')).magnitude

        new_temperature = newton(evaluate_root,
                                 current_temperature.to(unit.kelvin).magnitude * 1.5,
                                 evaluate_root_gradient) * unit.kelvin

        if new_temperature < maximum_temperature:
            replica_temperatures.append(new_temperature)

        current_temperature = new_temperature

    replica_temperatures += [maximum_temperature]

    for index, temperature in enumerate(replica_temperatures):

        directory = f'replica_{temperature.magnitude:.1f}'
        os.makedirs(directory, exist_ok=True)

        protocol = simulation.RunOpenMMSimulation('protocol')
        protocol.input_coordinate_file = coordinate_path
        protocol.system_path = system_path
        protocol.steps_per_iteration = 50000
        protocol.output_frequency = 500
        protocol.ensemble = Ensemble.NVT
        protocol.enable_pbc = False
        protocol.allow_gpu_platforms = number_of_molecules != 1
        protocol.high_precision = number_of_molecules == 1
        protocol.thermodynamic_state = ThermodynamicState(temperature)

        result = protocol.execute(directory, compute_resources)
        assert not isinstance(result, PropertyEstimatorException)

        statistics_array = StatisticsArray.from_pandas_csv(protocol.statistics_file_path)
        potentials = statistics_array[ObservableType.PotentialEnergy].to(unit.kilojoule / unit.mole).magnitude

        mu = np.mean(potentials)
        sigma = np.std(potentials)

        if plot:
            x_values = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
            plt.plot(x_values, norm.pdf(x_values, mu, sigma))

    if plot:
        plt.show()

    return replica_temperatures


def _run_regular_simulation(temperature, total_number_of_iterations, steps_per_iteration, number_of_molecules,
                            coordinate_path, system_path, compute_resources):
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


def _run_parallel_tempering(temperatures, total_number_of_iterations, steps_per_iteration, number_of_molecules,
                            coordinate_path, system_path, compute_resources):
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
    protocol.ensemble = Ensemble.NVT
    protocol.enable_pbc = False
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


def _run(root_directory, smiles, maximum_temperature, number_of_molecules, exchange_probability,
         total_number_of_iterations, steps_per_iteration, plot, available_resources):

    """A convenience function for running the full parallel tempering workflow
    with a given set of parameters.
    """

    logger.info(f'Starting')

    with temporarily_change_directory(root_directory):

        # Define the resources to use.
        coordinate_path = 'input.pdb'
        system_path = 'system.xml'

        # Create the initial system object and coordinates.
        os.makedirs('setup', exist_ok=True)

        _build_system(smiles=smiles,
                      number_of_molecules=number_of_molecules,
                      directory='setup',
                      coordinate_file_name=coordinate_path,
                      system_file_name=system_path)

        # Make the system an ideal gas.
        ideal_system_path = 'ideal_system.xml'

        _make_system_ideal(smiles=smiles,
                           coordinate_file_path=coordinate_path,
                           system_file_path=system_path,
                           new_system_file_path=ideal_system_path)

        parallel_tempering_start_time = time.perf_counter()

        # Determine the replica temperatures
        temperatures = _determine_temperatures(298.0 * unit.kelvin,
                                               maximum_temperature,
                                               1,
                                               number_of_molecules,
                                               exchange_probability,
                                               coordinate_path,
                                               ideal_system_path,
                                               available_resources,
                                               plot=plot)

        logger.info(f'Chosen Temperatures: {temperatures}')

        # Run the parallel tempering
        logger.info('Starting parallel tempering')
        enhanced_average_energy = _run_parallel_tempering(temperatures,
                                                          total_number_of_iterations,
                                                          steps_per_iteration,
                                                          number_of_molecules,
                                                          coordinate_path,
                                                          system_path,
                                                          available_resources)

        logger.info('Finished parallel tempering')
        parallel_tempering_end_time = time.perf_counter()

        # Run a regular simulation for comparison
        simulation_start_time = time.perf_counter()

        logger.info('Starting simulation')
        average_energy = _run_regular_simulation(298.0 * unit.kelvin,
                                                 total_number_of_iterations,
                                                 steps_per_iteration,
                                                 number_of_molecules,
                                                 coordinate_path,
                                                 system_path,
                                                 available_resources)

        logger.info('Finished simulation')
        simulation_end_time = time.perf_counter()

        logger.info(f'{smiles} {exchange_probability} {maximum_temperature} '
                    f'Temperatures {temperatures} '
                    f'Regular_Time {(simulation_end_time - simulation_start_time) * 1000} '
                    f'Parallel_Time {(parallel_tempering_end_time - parallel_tempering_start_time) * 1000} '
                    f'Regular_Energy {average_energy} '
                    f'Parallel_Energy {enhanced_average_energy}')


def main():

    # Define the core settings to use.
    number_of_molecules = 1

    smiles = ['CCOC(=O)CC(=O)C', 'CC(=O)CCC(=O)O', 'CCC(=O)O']

    exchange_probabilities = [0.2, 0.3, 0.4]
    maximum_temperatures = [1000 * unit.kelvin, 800 * unit.kelvin]

    # Set up a parallel backend.
    number_of_gpus = 0 if number_of_molecules == 1 else 1
    compute_resources = ComputeResources(number_of_threads=1, number_of_gpus=number_of_gpus,
                                         preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA)

    # Extract the indices to use
    parser = argparse.ArgumentParser()
    parser.add_argument('index', type=int)

    index = parser.parse_args().index
    setup_timestamp_logging(f'{index}.log')

    temperature_index = int(index % len(maximum_temperatures))
    probability_index = int((index / len(maximum_temperatures)) % len(exchange_probabilities))
    smiles_index = int(index / (len(exchange_probabilities) * len(maximum_temperatures)))

    maximum_temperature = maximum_temperatures[temperature_index]
    exchange_probability = exchange_probabilities[probability_index]
    smiles_pattern = smiles[smiles_index]

    directory_name = f'{smiles_index}_{probability_index}_{temperature_index}'
    os.makedirs(directory_name, exist_ok=True)

    _run(root_directory=directory_name,
         smiles=smiles_pattern,
         maximum_temperature=maximum_temperature,
         number_of_molecules=number_of_molecules,
         exchange_probability=exchange_probability,
         total_number_of_iterations=2000,
         steps_per_iteration=500,
         plot=False,
         available_resources=compute_resources)


def _compute_dihedral(trajectory, indices):

    import mdtraj

    dihedrals = mdtraj.compute_dihedrals(trajectory, indices=indices)[:, 0]
    dihedrals = dihedrals / np.pi * 180.0

    return dihedrals


def analyse_dihedrals():

    import mdtraj

    smiles = ['CCOC(=O)CC(=O)C', 'CC(=O)CCC(=O)O', 'CCC(=O)O']

    exchange_probabilities = [0.2, 0.3, 0.4]
    maximum_temperatures = [1000 * unit.kelvin, 800 * unit.kelvin]

    per_smiles_indices = {
        'CCOC(=O)CC(=O)C': np.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
        'CC(=O)CCC(=O)O': np.array([[6, 5, 7, 15]]),
        'CCC(=O)O': np.array([[3, 2, 4, 10]])
    }

    for temperature_index, maximum_temperature in enumerate(maximum_temperatures):

        for probability_index, exchange_probability in enumerate(exchange_probabilities):

            for smiles_index, smiles_pattern in enumerate(smiles):

                if smiles_pattern != 'CC(=O)CCC(=O)O':
                    continue

                directory_name = f'{smiles_index}_{probability_index}_{temperature_index}'

                coordinate_path = os.path.join(directory_name, 'input.pdb')

                original_trajectory_path = os.path.join(directory_name, 'run_regular', 'trajectory.dcd')
                parallel_trajectory_path = os.path.join(directory_name, 'run_parallel', 'trajectory.dcd')

                indices = per_smiles_indices[smiles_pattern]

                if not os.path.isfile(original_trajectory_path) or not os.path.isfile(parallel_trajectory_path):
                    continue

                original_trajectory = mdtraj.load_dcd(original_trajectory_path, top=coordinate_path)
                parallel_trajectory = mdtraj.load_dcd(parallel_trajectory_path, top=coordinate_path)

                original_dihedrals = _compute_dihedral(original_trajectory, indices)
                parallel_dihedrals = _compute_dihedral(parallel_trajectory, indices)

                figure, axes = plt.subplots(nrows=1,
                                            ncols=2,
                                            figsize=(17, 8.5))

                figure.suptitle(f'{smiles_pattern} {exchange_probability} {maximum_temperature:~}')

                axes[0].plot(original_dihedrals, alpha=0.5, label='Regular', marker='x', linestyle ='None')
                axes[0].plot(parallel_dihedrals, alpha=0.5, label='Parallel', marker='x', linestyle='None')

                axes[0].legend(loc='upper center')

                axes[1].hist(original_dihedrals, 180, range=[-180.0, 180], alpha=0.5, label='Regular', density=True)
                axes[1].hist(parallel_dihedrals, 180, range=[-180.0, 180], alpha=0.5, label='Parallel', density=True)

                axes[1].set_ylim(0.0, 0.04)

                axes[1].legend(loc='upper center')

                figure.tight_layout(pad=2)
                figure.show()


def analyse_energies():

    import seaborn
    seaborn.set_palette('colorblind')

    figure, axes = plt.subplots(nrows=1,
                                ncols=1,
                                figsize=(8, 8))

    # Plot the energies from the single temperature simulation.
    simulation_statistics = StatisticsArray.from_pandas_csv(f'0_0_0/run_regular/statistics.csv')
    reduced_potentials = simulation_statistics[ObservableType.ReducedPotential].magnitude

    mu = np.mean(reduced_potentials)
    sigma = np.std(reduced_potentials)

    x_values = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axes.plot(x_values, norm.pdf(x_values, mu, sigma), label=f'Simulation')

    # Plot the energies from the parallel tempering simulation.
    simulation_statistics = StatisticsArray.from_pandas_csv(f'0_0_0/run_parallel/statistics.csv')
    reduced_potentials = simulation_statistics[ObservableType.ReducedPotential].magnitude

    mu = np.mean(reduced_potentials)
    sigma = np.std(reduced_potentials)

    x_values = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    axes.plot(x_values, norm.pdf(x_values, mu, sigma), label=f'Simulation')

    axes.legend(loc='upper right')

    figure.show()


def verify_energies(trajectory_path, statistics_path):

    compute_resources = ComputeResources(number_of_threads=1)

    protocol = reweighting.CalculateReducedPotentialOpenMM('')
    protocol.thermodynamic_state = ThermodynamicState(298.0 * unit.kelvin,)
    protocol.system_path = 'ideal_system.xml'
    protocol.enable_pbc = False
    protocol.coordinate_file_path = 'input.pdb'
    protocol.trajectory_file_path = trajectory_path
    protocol.high_precision = True

    assert not isinstance(protocol.execute('', compute_resources), PropertyEstimatorException)

    trajectory_statistics = StatisticsArray.from_pandas_csv(protocol.statistics_file_path)
    reporter_statistics = StatisticsArray.from_pandas_csv(statistics_path)

    assert np.allclose(trajectory_statistics[ObservableType.ReducedPotential].magnitude,
                       reporter_statistics[ObservableType.ReducedPotential].magnitude, atol=1e-0)

    assert np.allclose(trajectory_statistics[ObservableType.PotentialEnergy].magnitude,
                       reporter_statistics[ObservableType.PotentialEnergy].magnitude, atol=1e-0)


if __name__ == "__main__":

    main()
    analyse_energies()
    # analyse_dihedrals()
