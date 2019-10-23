import logging
import os
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np
from openforcefield.topology import Molecule, Topology
from scipy.optimize import newton
from scipy.stats import norm
from simtk import openmm, unit as simtk_unit
from simtk.openmm import app

from propertyestimator import unit
from propertyestimator.backends import ComputeResources
from propertyestimator.protocols import coordinates, forcefield, simulation
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.thermodynamics import Ensemble, ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import disable_pbc
from propertyestimator.utils.statistics import StatisticsArray, ObservableType
from propertyestimator.utils.utils import temporarily_change_directory


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
        protocol.steps_per_iteration = 50000
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

    if plot:
        plt.show()

    # Fit a polynomial
    coefficients = np.polyfit(temperatures.to(unit.kelvin).magnitude, mu, 2)

    current_temperature = temperatures[0]
    replica_temperatures = [current_temperature]

    while current_temperature < maximum_temperature:

        def evaluate(x):

            beta_1 = 1.0 / (current_temperature * unit.molar_gas_constant)
            beta_2 = 1.0 / (x * unit.kelvin * unit.molar_gas_constant)

            energy_1 = (coefficients[0] * current_temperature.to(unit.kelvin).magnitude ** 2 +
                        coefficients[1] * current_temperature.to(unit.kelvin).magnitude +
                        coefficients[2]) * unit.kilojoule / unit.mole

            energy_2 = (coefficients[0] * x ** 2 +
                        coefficients[1] * x +
                        coefficients[2]) * unit.kilojoule / unit.mole

            return ((beta_1 - beta_2) *
                    (energy_1 - energy_2)).to(unit.dimensionless).magnitude - np.log(desired_probability)

        def evaluate_gradient(x):

            beta_1 = 1.0 / (current_temperature * unit.molar_gas_constant)
            beta_2 = 1.0 / (x * unit.kelvin * unit.molar_gas_constant)

            energy_gradient = (coefficients[0] * x * 2 +
                               coefficients[1]) * unit.kilojoule / unit.mole

            return -((beta_1 - beta_2) * energy_gradient).to(unit.dimensionless).magnitude

        new_temperature = newton(evaluate,
                                 current_temperature.to(unit.kelvin).magnitude * 1.5,
                                 evaluate_gradient,
                                 tol=5.0,
                                 maxiter=10000) * unit.kelvin

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


def _run_parallel_tempering(temperatures, total_number_of_iterations, steps_per_iteration, number_of_molecules,
                            coordinate_path, system_path, compute_resources, plot=False):
    """Perform the parallel tempering simulation using the
    OpenMMParallelTempering protocol.
    """

    if os.path.isdir('run'):
        shutil.rmtree('run')

    os.makedirs('run', exist_ok=True)

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

    result = protocol.execute('run', compute_resources)
    assert not isinstance(result, PropertyEstimatorException)

    if plot:

        import mdtraj

        trajectory = mdtraj.load_dcd(protocol.trajectory_file_path, top=coordinate_path)

        dihedrals = mdtraj.compute_dihedrals(trajectory, indices=np.array([[3, 2, 4, 10]]))[:, 0]
        dihedrals = dihedrals / np.pi * 180.0 + 180

        num_bins = 180

        plt.hist(dihedrals, num_bins, range=[0.0, 360])
        plt.show()

        statistics = StatisticsArray.from_pandas_csv(protocol.statistics_file_path)
        plt.plot(statistics[ObservableType.PotentialEnergy].magnitude)
        plt.show()


def _run(root_directory, smiles, maximum_temperature, number_of_molecules, exchange_probability,
         total_number_of_iterations, steps_per_iteration):

    """A convenience function for running the full parallel tempering workflow
    with a given set of parameters.
    """

    with temporarily_change_directory(root_directory):

        # Define the resources to use.
        number_of_gpus = 1 if number_of_molecules == 1 else 0
        compute_resources = ComputeResources(number_of_threads=1, number_of_gpus=number_of_gpus,
                                             preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA)

        coordinate_path = 'input.pdb'
        system_path = 'system.xml'

        # Create the initial system object and coordinates.
        os.makedirs('setup', exist_ok=True)

        _build_system(smiles='CCC(=O)O',
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

        # Determine the replica temperatures
        temperatures = _determine_temperatures(298.0 * unit.kelvin,
                                               maximum_temperature,
                                               1,
                                               number_of_molecules,
                                               exchange_probability,
                                               coordinate_path,
                                               ideal_system_path,
                                               compute_resources,
                                               plot=False)

        # Run the parallel tempering
        _run_parallel_tempering(temperatures,
                                total_number_of_iterations,
                                steps_per_iteration,
                                number_of_molecules,
                                coordinate_path,
                                system_path,
                                compute_resources,
                                plot=False)


def main():

    formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                                  datefmt='%H:%M:%S')

    logger_handler = logging.StreamHandler(stream=sys.stdout)
    logger_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logger_handler)

    # Define the core settings to use.
    smiles = ['CCC(=O)O', 'CC(=O)CCC(=O)O', 'CCOC(=O)CC(=O)C']

    exchange_probabilities = [0.2, 0.3, 0.4]
    maximum_temperatures = [600 * unit.kelvin, 800 * unit.kelvin, 1000 * unit.kelvin]

    for smiles_index, smiles in enumerate(smiles):

        for probability_index, exchange_probability in enumerate(exchange_probabilities):

            for temperature_index, maximum_temperature in enumerate(maximum_temperatures):

                directory_name = f'{smiles_index}_{probability_index}_{temperature_index}'
                os.makedirs(directory_name, exist_ok=True)

                logger.info(f'Starting {smiles} {exchange_probability} {maximum_temperature}')

                try:

                    _run(root_directory=directory_name,
                         smiles=smiles,
                         maximum_temperature=maximum_temperature,
                         number_of_molecules=1,
                         exchange_probability=exchange_probability,
                         total_number_of_iterations=10000,
                         steps_per_iteration=100)

                except:
                    logger.info(f'Run failed')
                else:
                    logger.info(f'Finished run.')


if __name__ == "__main__":
    main()
