#!/usr/bin/env python
import copy
import logging
import shutil
import time

from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField
from simtk import unit
from simtk.openmm import XmlSerializer
from simtk.openmm.app import PDBFile

from propertyestimator.backends import ComputeResources
from propertyestimator.protocols.analysis import ExtractAverageStatistic, ExtractUncorrelatedTrajectoryData, \
    ExtractUncorrelatedStatisticsData
from propertyestimator.protocols.reweighting import ReweightWithMBARProtocol, CalculateReducedPotentialOpenMM
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import get_data_filename, setup_timestamp_logging, statistics


def all_differentiable_parameters(force_field, topology):
    """A generator function which yields all of the differentiable parameters
    which may be applied to a given topology.

    Parameters
    ----------
    force_field: openforcefield.typing.engines.smirnoff.ForceField
        The force field being applied.
    topology: openforcefield.topology.Topology
        The topology the force field is being applied to.

    Returns
    -------
    str
        The type of parameter, e.g. Bonds, Angles...
    ParameterType
        The differentiable parameter type.
    str
        The differentiable attribute (e.g. k or length) of the parameter.
    """

    parameters_by_tag_smirks = {}

    for parameter_set in force_field.label_molecules(topology):

        for parameter_tag in parameter_set:

            if parameter_tag in ['Electrostatics', 'ToolkitAM1BCC']:
                continue

            if parameter_tag not in parameters_by_tag_smirks:
                parameters_by_tag_smirks[parameter_tag] = {}

            for parameter in parameter_set[parameter_tag].store.values():
                parameters_by_tag_smirks[parameter_tag][parameter.smirks] = parameter

    for parameter_tag in parameters_by_tag_smirks:

        for smirks in parameters_by_tag_smirks[parameter_tag]:

            parameter = parameters_by_tag_smirks[parameter_tag][smirks]

            for parameter_attribute in parameter._REQUIRE_UNITS:

                if not hasattr(parameter, parameter_attribute):
                    continue

                parameter_value = getattr(parameter, parameter_attribute)

                if not isinstance(parameter_value, unit.Quantity):
                    continue

                parameter_value = parameter_value.value_in_unit_system(unit.md_unit_system)

                if not isinstance(parameter_value, float) and not isinstance(parameter_value, int):
                    continue

                yield parameter_tag, parameter, parameter_attribute


def evaluate_reduced_potential(force_field, topology, trajectory_path, thermodynamic_state,
                               compute_resources, output_path=None, charged_molecules=None):
    """Evaluates the reduced potential of a given trajectory using the
    provided force field, and stores the results in a pandas csv file.

    Parameters
    ----------
    force_field: openforcefield.typing.engines.smirnoff.ForceField
        The force field to use when evaluating the energy.
    topology: openforcefield.topology.Topology
        The topology of the system.
    trajectory_path: str
        The file path to the trajectory of configurations to evaluate.
    thermodynamic_state: ThermodynamicState
        The state to evaluate the reduced potential at.
    compute_resources: ComputeResources
        The compute resources to use to evaluate the reduced potentials.
    output_path: str
        The path to save the reduced potential csv file to.
    charged_molecules: list of openforcefield.topology.Molecule
        A list of pre-charged molecules, from which to source the system charges.
    """

    if charged_molecules is None:
        system = force_field.create_openmm_system(topology=topology, allow_missing_parameters=True)
    else:
        system = force_field.create_openmm_system(topology=topology,
                                                  allow_missing_parameters=True,
                                                  charge_from_molecules=charged_molecules)

    system_xml = XmlSerializer.serialize(system)

    with open('system.xml', 'wb') as file:
        file.write(system_xml.encode('utf-8'))

    reduced_potential_protocol = CalculateReducedPotentialOpenMM('reduced_potential_protocol')
    reduced_potential_protocol.thermodynamic_state = thermodynamic_state
    reduced_potential_protocol.system_path = 'system.xml'
    reduced_potential_protocol.coordinate_file_path = 'methanol/methanol.pdb'
    reduced_potential_protocol.trajectory_file_path = trajectory_path
    reduced_potential_protocol.high_precision = True

    reduced_potential_protocol.execute('', compute_resources)

    if output_path is not None:
        shutil.move(reduced_potential_protocol.statistics_file_path, output_path)


def estimate_gradient(trajectory_path, topology, thermodynamic_state, original_parameter, parameter_tag,
                      parameter_attribute, observable_values, scale_amount, full_force_field,
                      compute_resources, use_subset=False, reference_statistics_path=None):

    if reference_statistics_path is None:

        logging.info('Calculating reference potentials.')

        if use_subset:

            reference_force_field = ForceField()
            reference_handler = copy.deepcopy(full_force_field.get_parameter_handler(parameter_tag))
            reference_force_field.register_parameter_handler(reference_handler)

        else:

            reference_force_field = full_force_field

        reference_statistics_path = 'reference.csv'

        evaluate_reduced_potential(reference_force_field,
                                   topology,
                                   trajectory_path,
                                   thermodynamic_state,
                                   compute_resources,
                                   reference_statistics_path)

    logging.info('Calculating reverse potentials.')

    if use_subset:

        reverse_force_field = ForceField()
        reverse_handler = copy.deepcopy(full_force_field.get_parameter_handler(parameter_tag))
        reverse_force_field.register_parameter_handler(reverse_handler)

    else:

        reverse_force_field = copy.deepcopy(full_force_field)
        reverse_handler = reverse_force_field.get_parameter_handler(parameter_tag)

    existing_parameter = reverse_handler.parameters[original_parameter.smirks]

    reverse_parameter_value = getattr(original_parameter, parameter_attribute) * (1.0 - scale_amount)
    setattr(existing_parameter, parameter_attribute, reverse_parameter_value)

    evaluate_reduced_potential(reverse_force_field,
                               topology,
                               trajectory_path,
                               thermodynamic_state,
                               compute_resources,
                               'reverse.csv')

    logging.info('Calculating reverse mbar.')

    mbar_protocol = ReweightWithMBARProtocol('mbar_protocol')
    mbar_protocol.reference_reduced_potentials = [reference_statistics_path]
    mbar_protocol.reference_observables = [observable_values]
    mbar_protocol.target_reduced_potentials = ['reverse.csv']
    mbar_protocol.required_effective_samples = 0
    mbar_protocol.bootstrap_uncertainties = False
    mbar_protocol.execute('', None)

    reverse_value = copy.deepcopy(mbar_protocol.value)

    logging.info('Calculating forward potentials.')

    if use_subset:

        forward_force_field = ForceField()
        forward_handler = copy.deepcopy(full_force_field.get_parameter_handler(parameter_tag))
        forward_force_field.register_parameter_handler(forward_handler)

    else:

        forward_force_field = copy.deepcopy(full_force_field)
        forward_handler = forward_force_field.get_parameter_handler(parameter_tag)

    existing_parameter = forward_handler.parameters[original_parameter.smirks]

    forward_parameter_value = getattr(original_parameter, parameter_attribute) * (1.0 + scale_amount)
    setattr(existing_parameter, parameter_attribute, forward_parameter_value)

    evaluate_reduced_potential(forward_force_field,
                               topology,
                               trajectory_path,
                               compute_resources,
                               thermodynamic_state,
                               'forward.csv')

    logging.info('Calculating forward mbar.')

    mbar_protocol.target_reduced_potentials = ['forward.csv']
    mbar_protocol.execute('', None)

    forward_value = copy.deepcopy(mbar_protocol.value)

    gradient = (forward_value.value - reverse_value.value) / (forward_parameter_value - reverse_parameter_value)

    logging.info(f'Reverse value: {reverse_value.value}+/-{reverse_value.uncertainty} '
                 f'Forward value: {forward_value.value}+/-{forward_value.uncertainty} '
                 f'Gradient: {gradient}')

    return gradient


def estimate_gradients():
    """An integrated test of calculating the gradients of observables with
    respect to force field parameters using the property estimator"""

    setup_timestamp_logging()

    compute_resource = ComputeResources(number_of_threads=1, number_of_gpus=1,
                                        preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA)

    # compute_resource = ComputeResources(number_of_threads=1)

    thermodynamic_state = ThermodynamicState(temperature=298*unit.kelvin,
                                             pressure=1.0*unit.atmosphere)

    logging.info('Setting up molecule.')
    methanol_pdb_file = PDBFile(file='methanol/methanol.pdb')

    methanol_molecule = Molecule.from_smiles(smiles='CO')

    methanol_topology = Topology.from_openmm(openmm_topology=methanol_pdb_file.topology,
                                             unique_molecules=[methanol_molecule])

    logging.info('Loading force field.')
    full_force_field = ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))

    extract_densities = ExtractAverageStatistic('extract_densities')
    extract_densities.bootstrap_iterations = 0
    extract_densities.statistics_path = 'methanol/methanol.csv'
    extract_densities.statistics_type = statistics.ObservableType.Density
    extract_densities.execute('', None)

    extract_trajectory = ExtractUncorrelatedTrajectoryData('extract_trajectory')
    extract_trajectory.input_coordinate_file = 'methanol/methanol.pdb'
    extract_trajectory.input_trajectory_path = 'methanol/methanol.dcd'
    extract_trajectory.equilibration_index = extract_densities.equilibration_index
    extract_trajectory.statistical_inefficiency = extract_densities.statistical_inefficiency
    extract_trajectory.execute('', None)

    extract_statistics = ExtractUncorrelatedStatisticsData('extract_statistics')
    extract_statistics.input_statistics_path = 'methanol/methanol.csv'
    extract_statistics.equilibration_index = extract_densities.equilibration_index
    extract_statistics.statistical_inefficiency = extract_densities.statistical_inefficiency
    extract_statistics.execute('', None)

    logging.info(f'Loading the trajectory.')
    methanol_trajectory_path = extract_trajectory.output_trajectory_path
    logging.info(f'Loaded the trajectory.')

    for parameter_tag, parameter, parameter_attribute in all_differentiable_parameters(full_force_field,
                                                                                       methanol_topology):

        start_time = time.perf_counter()

        gradient_fast = estimate_gradient(methanol_trajectory_path,
                                          methanol_topology,
                                          thermodynamic_state,
                                          parameter,
                                          parameter_tag,
                                          parameter_attribute,
                                          extract_densities.uncorrelated_values,
                                          1.0e-4,
                                          full_force_field=full_force_field,
                                          compute_resources=compute_resource,
                                          use_subset=True)

        end_time = time.perf_counter()
        fast_time = (end_time-start_time)

        logging.info(f'Finished fast gradient estimate after {fast_time*1000} ms ('
                     f'{parameter_tag} {parameter_attribute} {parameter.smirks}).')

        gradient_slow = estimate_gradient(methanol_trajectory_path,
                                          methanol_topology,
                                          thermodynamic_state,
                                          parameter,
                                          parameter_tag,
                                          parameter_attribute,
                                          extract_densities.uncorrelated_values,
                                          1.0e-4,
                                          full_force_field=full_force_field,
                                          compute_resources=compute_resource,
                                          use_subset=False,
                                          reference_statistics_path=extract_statistics.output_statistics_path)

        end_time = time.perf_counter()
        slow_time = (end_time - start_time)

        logging.info(f'Finished slow gradient estimate after {slow_time*1000} ms ('
                     f'{parameter_tag} {parameter_attribute} {parameter.smirks}).')

        absolute_difference = gradient_fast - gradient_slow
        relative_difference = (gradient_fast - gradient_slow) / abs(gradient_slow)

        print(f'{parameter_tag}: {parameter.smirks} - dE/{parameter_attribute} '
              f'speedup={slow_time/fast_time:.2f}X '
              f'slow={gradient_slow} '
              f'fast={gradient_fast} '
              f'absdiff={absolute_difference} '
              f'reldiff={relative_difference}')


if __name__ == "__main__":
    estimate_gradients()
