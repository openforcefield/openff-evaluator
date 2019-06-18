#!/usr/bin/env python
import copy
import logging
import time

import mdtraj
import numpy as np
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField
from simtk import openmm, unit
from simtk.openmm import XmlSerializer, Platform
from simtk.openmm.app import PDBFile

from propertyestimator.utils import get_data_filename, setup_timestamp_logging


def evaluate_reduced_potential(force_field, topology, trajectory, charged_molecules=None, system_file_name=None):
    """Return the potential energy.

    Parameters
    ----------
    force_field: openforcefield.typing.engines.smirnoff.ForceField
        The force field to use when evaluating the energy.
    topology: openforcefield.topology.Topology
        The topology of the system.
    trajectory: mdtraj.Trajectory
        A trajectory of configurations to evaluate.
    charged_molecules: list of openforcefield.topology.Molecule

    Returns
    ---------
    simtk.unit.Quantity
        The energy of the provided configuration evaluated using the provided
        force field parameters.
    """

    if charged_molecules is None:
        system = force_field.create_openmm_system(topology=topology, allow_missing_parameters=True)
    else:
        system = force_field.create_openmm_system(topology=topology,
                                                  allow_missing_parameters=True,
                                                  charge_from_molecules=charged_molecules)

    if system_file_name is not None:

        system_xml = XmlSerializer.serialize(system)

        with open(system_file_name, 'wb') as file:
            file.write(system_xml.encode('utf-8'))

    beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * 298.0 * unit.kelvin)

    integrator = openmm.VerletIntegrator(1.0*unit.femtoseconds)

    # noinspection PyTypeChecker,PyCallByClass
    # platform = Platform.getPlatformByName('Reference')
    platform = Platform.getPlatformByName('CUDA')
    properties = {'Precision': 'double'}

    openmm_context = openmm.Context(system, integrator, platform, properties)

    reduced_potentials = np.zeros(trajectory.n_frames)

    for frame_index in range(trajectory.n_frames):
        # positions = trajectory.openmm_positions(frame_index)
        positions = trajectory.xyz[frame_index]
        box_vectors = trajectory.openmm_boxes(frame_index)

        openmm_context.setPeriodicBoxVectors(*box_vectors)
        openmm_context.setPositions(positions)

        state = openmm_context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        # set box vectors
        reduced_potentials[frame_index] = potential_energy

    reduced_potentials *= unit.kilojoules_per_mole * (beta / unit.AVOGADRO_CONSTANT_NA)
    return reduced_potentials


def estimate_gradient(trajectory, topology, original_parameter, parameter_tag, parameter_handler_class,
                      unitted_attribute, scale_amount, *extra_parameters, full_force_field=None):

    logging.debug(f'Building reverse perturbed parameter for {unitted_attribute}.')
    reverse_parameter = copy.deepcopy(original_parameter)

    reverse_parameter_value = getattr(reverse_parameter, unitted_attribute) * (1.0 - scale_amount)
    setattr(reverse_parameter, unitted_attribute, reverse_parameter_value)

    logging.debug('Building reverse force field.')

    if full_force_field is None:
        reverse_force_field = ForceField()

        reverse_handler = parameter_handler_class(skip_version_check=True)
        reverse_handler.parameters.append(reverse_parameter)

        for extra_parameter in extra_parameters:
            reverse_handler.parameters.append(extra_parameter)

        reverse_force_field.register_parameter_handler(reverse_handler)

    else:

        reverse_force_field = copy.deepcopy(full_force_field)

        existing_parameter = reverse_force_field.get_parameter_handler(parameter_tag).parameters[
            reverse_parameter.smirks]

        setattr(existing_parameter, unitted_attribute, reverse_parameter_value)

    logging.debug('Evaluating reverse reduced potential.')
    reverse_reduced_energies = evaluate_reduced_potential(reverse_force_field,
                                                          topology,
                                                          trajectory,
                                                          system_file_name='reverse.xml')

    logging.debug(f'Building forward perturbed parameter for {unitted_attribute}.')
    forward_parameter = copy.deepcopy(original_parameter)

    forward_parameter_value = getattr(forward_parameter, unitted_attribute) * (1.0 + scale_amount)
    setattr(forward_parameter, unitted_attribute, forward_parameter_value)

    logging.debug('Building forward force field.')

    if full_force_field is None:

        forward_force_field = ForceField()

        forward_handler = parameter_handler_class(skip_version_check=True)
        forward_handler.parameters.append(forward_parameter)

        for extra_parameter in extra_parameters:
            forward_handler.parameters.append(extra_parameter)

        forward_force_field.register_parameter_handler(forward_handler)

    else:

        forward_force_field = copy.deepcopy(full_force_field)

        existing_parameter = forward_force_field.get_parameter_handler(parameter_tag).parameters[
            forward_parameter.smirks]

        setattr(existing_parameter, unitted_attribute, forward_parameter_value)

    logging.debug('Evaluating forward reduced potential.')
    forward_reduced_energies = evaluate_reduced_potential(forward_force_field,
                                                          topology,
                                                          trajectory,
                                                          system_file_name='forward.xml')

    reverse_parameter_value = getattr(reverse_parameter, unitted_attribute)

    energy_differences = forward_reduced_energies - reverse_reduced_energies
    denominator = forward_parameter_value - reverse_parameter_value

    gradient = energy_differences.mean() / denominator

    logging.info(f'delta_E={energy_differences} '
                 f'delta_value={denominator} '
                 f'grad={gradient} '
                 f'is_close={np.allclose(energy_differences, np.array([0.0]*len(energy_differences)))}')

    return gradient


def estimate_gradients():
    """An integrated test of calculating the gradients of observables with
    respect to force field parameters using the property estimator"""

    setup_timestamp_logging()

    logging.info('Setting up molecule.')
    methanol_pdb_file = PDBFile(file='methanol/methanol.pdb')

    logging.info(f'Loading the trajectory.')
    methanol_trajectory = mdtraj.load('methanol/methanol.dcd',
                                      top='methanol/methanol.pdb')
    logging.info(f'Loaded the trajectory.')

    methanol_molecule = Molecule.from_smiles(smiles='CO')

    methanol_topology = Topology.from_openmm(openmm_topology=methanol_pdb_file.topology,
                                             unique_molecules=[methanol_molecule])

    logging.info('Loading force field.')
    full_force_field = ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))

    all_parameters = full_force_field.label_molecules(methanol_topology)[0]

    scale_amounts = [1.0e-3, 1.0e-4, 1.0e-5]

    for parameter_tag in all_parameters:  # ['vdW']:

        if parameter_tag in ['Electrostatics', 'ToolkitAM1BCC']:
            continue

        parameter_dictionary = all_parameters[parameter_tag]
        parameter_handler_class = full_force_field.get_parameter_handler(parameter_tag).__class__

        parameters_by_smirks = {}

        for parameter in parameter_dictionary.store.values():
            parameters_by_smirks[parameter.smirks] = parameter

        for smirks in parameters_by_smirks:

            parameter = parameters_by_smirks[smirks]

            remaining_parameters = {}

            if parameter_tag == 'vdW':

                remaining_parameters = {
                    key: parameters_by_smirks[key] for key in parameters_by_smirks if key != smirks
                }

            for unitted_attribute in parameter._REQUIRE_UNITS:

                if not hasattr(parameter, unitted_attribute):
                    continue

                parameter_value = getattr(parameter, unitted_attribute)

                if not isinstance(parameter_value, unit.Quantity):
                    continue

                parameter_value = parameter_value.value_in_unit_system(unit.md_unit_system)

                if not isinstance(parameter_value, float) and not isinstance(parameter_value, int):
                    continue

                logging.info(f'Starting fast gradient estimate ('
                             f'{parameter_tag} {unitted_attribute} {smirks}).')
                start_time = time.perf_counter()

                gradients_fast = []

                for scale_amount in scale_amounts:

                    gradients_fast.append(estimate_gradient(methanol_trajectory,
                                                            methanol_topology,
                                                            parameter,
                                                            parameter_tag,
                                                            parameter_handler_class,
                                                            unitted_attribute,
                                                            scale_amount,
                                                            *remaining_parameters.values()))

                end_time = time.perf_counter()
                fast_time = (end_time-start_time) / float(len(scale_amounts))

                logging.info(f'Finished fast gradient estimate after {fast_time*1000} ms ('
                             f'{parameter_tag} {unitted_attribute} {smirks}).')

                logging.info(f'Starting slow gradient estimate ('
                             f'{parameter_tag} {unitted_attribute} {smirks}).')
                start_time = time.perf_counter()

                gradient_slow = estimate_gradient(methanol_trajectory,
                                                  methanol_topology,
                                                  parameter,
                                                  parameter_tag,
                                                  parameter_handler_class,
                                                  unitted_attribute,
                                                  *remaining_parameters.values(),
                                                  full_force_field=full_force_field)

                end_time = time.perf_counter()
                slow_time = (end_time - start_time)

                logging.info(f'Finished slow gradient estimate after {slow_time*1000} ms ('
                             f'{parameter_tag} {unitted_attribute} {smirks}).')

                # print(f'{parameter_tag}: {smirks} - dE/{unitted_attribute} fast={gradient_fast}')

                abs_differences = [abs(gradient_fast - gradient_slow) for gradient_fast in gradients_fast]

                print(f'{parameter_tag}: {smirks} - dE/{unitted_attribute} speedup={slow_time/fast_time:.2f}X '
                      f'slow={gradient_slow} scales={scale_amounts} fast={gradients_fast} abs_diff={abs_differences}')


if __name__ == "__main__":
    estimate_gradients()
