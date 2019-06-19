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

            for unitted_attribute in parameter._REQUIRE_UNITS:

                if not hasattr(parameter, unitted_attribute):
                    continue

                parameter_value = getattr(parameter, unitted_attribute)

                if not isinstance(parameter_value, unit.Quantity):
                    continue

                parameter_value = parameter_value.value_in_unit_system(unit.md_unit_system)

                if not isinstance(parameter_value, float) and not isinstance(parameter_value, int):
                    continue

                yield parameter_tag, parameter, unitted_attribute


def evaluate_reduced_potential(force_field, topology, trajectory, charged_molecules=None,
                               system_file_name=None):
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
        A list of pre-charged molecules, from which to source the system charges.
    system_file_name: str, optional
        The name of the system xml file to save out. If `None`, no file is saved.

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
    platform = Platform.getPlatformByName('Reference')
    properties = {}
    # platform = Platform.getPlatformByName('CUDA')
    # properties = {'Precision': 'double'}

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


def estimate_gradient(trajectory, topology, original_parameter, parameter_tag,
                      unitted_attribute, scale_amount, full_force_field, use_subset=False):

    if use_subset:

        reverse_force_field = ForceField()
        reverse_handler = copy.deepcopy(full_force_field.get_parameter_handler(parameter_tag))
        reverse_force_field.register_parameter_handler(reverse_handler)

    else:

        reverse_force_field = copy.deepcopy(full_force_field)
        reverse_handler = reverse_force_field.get_parameter_handler(parameter_tag)

    existing_parameter = reverse_handler.parameters[original_parameter.smirks]

    reverse_value = getattr(original_parameter, unitted_attribute) * (1.0 - scale_amount)
    setattr(existing_parameter, unitted_attribute, reverse_value)

    reverse_reduced_energies = evaluate_reduced_potential(reverse_force_field,
                                                          topology,
                                                          trajectory,
                                                          system_file_name='reverse.xml')

    if use_subset:

        forward_force_field = ForceField()
        forward_handler = copy.deepcopy(full_force_field.get_parameter_handler(parameter_tag))
        forward_force_field.register_parameter_handler(forward_handler)

    else:

        forward_force_field = copy.deepcopy(full_force_field)
        forward_handler = forward_force_field.get_parameter_handler(parameter_tag)

    existing_parameter = forward_handler.parameters[original_parameter.smirks]

    forward_value = getattr(original_parameter, unitted_attribute) * (1.0 + scale_amount)
    setattr(existing_parameter, unitted_attribute, forward_value)

    forward_reduced_energies = evaluate_reduced_potential(forward_force_field,
                                                          topology,
                                                          trajectory,
                                                          system_file_name='forward.xml')

    gradient = (forward_reduced_energies - reverse_reduced_energies) / (forward_value - reverse_value)

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

    for parameter_tag, parameter, unitted_attribute in all_differentiable_parameters(full_force_field,
                                                                                     methanol_topology):

        start_time = time.perf_counter()

        gradient_fast = estimate_gradient(methanol_trajectory,
                                          methanol_topology,
                                          parameter,
                                          parameter_tag,
                                          unitted_attribute,
                                          1.0e-4,
                                          full_force_field=full_force_field,
                                          use_subset=True)

        end_time = time.perf_counter()
        fast_time = (end_time-start_time)

        logging.info(f'Finished fast gradient estimate after {fast_time*1000} ms ('
                     f'{parameter_tag} {unitted_attribute} {parameter.smirks}).')

        gradient_slow = estimate_gradient(methanol_trajectory,
                                          methanol_topology,
                                          parameter,
                                          parameter_tag,
                                          unitted_attribute,
                                          1.0e-4,
                                          full_force_field=full_force_field,
                                          use_subset=False)

        end_time = time.perf_counter()
        slow_time = (end_time - start_time)

        logging.info(f'Finished slow gradient estimate after {slow_time*1000} ms ('
                     f'{parameter_tag} {unitted_attribute} {parameter.smirks}).')

        abs_difference = abs(gradient_fast - gradient_slow)

        print(f'{parameter_tag}: {parameter.smirks} - dE/{unitted_attribute} speedup={slow_time/fast_time:.2f}X '
              f'slow={gradient_slow} fast={gradient_fast} absdiff={abs_difference}')


if __name__ == "__main__":
    estimate_gradients()
