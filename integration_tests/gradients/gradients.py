#!/usr/bin/env python
import copy
import logging

import mdtraj
import numpy as np
from openforcefield.topology import Molecule, Topology
from openforcefield.typing.engines.smirnoff import ForceField
from simtk import openmm, unit
from simtk.openmm import XmlSerializer
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

    openmm_context = openmm.Context(system, integrator)

    reduced_potentials = np.zeros(trajectory.n_frames)

    for frame_index in range(trajectory.n_frames):
        positions = trajectory.openmm_positions(frame_index)
        box_vectors = trajectory.openmm_boxes(frame_index)

        openmm_context.setPeriodicBoxVectors(*box_vectors)
        openmm_context.setPositions(positions)

        state = openmm_context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)

        # set box vectors
        reduced_potentials[frame_index] = potential_energy

    reduced_potentials *= unit.kilojoules_per_mole * (beta / unit.AVOGADRO_CONSTANT_NA)
    return reduced_potentials


def estimate_gradient(trajectory, topology, original_parameter, parameter_handler_class, unitted_attribute):

    original_parameter_value = getattr(original_parameter, unitted_attribute)

    logging.info(f'Building reverse perturbed parameter for {unitted_attribute}.')
    reverse_parameter = copy.deepcopy(original_parameter)

    reverse_parameter_value = getattr(reverse_parameter, unitted_attribute) * (1.0 - 1e-6)
    setattr(reverse_parameter, unitted_attribute, reverse_parameter_value)

    logging.info('Building reverse force field.')
    reverse_force_field = ForceField()
    reverse_handler = parameter_handler_class(skip_version_check=True)
    reverse_handler.parameters.append(reverse_parameter)
    reverse_force_field.register_parameter_handler(reverse_handler)

    logging.info('Evaluating reverse reduced potential.')
    reverse_reduced_energies = evaluate_reduced_potential(reverse_force_field,
                                                          topology,
                                                          trajectory,
                                                          system_file_name='reverse.xml')

    logging.info(f'Building forward perturbed parameter for {unitted_attribute}.')
    forward_parameter = copy.deepcopy(original_parameter)

    forward_parameter_value = getattr(forward_parameter, unitted_attribute) * (1.0 + 1e-6)
    setattr(forward_parameter, unitted_attribute, forward_parameter_value)

    logging.info('Building forward force field.')
    forward_force_field = ForceField()
    forward_bond_handler = parameter_handler_class(skip_version_check=True)
    forward_bond_handler.parameters.append(forward_parameter)
    forward_force_field.register_parameter_handler(forward_bond_handler)

    logging.info('Evaluating forward reduced potential.')
    forward_reduced_energies = evaluate_reduced_potential(forward_force_field,
                                                          topology,
                                                          trajectory,
                                                          system_file_name='forward.xml')

    reverse_parameter_value = getattr(reverse_parameter, unitted_attribute)

    energy_differences = forward_reduced_energies - reverse_reduced_energies
    denominator = forward_parameter_value - reverse_parameter_value

    gradient = energy_differences.mean() / denominator
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

    for parameter_tag in ['vdW']:  # all_parameters:

        if parameter_tag in ['Electrostatics', 'ToolkitAM1BCC']:
            continue

        parameter_dictionary = all_parameters[parameter_tag]
        parameter_handler_class = full_force_field.get_parameter_handler(parameter_tag).__class__

        parameters_by_smirks = {}

        for parameter in parameter_dictionary.store.values():
            parameters_by_smirks[parameter.smirks] = parameter

        for smirks in parameters_by_smirks:

            parameter = parameters_by_smirks[smirks]

            for unitted_attribute in parameter._REQUIRE_UNITS:

                if not hasattr(parameter, unitted_attribute):
                    continue

                parameter_value = getattr(parameter, unitted_attribute)

                if not isinstance(parameter_value, unit.Quantity):
                    continue

                parameter_value = parameter_value.value_in_unit_system(unit.md_unit_system)

                if not isinstance(parameter_value, float) and not isinstance(parameter_value, int):
                    continue

                gradient = estimate_gradient(methanol_trajectory,
                                             methanol_topology,
                                             parameter,
                                             parameter_handler_class,
                                             unitted_attribute)

                print(f'{parameter_tag}: {smirks} - dE/{unitted_attribute} = {gradient}')


if __name__ == "__main__":
    estimate_gradients()
