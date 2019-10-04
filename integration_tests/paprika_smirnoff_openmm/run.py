#!/usr/bin/env python
import logging
import os as os

from propertyestimator import unit

from integration_tests.utils import get_paprika_host_guest_substance
from propertyestimator.backends import ComputeResources
from propertyestimator.protocols.paprika import OpenMMPaprikaProtocol
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import setup_timestamp_logging
from propertyestimator.utils.exceptions import PropertyEstimatorException


def main():
    """An integrated test of calculating the gradients of observables with
    respect to force field parameters using the property estimator"""
    setup_timestamp_logging()

    host = 'acd'
    guest = 'bam'

    # Set up the object which describes how many compute resources available
    # on the machine on which the calculations will run.
    resources = ComputeResources(number_of_threads=1, number_of_gpus=0,
                                 preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA)

    # Create a copy of the smirnoff + tip3p offxml file
    force_field_path = build_tip3p_smirnoff_force_field()

    # Set up the state at which we want the calculations to be performed.
    thermodynamic_state = ThermodynamicState(temperature=298.15 * unit.kelvin,
                                             pressure=1.0 * unit.atmosphere)

    host_guest_substances, host_guest_orientations = get_paprika_host_guest_substance(host, guest)
    host_substance, _ = get_paprika_host_guest_substance(host, None)[0]

    substance_results = []
    for substance, orientation in zip(host_guest_substances, host_guest_orientations):

        # Create the protocol which will run the attach pull calculations
        host_guest_protocol = OpenMMPaprikaProtocol(f'host_guest-{orientation}')

        host_guest_protocol.substance = substance
        host_guest_protocol.taproom_guest_orientation = orientation

        # Set up the required directories.
        host_guest_directory = f'{host}-{guest}-{orientation}'
        os.makedirs(host_guest_directory, exist_ok=True)

        host_guest_protocol.thermodynamic_state = thermodynamic_state

        host_guest_protocol.number_of_equilibration_steps = 50
        host_guest_protocol.number_of_production_steps = 50
        host_guest_protocol.equilibration_output_frequency = 1
        host_guest_protocol.production_output_frequency = 1
        host_guest_protocol.number_of_solvent_molecules = 500

        host_guest_protocol.taproom_host_name = host
        host_guest_protocol.taproom_guest_name = guest

        host_guest_protocol.setup = True
        host_guest_protocol.simulate = True



        host_guest_protocol.force_field = OpenMMPaprikaProtocol.ForceField.SMIRNOFF
        host_guest_protocol.force_field_path = force_field_path

        result = host_guest_protocol.execute(host_guest_directory, resources)

        if isinstance(result, PropertyEstimatorException):
            logging.info(f'The attach / pull calculations failed with error: {result.message}')
            return

        substance_results.append(host_guest_protocol)

    if len(host_guest_substances) > 1:

        from propertyestimator.protocols.binding import AddBindingFreeEnergies

        sum_protocol = AddBindingFreeEnergies("add_binding_free_energies")

        free_energies = [result.attach_free_energy + result.pull_free_energy for result in substance_results]
        for result in substance_results:
            logging.info(f"Attach = {result.attach_free_energy.value.to(unit.kilocalorie / unit.mole)} ± {result.attach_free_energy.uncertainty.to(unit.kilocalorie / unit.mole)}",
                         f"Pull={result.pull_free_energy.value.to(unit.kilocalorie / unit.mole)} ± {result.pull_free_energy.uncertainty.to(unit.kilocalorie / unit.mole)}")

        logging.info(f"Combined Attach = {free_energies[0].value.to(unit.kilocalorie / unit.mole)} ± {free_energies[0].uncertainty.to(unit.kilocalorie / unit.mole)}")
        logging.info(f"Combined Pull = {free_energies[1].value.to(unit.kilocalorie / unit.mole)} ± {free_energies[1].uncertainty.to(unit.kilocalorie / unit.mole)}")

        sum_protocol.values = free_energies
        sum_protocol.thermodynamic_state = thermodynamic_state

        sum_protocol.execute('', resources)

    # Create the protocol which will run the release calculations
    host_directory = f'{host}'
    os.makedirs(host_directory, exist_ok=True)

    # Create the protocol which will run the release calculations
    host_protocol = OpenMMPaprikaProtocol('host')

    host_protocol.substance = host_substance
    host_protocol.thermodynamic_state = thermodynamic_state

    host_protocol.taproom_host_name = host
    host_protocol.taproom_name = None

    host_protocol.number_of_equilibration_steps = 50
    host_protocol.number_of_production_steps = 1
    host_protocol.equilibration_output_frequency = 25
    host_protocol.production_output_frequency = 1
    host_protocol.number_of_solvent_molecules = 2

    host_protocol.force_field = OpenMMPaprikaProtocol.ForceField.SMIRNOFF
    host_protocol.force_field_path = force_field_path

    result = host_protocol.execute(host_directory, resources)

    if isinstance(result, PropertyEstimatorException):
        logging.info(f'The release calculations failed with error: {result.message}')
        return

    logging.info(f"Attach and Pull={sum_protocol.result} "
                 f"Release={host_protocol.release_free_energy} "
                 f'Reference={host_guest_protocol.reference_free_energy}')



if __name__ == "__main__":
    main()