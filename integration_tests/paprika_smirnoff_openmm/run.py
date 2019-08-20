#!/usr/bin/env python
import logging
import os as os

from simtk import unit

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
    resources = ComputeResources(number_of_threads=4, number_of_gpus=4,
                                 preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA)

    # Create a copy of the smirnoff + tip3p offxml file
    force_field_path = build_tip3p_smirnoff_force_field()

    # Set up the state at which we want the calculations to be performed.
    thermodynamic_state = ThermodynamicState(temperature=298.15 * unit.kelvin,
                                             pressure=1.0 * unit.atmosphere)

    # Set up the substance definitions.
    host_guest_substance = get_paprika_host_guest_substance(host, guest)
    host_substance = get_paprika_host_guest_substance(host, None)

    # Set up the required directories.
    host_guest_directory = 'paprika_attach_pull'
    os.makedirs(host_guest_directory, exist_ok=True)

    host_directory = 'paprika_release'
    os.makedirs(host_directory, exist_ok=True)

    # Create the protocol which will run the attach pull calculations
    host_guest_protocol = OpenMMPaprikaProtocol('host_guest')

    host_guest_protocol.substance = host_guest_substance
    host_guest_protocol.thermodynamic_state = thermodynamic_state

    host_guest_protocol.taproom_host_name = host
    host_guest_protocol.taproom_guest_name = guest

    host_guest_protocol.force_field = OpenMMPaprikaProtocol.ForceField.SMIRNOFF
    host_guest_protocol.force_field_path = force_field_path

    result = host_guest_protocol.execute(host_guest_directory, resources)

    if isinstance(result, PropertyEstimatorException):
        logging.info(f'The attach / pull calculations failed with error: {result.message}')
        return

    # Create the protocol which will run the release calculations
    host_protocol = OpenMMPaprikaProtocol('host')

    host_protocol.substance = host_substance
    host_protocol.thermodynamic_state = thermodynamic_state

    host_protocol.taproom_host_name = host
    host_protocol.taproom_name = None

    host_protocol.force_field = OpenMMPaprikaProtocol.ForceField.SMIRNOFF
    host_protocol.force_field_path = force_field_path

    result = host_protocol.execute(host_directory, resources)

    if isinstance(result, PropertyEstimatorException):
        logging.info(f'The release calculations failed with error: {result.message}')
        return

    logging.info(f'Attach={host_guest_protocol.attach_free_energy} '
                 f'Pull={host_guest_protocol.pull_free_energy} '
                 f'Release={host_protocol.release_free_energy} '
                 f'Reference={host_guest_protocol.reference_free_energy}')


if __name__ == "__main__":
    main()
