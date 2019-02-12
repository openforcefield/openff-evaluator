#!/usr/bin/env python

import logging

from propertyestimator.datasets import ThermoMLDataSet
from propertyestimator import client
from openforcefield.typing.engines import smirnoff
from propertyestimator.utils import get_data_filename, setup_timestamp_logging


def compute_estimate_sync():
    """Submit calculations to a running server instance"""
    setup_timestamp_logging()

    # Load in the data set of interest.
    data_set = ThermoMLDataSet.from_file(get_data_filename('properties/single_density.xml'))
    # Load in the force field to use.
    force_field = smirnoff.ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))

    # Create the client object.
    property_estimator = client.PropertyEstimatorClient()
    # Submit the request to a running server, and wait for the results.
    result = property_estimator.estimate(data_set, force_field)

    logging.info('The server has returned a response: {}'.format(result))


def compute_estimate_async():
    """Submit calculations to a running server instance"""
    setup_timestamp_logging()

    # Load in the data set of interest.
    data_set = ThermoMLDataSet.from_file(get_data_filename('properties/single_density.xml'))
    # Load in the force field to use.
    force_field = smirnoff.ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))

    # Create the client object.
    property_estimator = client.PropertyEstimatorClient()
    # Submit the request to a running server.
    ticket_id = property_estimator.estimate(data_set, force_field)

    logging.info('Ticket info: {}'.format(ticket_id))
    # Wait for the results.
    result = property_estimator.retrieve_estimate(ticket_id, synchronous=True)

    logging.info('The server has returned a response: {}'.format(result))


if __name__ == "__main__":
    compute_estimate_async()
