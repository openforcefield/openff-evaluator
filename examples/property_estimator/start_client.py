#!/usr/bin/env python

import logging

from propertyestimator.client import PropertyEstimatorOptions
from propertyestimator.datasets import ThermoMLDataSet, PhysicalPropertyDataSet
from propertyestimator import client
from openforcefield.typing.engines import smirnoff
from propertyestimator.utils import get_data_filename, setup_timestamp_logging


def submit_calculation_to_server():
    """Submit calculations to a running server instance"""

    setup_timestamp_logging()

    # data_set = ThermoMLDataSet.from_file_list(get_data_filename('properties/single_density.xml'))

    complete_data_set = ThermoMLDataSet.from_file_list(get_data_filename('properties/JCT/j.jct.2008.12.004.xml'))

    data_set = PhysicalPropertyDataSet()

    data_set.properties['[CH3].C1CCCCC1{0.074}|[H].CCCCCC{0.926}'] = \
        complete_data_set.properties['[CH3].C1CCCCC1{0.074}|[H].CCCCCC{0.926}']

    force_field = smirnoff.ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))

    options = PropertyEstimatorOptions()
    options.relative_uncertainty = 10000.0

    property_estimator = client.PropertyEstimator()
    ticket_ids = property_estimator.submit_computations(data_set, force_field, options)

    logging.info('Ticket info: {}'.format(ticket_ids))
    result = property_estimator.wait_for_result(ticket_ids)

    logging.info('The server has returned a response: {}'.format(result.json()))


if __name__ == "__main__":
    submit_calculation_to_server()
