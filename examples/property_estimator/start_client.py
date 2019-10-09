#!/usr/bin/env python
import logging

from propertyestimator import client
from propertyestimator.datasets import ThermoMLDataSet
from propertyestimator.forcefield import SmirnoffForceFieldSource
from propertyestimator.utils import get_data_filename, setup_timestamp_logging


def main():
    """Submit calculations to a running server instance"""
    from openforcefield.typing.engines import smirnoff

    setup_timestamp_logging()

    # Load in the data set of interest.
    data_set = ThermoMLDataSet.from_file(get_data_filename('properties/single_density.xml'))

    # Load in the force field to use.
    smirnoff_force_field = smirnoff.ForceField('smirnoff99Frosst-1.1.0.offxml')
    force_field_source = SmirnoffForceFieldSource.from_object(smirnoff_force_field)

    # Create the client object.
    property_estimator = client.PropertyEstimatorClient()
    # Submit the request to a running server.
    request = property_estimator.request_estimate(data_set, force_field_source)

    # Wait for the results.
    results = request.results(True, 5)
    logging.info('The server has returned a response: {}'.format(results))


if __name__ == "__main__":
    main()
