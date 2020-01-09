#!/usr/bin/env python
import logging

from propertyestimator import client
from propertyestimator.datasets.thermoml import ThermoMLDataSet
from propertyestimator.utils import get_data_filename, setup_timestamp_logging

logger = logging.getLogger(__name__)


def main():
    """Submit calculations to a running server instance"""
    from openforcefield.typing.engines import smirnoff

    setup_timestamp_logging()

    # Load in the data set of interest.
    data_set = ThermoMLDataSet.from_file(
        get_data_filename("properties/single_density.xml")
    )

    # Load in the force field to use.
    force_field = smirnoff.ForceField("smirnoff99Frosst-1.1.0.offxml")

    # Create the client object.
    property_estimator = client.EvaluatorClient()
    # Submit the request to a running server.
    request = property_estimator.request_estimate(data_set, force_field)

    # Wait for the results.
    results, error = request.results(True, 60)

    if error is not None:

        logger.info(f"The server failed to complete the request:")
        logger.info(f"Message: {error.message}")
        return

    logger.info(f"The server has completed the request.")

    # Save the response
    results.json("results.json")


if __name__ == "__main__":
    main()
