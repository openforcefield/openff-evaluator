#!/usr/bin/env python

import shutil
from os import path

from propertyestimator import server
from propertyestimator.backends import DaskLocalClusterBackend
from propertyestimator.storage import LocalFileStorage
from propertyestimator.utils import setup_timestamp_logging


def start_property_estimator_server():
    """An integrated test of the property estimator"""

    setup_timestamp_logging()

    # Set the name of the directory in which all temporary files
    # will be generated.
    working_directory = 'working_directory'

    # Remove any existing data.
    if path.isdir(working_directory):
        shutil.rmtree(working_directory)

    # Create a calculation backend to perform workflow
    # calculations on.
    calculation_backend = DaskLocalClusterBackend(1)
    # Create a backend to handle storing and retrieving
    # cached simulation data.
    storage_backend = LocalFileStorage()

    # Create a server instance.
    property_server = server.PropertyEstimatorServer(calculation_backend,
                                                       storage_backend,
                                                       working_directory=working_directory)

    # Tell the server to start listening for incoming
    # estimation requests.
    property_server.start_listening_loop()


if __name__ == "__main__":
    start_property_estimator_server()
