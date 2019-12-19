"""
Units tests for the propertyestimator.server module.
"""
import tempfile
from time import sleep

from propertyestimator.backends import DaskLocalCluster
from propertyestimator.server.server import EvaluatorServer
from propertyestimator.storage import LocalFileStorage


def test_server_spin_up():

    with tempfile.TemporaryDirectory() as directory:

        calculation_backend = DaskLocalCluster()
        calculation_backend.start()

        storage_backend = LocalFileStorage(directory)

        server = EvaluatorServer(
            calculation_backend=calculation_backend,
            storage_backend=storage_backend,
            working_directory=directory,
        )

        server.start(asynchronous=True)
        sleep(0.5)
        server.stop()
