"""
Units tests for the propertyestimator.server module.
"""
import tempfile
from time import sleep

from propertyestimator.backends import DaskLocalCluster
from propertyestimator.server.server import EvaluatorServer


def test_server_spin_up():

    with tempfile.TemporaryDirectory() as directory:

        calculation_backend = DaskLocalCluster()

        with calculation_backend:

            server = EvaluatorServer(
                calculation_backend=calculation_backend, working_directory=directory,
            )

            with server:
                sleep(0.5)
