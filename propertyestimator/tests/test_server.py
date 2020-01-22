"""
Units tests for the propertyestimator.server module.
"""
import tempfile
from time import sleep

from propertyestimator.backends import DaskLocalCluster
from propertyestimator.client import RequestOptions
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.layers import (
    CalculationLayer,
    CalculationLayerSchema,
    calculation_layer,
)
from propertyestimator.layers.layers import CalculationLayerResult
from propertyestimator.properties import Density
from propertyestimator.server.server import EvaluatorServer, _Batch
from propertyestimator.tests.utils import create_dummy_property
from propertyestimator.utils.utils import temporarily_change_directory


@calculation_layer()
class QuickCalculationLayer(CalculationLayer):
    """A dummy calculation layer class to test out the base
    calculation layer methods.
    """

    @classmethod
    def required_schema_type(cls):
        return CalculationLayerSchema

    @classmethod
    def _schedule_calculation(
        cls,
        calculation_backend,
        storage_backend,
        layer_directory,
        batch
    ):

        futures = [
            calculation_backend.submit_task(
                QuickCalculationLayer.process_property, batch.queued_properties[0],
            ),
        ]

        return futures

    @staticmethod
    def process_property(physical_property, **_):
        """Return a result as if the property had been successfully estimated.
        """
        return_object = CalculationLayerResult()
        return_object.physical_property = physical_property
        return_object.calculated_property = physical_property

        return return_object


def test_server_spin_up():

    with tempfile.TemporaryDirectory() as directory:

        with temporarily_change_directory(directory):

            with DaskLocalCluster() as calculation_backend:

                server = EvaluatorServer(
                    calculation_backend=calculation_backend,
                    working_directory=directory,
                )

                with server:
                    sleep(0.5)


def test_launch_batch():

    # Set up a dummy data set
    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(
        create_dummy_property(Density), create_dummy_property(Density)
    )

    batch = _Batch()
    batch.force_field_id = ""
    batch.options = RequestOptions()
    batch.options.calculation_layers = ["QuickCalculationLayer"]
    batch.options.calculation_schemas = {
        "Density": {"QuickCalculationLayer": CalculationLayerSchema()}
    }
    batch.parameter_gradient_keys = []
    batch.queued_properties = next(iter(data_set.properties.values()))
    batch.validate()

    with tempfile.TemporaryDirectory() as directory:

        with temporarily_change_directory(directory):

            with DaskLocalCluster() as calculation_backend:

                server = EvaluatorServer(
                    calculation_backend=calculation_backend,
                    working_directory=directory,
                )

                server._queued_batches[batch.id] = batch
                server._launch_batch(batch)

                while len(batch.queued_properties) > 0:
                    sleep(0.01)

                assert len(batch.estimated_properties) == 1
                assert len(batch.unsuccessful_properties) == 1
