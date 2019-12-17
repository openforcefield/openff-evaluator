"""
Units tests for propertyestimator.client and server
"""
import tempfile
from os import path

from propertyestimator.backends import ComputeResources, DaskLocalCluster
from propertyestimator.client import EvaluatorClient, RequestOptions
from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.forcefield import SmirnoffForceFieldSource
from propertyestimator.layers import (
    CalculationLayer,
    CalculationLayerSchema,
    calculation_layer,
)
from propertyestimator.properties import Density
from propertyestimator.server.server import EvaluatorServer
from propertyestimator.storage import LocalFileStorage
from propertyestimator.tests.utils import create_dummy_property
from propertyestimator.utils.exceptions import PropertyEstimatorException


@calculation_layer()
class TestCalculationLayer(CalculationLayer):
    """A calculation layer which marks properties to be calculated
    as finished for the purpose of testing.
    """

    @classmethod
    def required_schema_type(cls):
        return CalculationLayerSchema

    @classmethod
    def schedule_calculation(
        cls,
        calculation_backend,
        storage_backend,
        layer_directory,
        data_model,
        callback,
        synchronous=False,
    ):

        for physical_property in data_model.queued_properties:

            substance_id = physical_property.substance.identifier

            if substance_id not in data_model.estimated_properties:
                data_model.estimated_properties[substance_id] = []

            data_model.estimated_properties[substance_id].append(physical_property)

        data_model.queued_properties = []
        callback(data_model)


def test_estimate_request():
    """Test sending an estimator request to a server."""

    with tempfile.TemporaryDirectory() as temporary_directory:

        storage_directory = path.join(temporary_directory, "storage")
        working_directory = path.join(temporary_directory, "working")

        dummy_property = create_dummy_property(Density)

        dummy_data_set = PhysicalPropertyDataSet()
        dummy_data_set.properties[dummy_property.substance.identifier] = [
            dummy_property
        ]

        force_field_source = SmirnoffForceFieldSource.from_path(
            "smirnoff99Frosst-1.1.0.offxml"
        )

        calculation_backend = DaskLocalCluster(1, ComputeResources())
        storage_backend = LocalFileStorage(storage_directory)

        EvaluatorServer(
            calculation_backend, storage_backend, working_directory=working_directory
        )

        property_estimator = EvaluatorClient()
        options = RequestOptions(allowed_calculation_layers=[TestCalculationLayer])

        request = property_estimator.request_estimate(
            dummy_data_set, force_field_source, options
        )
        result = request.results(synchronous=True, polling_interval=0)

        assert not isinstance(result, PropertyEstimatorException)
