import json
import tempfile
from os import makedirs, path

from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.client import RequestOptions
from openff.evaluator.layers import (
    CalculationLayer,
    CalculationLayerResult,
    CalculationLayerSchema,
    calculation_layer,
)
from openff.evaluator.properties import Density
from openff.evaluator.server import server
from openff.evaluator.storage import LocalFileStorage
from openff.evaluator.storage.data import StoredSimulationData
from openff.evaluator.tests.utils import create_dummy_property
from openff.evaluator.utils.exceptions import EvaluatorException
from openff.evaluator.utils.observables import ObservableFrame
from openff.evaluator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder
from openff.evaluator.utils.utils import temporarily_change_directory


@calculation_layer()
class DummyCalculationLayer(CalculationLayer):
    """A dummy calculation layer class to test out the base
    calculation layer methods.
    """

    @classmethod
    def required_schema_type(cls):
        return CalculationLayerSchema

    @classmethod
    def _schedule_calculation(
        cls, calculation_backend, storage_backend, layer_directory, batch
    ):
        futures = [
            # Fake a success.
            calculation_backend.submit_task(
                DummyCalculationLayer.process_successful_property,
                batch.queued_properties[0],
                layer_directory,
            ),
            # Fake a failure.
            calculation_backend.submit_task(
                DummyCalculationLayer.process_failed_property,
                batch.queued_properties[1],
            ),
            # Cause an exception.
            calculation_backend.submit_task(
                DummyCalculationLayer.return_bad_result,
                batch.queued_properties[0],
                layer_directory,
            ),
        ]

        return futures

    @staticmethod
    def process_successful_property(physical_property, layer_directory, **_):
        """Return a result as if the property had been successfully estimated."""

        dummy_data_directory = path.join(layer_directory, "good_dummy_data")
        makedirs(dummy_data_directory, exist_ok=True)

        dummy_stored_object = StoredSimulationData()
        dummy_stored_object.substance = physical_property.substance
        dummy_stored_object.thermodynamic_state = physical_property.thermodynamic_state
        dummy_stored_object.property_phase = physical_property.phase
        dummy_stored_object.force_field_id = ""
        dummy_stored_object.coordinate_file_name = ""
        dummy_stored_object.trajectory_file_name = ""
        dummy_stored_object.observables = ObservableFrame()
        dummy_stored_object.statistical_inefficiency = 1.0
        dummy_stored_object.number_of_molecules = 10
        dummy_stored_object.source_calculation_id = ""

        dummy_stored_object_path = path.join(layer_directory, "good_dummy_data.json")

        with open(dummy_stored_object_path, "w") as file:
            json.dump(dummy_stored_object, file, cls=TypedJSONEncoder)

        return_object = CalculationLayerResult()
        return_object.physical_property = physical_property
        return_object.data_to_store = [(dummy_stored_object_path, dummy_data_directory)]

        return return_object

    @staticmethod
    def process_failed_property(physical_property, **_):
        """Return a result as if the property could not be estimated."""

        return_object = CalculationLayerResult()
        return_object.physical_property = physical_property
        return_object.exceptions = [EvaluatorException(message="Failure Message")]

        return return_object

    @staticmethod
    def return_bad_result(physical_property, layer_directory, **_):
        """Return a result which leads to an unhandled exception."""

        dummy_data_directory = path.join(layer_directory, "bad_dummy_data")
        makedirs(dummy_data_directory, exist_ok=True)

        dummy_stored_object = StoredSimulationData()
        dummy_stored_object_path = path.join(layer_directory, "bad_dummy_data.json")

        with open(dummy_stored_object_path, "w") as file:
            json.dump(dummy_stored_object, file, cls=TypedJSONEncoder)

        return_object = CalculationLayerResult()
        return_object.physical_property = physical_property
        return_object.data_to_store = [(dummy_stored_object_path, dummy_data_directory)]

        return return_object


def test_base_layer():
    properties_to_estimate = [
        create_dummy_property(Density),
        create_dummy_property(Density),
    ]

    dummy_options = RequestOptions()

    batch = server.Batch()
    batch.queued_properties = properties_to_estimate
    batch.options = dummy_options
    batch.force_field_id = ""
    batch.options.calculation_schemas = {
        "Density": {"DummyCalculationLayer": CalculationLayerSchema()}
    }

    with tempfile.TemporaryDirectory() as temporary_directory:
        with temporarily_change_directory(temporary_directory):
            # Create a simple calculation backend to test with.
            test_backend = DaskLocalCluster()
            test_backend.start()

            # Create a simple storage backend to test with.
            test_storage = LocalFileStorage()

            layer_directory = "dummy_layer"
            makedirs(layer_directory)

            def dummy_callback(returned_request):
                assert len(returned_request.estimated_properties) == 1
                assert len(returned_request.exceptions) == 2

            dummy_layer = DummyCalculationLayer()

            dummy_layer.schedule_calculation(
                test_backend,
                test_storage,
                layer_directory,
                batch,
                dummy_callback,
                True,
            )


def test_serialize_layer_result():
    """Tests that the `CalculationLayerResult` can be properly
    serialized and deserialized."""

    dummy_result = CalculationLayerResult()

    dummy_result.physical_property = create_dummy_property(Density)
    dummy_result.exceptions = [EvaluatorException()]

    dummy_result.data_to_store = [("dummy_object_path", "dummy_directory")]

    dummy_result_json = json.dumps(dummy_result, cls=TypedJSONEncoder)

    recreated_result = json.loads(dummy_result_json, cls=TypedJSONDecoder)
    recreated_result_json = json.dumps(recreated_result, cls=TypedJSONEncoder)

    assert recreated_result_json == dummy_result_json
