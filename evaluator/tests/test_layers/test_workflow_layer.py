import os
import tempfile

from evaluator import unit
from evaluator.backends.dask import DaskLocalCluster
from evaluator.client import RequestOptions
from evaluator.forcefield import SmirnoffForceFieldSource
from evaluator.layers.simulation import SimulationLayer, SimulationSchema
from evaluator.properties import Density
from evaluator.server import server
from evaluator.storage import LocalFileStorage
from evaluator.tests.test_workflow.utils import DummyInputOutputProtocol
from evaluator.tests.utils import create_dummy_property
from evaluator.utils.utils import temporarily_change_directory
from evaluator.workflow import WorkflowSchema
from evaluator.workflow.utils import ProtocolPath


def test_workflow_layer():
    """Test the `WorkflowLayer` calculation layer. As the `SimulationLayer`
    is the simplest implementation of the abstract layer, we settle for
    testing this."""

    properties_to_estimate = [
        create_dummy_property(Density),
        create_dummy_property(Density),
    ]

    # Create a very simple workflow which just returns some placeholder
    # value.
    estimated_value = (1 * unit.kelvin).plus_minus(0.1 * unit.kelvin)
    protocol_a = DummyInputOutputProtocol("protocol_a")
    protocol_a.input_value = estimated_value

    schema = WorkflowSchema()
    schema.protocol_schemas = [protocol_a.schema]
    schema.final_value_source = ProtocolPath("output_value", protocol_a.id)

    layer_schema = SimulationSchema()
    layer_schema.workflow_schema = schema

    options = RequestOptions()
    options.add_schema("SimulationLayer", "Density", layer_schema)

    batch = server.Batch()
    batch.queued_properties = properties_to_estimate
    batch.options = options

    with tempfile.TemporaryDirectory() as directory:

        with temporarily_change_directory(directory):

            # Create a directory for the layer.
            layer_directory = "simulation_layer"
            os.makedirs(layer_directory)

            # Set-up a simple storage backend and add a force field to it.
            force_field = SmirnoffForceFieldSource.from_path(
                "smirnoff99Frosst-1.1.0.offxml"
            )

            storage_backend = LocalFileStorage()
            batch.force_field_id = storage_backend.store_force_field(force_field)

            # Create a simple calculation backend to test with.
            with DaskLocalCluster() as calculation_backend:

                def dummy_callback(returned_request):

                    assert len(returned_request.estimated_properties) == 2
                    assert len(returned_request.exceptions) == 0

                simulation_layer = SimulationLayer()

                simulation_layer.schedule_calculation(
                    calculation_backend,
                    storage_backend,
                    layer_directory,
                    batch,
                    dummy_callback,
                    True,
                )
