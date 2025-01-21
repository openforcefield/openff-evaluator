import os
import pathlib
import shutil

import pytest
from openff.utilities.utilities import get_data_dir_path

from openff.evaluator._tests.utils import (
    _copy_property_working_data,
    _write_force_field,
)
from openff.evaluator.attributes.attributes import UNDEFINED
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.client import BatchMode, EvaluatorClient, RequestOptions
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.layers.simulation import SimulationLayer
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.server.server import Batch, EvaluatorServer
from openff.evaluator.storage.localfile import LocalFileStorage
from openff.evaluator.workflow import Workflow


def _get_simulation_request_options(
    n_molecules: int = 256,
):
    dhmix_schema = EnthalpyOfMixing.default_simulation_schema(
        n_molecules=n_molecules,
        relative_tolerance=0.2,
    )
    density_schema = Density.default_simulation_schema(
        n_molecules=n_molecules,
        relative_tolerance=0.2,
    )

    options = RequestOptions()
    options.calculation_layers = ["SimulationLayer"]

    options.add_schema(
        "SimulationLayer",
        "Density",
        density_schema,
    )
    options.add_schema(
        "SimulationLayer",
        "EnthalpyOfMixing",
        dhmix_schema,
    )
    return options


class TestSimulationLayer:

    @pytest.fixture
    def dhmix_density_CCCO(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("dhmix-density-CCCO")
        path.mkdir(exist_ok=True, parents=True)

        _copy_property_working_data(
            "test/workflows/simulation/dhmix-density-CCCO",
            uuid_prefix=["0", "1"],
            destination_directory=path,
        )
        return path

    def test_simulation(self, dummy_enthalpy_of_mixing, dhmix_density_CCCO):
        """
        Test direct execution of an EnthalpyOfMixing protocol
        """
        os.chdir(dhmix_density_CCCO)

        _write_force_field()
        schema = EnthalpyOfMixing.default_simulation_schema(
            n_molecules=256,
            relative_tolerance=0.2,
        )

        metadata = Workflow.generate_default_metadata(
            dummy_enthalpy_of_mixing, "force-field.json"
        )
        metadata.update(
            SimulationLayer._get_workflow_metadata(
                ".",
                dummy_enthalpy_of_mixing,
                "force-field.json",
                [],
                LocalFileStorage(),
                schema,
            )
        )

        workflow_schema = schema.workflow_schema
        workflow_schema.replace_protocol_types(
            {"BaseBuildSystem": "BuildSmirnoffSystem"}
        )
        uuid_prefix = "1"
        workflow = Workflow.from_schema(
            workflow_schema, metadata=metadata, unique_id=uuid_prefix
        )
        workflow_graph = workflow.to_graph()
        protocol_graph = workflow_graph._protocol_graph

        previous_output_paths = []

        for name, protocol in workflow_graph.protocols.items():
            path = name.replace("|", "_")
            if "conditional" not in name:
                output = protocol_graph._execute_protocol(
                    path,
                    protocol,
                    True,
                    *previous_output_paths,
                    available_resources=None,
                    safe_exceptions=False,
                )
                previous_output_paths.append(output)

        # delete existing output so we can re-create it
        pattern = "*conditional*/*conditional*output.json"
        for file in pathlib.Path(".").rglob(pattern):
            file.unlink()

        for name, protocol in workflow_graph.protocols.items():
            path = name.replace("|", "_")
            if "conditional" in name:
                output = protocol_graph._execute_protocol(
                    path,
                    protocol,
                    True,
                    *previous_output_paths,
                    available_resources=None,
                    safe_exceptions=False,
                )
                previous_output_paths.append(output)

    def test_simulation_with_server(self, dummy_dataset, dhmix_density_CCCO):
        """
        Test the full workflow with a server
        """
        force_field_path = "openff-2.1.0.offxml"
        force_field_source = SmirnoffForceFieldSource.from_path(force_field_path)

        os.chdir(dhmix_density_CCCO)
        options = _get_simulation_request_options(
            n_molecules=256,
        )
        options.batch_mode = BatchMode.NoBatch

        batch_path = pathlib.Path("SimulationLayer") / "batch_0000"
        batch_path.mkdir(parents=True, exist_ok=True)
        _copy_property_working_data(
            "test/workflows/simulation/dhmix-density-CCCO",
            uuid_prefix="0",
            destination_directory=batch_path,
        )
        _copy_property_working_data(
            "test/workflows/simulation/dhmix-density-CCCO",
            uuid_prefix="1",
            destination_directory=batch_path,
        )

        with DaskLocalCluster(number_of_workers=1) as calculation_backend:
            server = EvaluatorServer(
                calculation_backend=calculation_backend,
                working_directory=".",
                delete_working_files=False,
            )
            with server:
                client = EvaluatorClient()
                request, error = client.request_estimate(
                    dummy_dataset, force_field_source, options
                )

                assert error is None
                results, exception = request.results(
                    synchronous=True, polling_interval=30
                )

                assert exception is None
                assert len(results.queued_properties) == 0
                assert len(results.estimated_properties) == 2
                assert len(results.unsuccessful_properties) == 0
                assert len(results.exceptions) == 0
