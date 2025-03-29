import os
import pathlib
import shutil

import pytest
from openff.utilities.utilities import get_data_dir_path, temporary_cd

from openff.evaluator._tests.test_full_workflows.test_equilibration import (
    _generate_error_tolerances,
)
from openff.evaluator._tests.utils import (
    _copy_property_working_data,
    _write_force_field,
)
from openff.evaluator.attributes.attributes import UNDEFINED
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.client import BatchMode, EvaluatorClient, RequestOptions
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.layers.preequilibrated_simulation import (
    PreequilibratedSimulationLayer,
)
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.server.server import Batch, EvaluatorServer
from openff.evaluator.storage.localfile import LocalFileStorage
from openff.evaluator.workflow import Workflow


def _get_preequilibrated_simulation_request_options(
    n_molecules: int = 256,
):
    dhmix_schema = EnthalpyOfMixing.default_preequilibrated_simulation_schema(
        n_molecules=n_molecules,
    )
    density_schema = Density.default_preequilibrated_simulation_schema(
        n_molecules=n_molecules,
    )

    options = RequestOptions()
    options.calculation_layers = ["PreequilibratedSimulationLayer"]

    options.add_schema(
        "PreequilibratedSimulationLayer",
        "Density",
        density_schema,
    )
    options.add_schema(
        "PreequilibratedSimulationLayer",
        "EnthalpyOfMixing",
        dhmix_schema,
    )
    return options


class TestPreequilibratedSimulationLayer:

    @pytest.fixture
    def dhmix_density_CCCO(self, tmp_path_factory):
        data_directory = pathlib.Path(
            get_data_dir_path(
                "test/workflows/preequilibrated_simulation/dhmix-density-CCCO/stored_data",
                "openff.evaluator",
            )
        )
        abs_path = data_directory.resolve()
        path = tmp_path_factory.mktemp("dhmix-density-CCCO")
        path.mkdir(exist_ok=True, parents=True)
        shutil.copytree(abs_path, path / "stored_data")
        return path

    def _base_test_metadata(self, dummy_enthalpy_of_mixing, directory_name):
        """
        Base test metadata generation
        """
        error_tolerances = _generate_error_tolerances()
        schema = EnthalpyOfMixing.default_preequilibrated_simulation_schema(
            n_molecules=256,
            relative_tolerance=0.2,
            equilibration_error_tolerances=error_tolerances,
        )

        metadata = PreequilibratedSimulationLayer._get_workflow_metadata(
            ".",
            dummy_enthalpy_of_mixing,
            "force-field.json",
            [],
            LocalFileStorage(root_directory=directory_name),
            schema,
        )

        assert len(metadata) == 11
        assert (
            metadata["thermodynamic_state"]
            == dummy_enthalpy_of_mixing.thermodynamic_state
        )
        assert metadata["substance"] == dummy_enthalpy_of_mixing.substance
        assert metadata["force_field_path"] == "force-field.json"
        assert metadata["equilibration_error_tolerances"] == error_tolerances
        assert metadata["equilibration_error_aggregration"].value == "All"
        return metadata

    def test_metadata_generation_from_stored(
        self, dummy_enthalpy_of_mixing, dhmix_density_CCCO
    ):
        """Test that query paths are found from saved data"""
        os.chdir(dhmix_density_CCCO)

        metadata = self._base_test_metadata(dummy_enthalpy_of_mixing, "stored_data")

        assert metadata["full_system_data"] == (
            "./bbd8a8682e94467481db13bc57ed5093",
            "stored_data/bbd8a8682e94467481db13bc57ed5093",
            "force-field.json",
        )
        assert metadata["component_data"] == [
            (
                "./b3d2d2e4fa734d4ab45ef929a93fa35e",
                "stored_data/b3d2d2e4fa734d4ab45ef929a93fa35e",
                "force-field.json",
            ),
            (
                "./f39842bc8be24b04b364f413fa5ea250",
                "stored_data/f39842bc8be24b04b364f413fa5ea250",
                "force-field.json",
            ),
        ]

    def test_metadata_generation_from_empty(self, dummy_enthalpy_of_mixing):
        """Test that query paths are not found from empty data"""
        with temporary_cd():
            metadata = self._base_test_metadata(dummy_enthalpy_of_mixing, "empty")
            assert metadata["full_system_data"] == []
            assert metadata["component_data"] == [[], []]

    def test_preequilibrated_simulation_execution(
        self, dummy_enthalpy_of_mixing, dhmix_density_CCCO
    ):
        """
        Test direct execution of an EnthalpyOfMixing protocol
        """
        os.chdir(dhmix_density_CCCO)

        # set up equilibration options
        _write_force_field()
        schema = EnthalpyOfMixing.default_preequilibrated_simulation_schema(
            n_molecules=256,
            equilibration_error_tolerances=_generate_error_tolerances(),
            n_uncorrelated_samples=100,
        )

        metadata = Workflow.generate_default_metadata(
            dummy_enthalpy_of_mixing, "force-field.json"
        )
        # have to update with additional metadata
        metadata.update(
            PreequilibratedSimulationLayer._get_workflow_metadata(
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
        _copy_property_working_data(
            "test/workflows/preequilibrated_simulation/dhmix-density-CCCO",
            uuid_prefix="1",
            destination_directory=".",
        )

        for name, protocol in workflow_graph.protocols.items():
            path = name.replace("|", "_")
            if "conditional" not in name or "equilibration" in name:
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
            if "equilibration" not in str(file):
                file.unlink()

        for name, protocol in workflow_graph.protocols.items():
            path = name.replace("|", "_")
            if "conditional" in name and "equilibration" not in name:
                output = protocol_graph._execute_protocol(
                    path,
                    protocol,
                    True,
                    *previous_output_paths,
                    available_resources=None,
                    safe_exceptions=False,
                )
                previous_output_paths.append(output)

    def test_preequilibrated_simulation_with_server(
        self, dummy_dataset, dhmix_density_CCCO
    ):
        """
        Test the full workflow with a server
        """
        force_field_path = "openff-2.1.0.offxml"
        force_field_source = SmirnoffForceFieldSource.from_path(force_field_path)

        os.chdir(dhmix_density_CCCO)
        options = _get_preequilibrated_simulation_request_options(
            n_molecules=256,
        )
        options.batch_mode = BatchMode.NoBatch

        batch_path = pathlib.Path("PreequilibratedSimulationLayer") / "batch_0000"
        batch_path.mkdir(parents=True, exist_ok=True)
        _copy_property_working_data(
            "test/workflows/preequilibrated_simulation/dhmix-density-CCCO",
            uuid_prefix="0",
            destination_directory=batch_path,
        )
        _copy_property_working_data(
            "test/workflows/preequilibrated_simulation/dhmix-density-CCCO",
            uuid_prefix="1",
            destination_directory=batch_path,
        )

        # copy data to current directory as well due to messing with paths
        _copy_property_working_data(
            "test/workflows/preequilibrated_simulation/dhmix-density-CCCO",
            uuid_prefix="0",
            destination_directory=".",
        )

        _copy_property_working_data(
            "test/workflows/preequilibrated_simulation/dhmix-density-CCCO",
            uuid_prefix="1",
            destination_directory=".",
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
                assert len(results.exceptions) == 0, results.exceptions[0].message
                assert len(results.estimated_properties) == 2
                assert len(results.unsuccessful_properties) == 0
