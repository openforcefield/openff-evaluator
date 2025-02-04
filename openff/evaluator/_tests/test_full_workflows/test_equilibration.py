import os
import pathlib
import pytest

from openff.units import unit
from openff.utilities.utilities import get_data_dir_path, temporary_cd

from openff.evaluator.datasets import (
    MeasurementSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.storage.localfile import LocalFileStorage
from openff.evaluator.utils.observables import ObservableType
from openff.evaluator.backends import ComputeResources
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.properties import Density, EnthalpyOfMixing
from openff.evaluator.server.server import Batch, EvaluatorServer
from openff.evaluator.substances import Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.layers.equilibration import EquilibrationProperty, EquilibrationLayer
from openff.evaluator.client import EvaluatorClient, RequestOptions, BatchMode
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.workflow.attributes import ConditionAggregationBehavior
from openff.evaluator.storage.query import EquilibrationDataQuery
from openff.evaluator._tests.utils import _write_force_field, _copy_property_working_data

from openff.evaluator.workflow import Workflow



def _get_equilibration_request_options(
    n_molecules: int = 256,
    error_tolerances: list = [],
    condition_aggregation_behavior = ConditionAggregationBehavior.All,
    n_iterations: int = 0,

):
    dhmix_equilibration_schema = EnthalpyOfMixing.default_equilibration_schema(
        n_molecules=n_molecules,
        error_tolerances=error_tolerances,
        condition_aggregation_behavior=condition_aggregation_behavior,
        max_iterations=n_iterations
    )
    density_equilibration_schema = Density.default_equilibration_schema(
        n_molecules=n_molecules,
        error_tolerances=error_tolerances,
        condition_aggregation_behavior=condition_aggregation_behavior,
        max_iterations=n_iterations
    )

    equilibration_options = RequestOptions()
    equilibration_options.calculation_layers = ["EquilibrationLayer"]

    equilibration_options.add_schema(
        "EquilibrationLayer",
        "Density",
        density_equilibration_schema,
    )
    equilibration_options.add_schema(
        "EquilibrationLayer",
        "EnthalpyOfMixing",
        dhmix_equilibration_schema,
    )
    return equilibration_options


def _create_equilibration_data_query(
    substance,
    n_molecules: int = 256,
):
    query = EquilibrationDataQuery()
    query.number_of_molecules = n_molecules
    query.max_number_of_molecules = n_molecules
    query.calculation_layer = "EquilibrationLayer"
    query.substance = substance
    return query
    


class TestEquilibrationLayer:

    @pytest.fixture
    def dhmix_density_CCCO(self, tmp_path_factory):
        path = tmp_path_factory.mktemp("dhmix-density-CCCO")
        _copy_property_working_data(
            source_directory="test/workflows/equilibration/dhmix-density-CCCO",
            uuid_prefix="1",
            destination_directory=path,
        )
        return path

    @pytest.fixture
    def force_field_source(self):
        force_field_source = SmirnoffForceFieldSource.from_path(
            "openff-2.1.0.offxml"
        )
        return force_field_source

    @staticmethod
    def _generate_error_tolerances(
        absolute_potential_tolerance: float = 200,
        relative_density_tolerance: float = 0.2,
    ):
        abstol = absolute_potential_tolerance * unit.kilojoules_per_mole
        errors = [
            EquilibrationProperty(
                absolute_tolerance=abstol,
                observable_type=ObservableType.PotentialEnergy,
            ),
            EquilibrationProperty(
                relative_tolerance=relative_density_tolerance,
                observable_type=ObservableType.Density,
            ),
        ]
        return errors

    @pytest.mark.parametrize("potential_error, density_error, aggregation_behavior, error_on_nonconvergence, success", [
        # passes because both conditions are met
        (200, 0.2, ConditionAggregationBehavior.All, True, True),
        # passes because at least one condition is met
        (200, 0.2, ConditionAggregationBehavior.Any, True, True),
        # fails because density error is too high
        (200, 0.00002, ConditionAggregationBehavior.All, True, False),
        # passes because at least one condition is met
        (200, 0.00002, ConditionAggregationBehavior.Any, True, True),
        # fails because one condition is not met
        (0.0001, 0.2, ConditionAggregationBehavior.All, True, False),
        # passes because at least one condition is met
        (0.0001, 0.2, ConditionAggregationBehavior.Any, True, True),
        # fails because both conditions are not met
        (0.0001, 0.00001, ConditionAggregationBehavior.All, True, False),
        (0.0001, 0.00001, ConditionAggregationBehavior.Any, True, False),

        # all the above but they all pass because there is no error on nonconvergence
        (200, 0.2, ConditionAggregationBehavior.All, False, True),
        (200, 0.2, ConditionAggregationBehavior.Any, False, True),
        (200, 0.00002, ConditionAggregationBehavior.All, False, True),
        (200, 0.00002, ConditionAggregationBehavior.Any, False, True),
        (0.0001, 0.2, ConditionAggregationBehavior.All, False, True),
        (0.0001, 0.2, ConditionAggregationBehavior.Any, False, True),
        (0.0001, 0.00001, ConditionAggregationBehavior.All, False, True),
        (0.0001, 0.00001, ConditionAggregationBehavior.Any, False, True),
    ])
    def test_execute_conditions(
        self,
        potential_error,
        density_error,
        aggregation_behavior,
        error_on_nonconvergence,
        success,
        dummy_enthalpy_of_mixing,
        dhmix_density_CCCO,
    ):
        """
        Test the execution of the equilibration layer with different error tolerances.

        We test two convergence error types:
            - Potential energy error (absolute)
            - Density error (relative)

        We test two aggregation behaviors:
            - any
            - all

        We test allowing the workflow to fail and continue on nonconvergence.
        """
        errors = self._generate_error_tolerances(
            potential_error,
            density_error
        )
        os.chdir(dhmix_density_CCCO)
        _write_force_field()

        schema = EnthalpyOfMixing.default_equilibration_schema(
            n_molecules=256,
            error_tolerances=errors,
            condition_aggregation_behavior=aggregation_behavior,
            error_on_failure=error_on_nonconvergence,
            max_iterations=0
        )
        storage_backend = LocalFileStorage()
        metadata = EquilibrationLayer._get_workflow_metadata(
            ".",
            dummy_enthalpy_of_mixing,
            "force-field.json",
            [],
            storage_backend,
            schema
        )
        uuid_prefix = "1"

        
        workflow_schema = schema.workflow_schema
        workflow_schema.replace_protocol_types(
            {"BaseBuildSystem": "BuildSmirnoffSystem"}
        )
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
            if "conditional" in name:
                path = name.replace("|", "_")

                try:
                    output = protocol_graph._execute_protocol(
                        path,
                        protocol,
                        True,
                        *previous_output_paths,
                        available_resources=None,
                        safe_exceptions=False,
                    )
                except RuntimeError as e:
                    # this is meant to happen!
                    if not success and "failed to converge" in str(e):
                        continue
                    else:
                        raise e
                if not success:
                    raise AssertionError("Equilibration should have failed")
        

        
    def test_data_storage_and_retrieval(
        self,
        dummy_dataset,
        dhmix_density_CCCO,
        force_field_source
    ):
        """
        Test the storage and retrieval of equilibration data.
        """

        equilibration_options = _get_equilibration_request_options(
            error_tolerances=self._generate_error_tolerances(),
        )
        equilibration_options.batch_mode = BatchMode.NoBatch

        os.chdir(dhmix_density_CCCO)
        with DaskLocalCluster(number_of_workers=1) as calculation_backend:
            server = EvaluatorServer(
                calculation_backend=calculation_backend,
                working_directory=".",
                delete_working_files=False
            )
            with server:
                client = EvaluatorClient()

                # make and copy over working files to expected directory
                batch_path = pathlib.Path("EquilibrationLayer/batch_0000")
                batch_path.mkdir(exist_ok=True, parents=True)
                _copy_property_working_data(
                    source_directory="test/workflows/equilibration/dhmix-density-CCCO",
                    uuid_prefix="1",
                    destination_directory=batch_path,
                    # include_data_files=True
                )

                # check storage is empty
                storage_path = "stored_data"
                storage_path = pathlib.Path(storage_path)
                assert len(list(storage_path.rglob("*/output.pdb"))) == 0
                
                # test equilibration
                request, error = client.request_estimate(
                    dummy_dataset,
                    force_field_source,
                    equilibration_options,
                )
                assert error is None
                results, exception = request.results(synchronous=True, polling_interval=30)

                # check execution finished
                assert exception is None
                assert len(results.queued_properties) == 0
                assert len(results.estimated_properties) == 0
                assert len(results.unsuccessful_properties) == 0
                assert len(results.equilibrated_properties) == 2
                assert len(results.exceptions) == 0

                # check data stored
                assert len(list(storage_path.rglob("*/output.pdb"))) == 3

                # test data queries
                ccco_query = _create_equilibration_data_query(
                    substance=Substance.from_components("CCCO")
                )
                ccco_boxes = server._storage_backend.query(ccco_query)
                key = next(iter(ccco_boxes.keys()))
                assert len(ccco_boxes[key]) == 1

                o_query = _create_equilibration_data_query(
                    substance=Substance.from_components("O")
                )
                o_boxes = server._storage_backend.query(o_query)
                key = next(iter(o_boxes.keys()))
                assert len(o_boxes[key]) == 1

                ccco_o_query = _create_equilibration_data_query(
                    substance=Substance.from_components("CCCO", "O")
                )
                ccco_o_boxes = server._storage_backend.query(ccco_o_query)
                key = next(iter(ccco_o_boxes.keys()))
                assert len(ccco_o_boxes[key]) == 1
            
    def test_short_circuit_found_data(self, dummy_dataset, force_field_source):
        """
        Test that finding all equilibrated boxes for a dataset
        short circuits the equilibration layer.
        """
        with temporary_cd():
            _copy_property_working_data(
                source_directory="test/workflows/preequilibrated_simulation/dhmix-density-CCCO",
                uuid_prefix="stored",
                destination_directory="."
            )

            equilibration_options = _get_equilibration_request_options(
                error_tolerances=self._generate_error_tolerances()
            )
            equilibration_options.batch_mode = BatchMode.NoBatch

            with DaskLocalCluster(number_of_workers=1) as calculation_backend:
                server = EvaluatorServer(
                    calculation_backend=calculation_backend,
                    working_directory=".",
                    delete_working_files=False
                )
                with server:
                    client = EvaluatorClient()

                    # check storage is full
                    storage_path = "stored_data"
                    storage_path = pathlib.Path(storage_path)
                    assert len(list(storage_path.rglob("*/output.pdb"))) == 3

                    # test equilibration stops
                    request, error = client.request_estimate(
                        dummy_dataset,
                        force_field_source,
                        equilibration_options,
                    )
                    assert error is None
                    results, exception = request.results(synchronous=True, polling_interval=30)

                    # check execution finished
                    assert exception is None
                    assert len(results.queued_properties) == 0
                    assert len(results.estimated_properties) == 0
                    assert len(results.unsuccessful_properties) == 0
                    assert len(results.equilibrated_properties) == 2
                    assert len(results.exceptions) == 0

                    # check data stored has not increased
                    assert len(list(storage_path.rglob("*/output.pdb"))) == 3


    def test_single_missing_box(self, dummy_dataset, force_field_source):
        """
        Test that for properties that require multiple boxes,
        only the missing box/es are re-computed.
        """
        with temporary_cd():
            _copy_property_working_data(
                source_directory="test/workflows/preequilibrated_simulation/dhmix-density-CCCO",
                uuid_prefix="stored_data_without_O",
                destination_directory="."
            )

            # rename stored_data_without_O to stored_data
            os.rename("stored_data_without_O", "stored_data")

            # copy working files for component_0
            _copy_property_working_data(
                source_directory="test/workflows/equilibration/dhmix-density-CCCO",
                uuid_prefix="1",
                suffix="component_0",
                destination_directory="."
            )

            equilibration_options = _get_equilibration_request_options(
                error_tolerances=self._generate_error_tolerances()
            )
            equilibration_options.batch_mode = BatchMode.NoBatch

            with DaskLocalCluster(number_of_workers=1) as calculation_backend:
                server = EvaluatorServer(
                    calculation_backend=calculation_backend,
                    working_directory=".",
                    delete_working_files=False
                )
                with server:
                    client = EvaluatorClient()

                    # make and copy over working files to expected directory
                    batch_path = pathlib.Path("EquilibrationLayer/batch_0000")
                    batch_path.mkdir(exist_ok=True, parents=True)
                    _copy_property_working_data(
                        source_directory="test/workflows/equilibration/dhmix-density-CCCO",
                        uuid_prefix="1",
                        suffix="component_0",
                        destination_directory=batch_path
                    )

                    # check storage is missing 1
                    storage_path = "stored_data"
                    storage_path = pathlib.Path(storage_path)
                    assert len(list(storage_path.rglob("*/output.pdb"))) == 2

                    # test equilibration stops
                    request, error = client.request_estimate(
                        dummy_dataset,
                        force_field_source,
                        equilibration_options,
                    )
                    assert error is None
                    results, exception = request.results(synchronous=True, polling_interval=30)

                    # check execution finished
                    assert exception is None
                    assert len(results.queued_properties) == 0
                    assert len(results.estimated_properties) == 0
                    assert len(results.unsuccessful_properties) == 0
                    assert len(results.equilibrated_properties) == 2
                    assert len(results.exceptions) == 0

                    # check data stored has only increased by 1
                    assert len(list(storage_path.rglob("*/output.pdb"))) == 3
