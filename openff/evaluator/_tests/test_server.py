"""
Units tests for the openff.evaluator.server module.
"""

import tempfile
import os
import pytest
from time import sleep

from openff.units import unit

from openff.evaluator._tests.utils import create_dummy_property
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.client import EvaluatorClient, RequestOptions
from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.layers import (
    CalculationLayer,
    CalculationLayerResult,
    CalculationLayerSchema,
    calculation_layer,
)
from openff.evaluator.properties import Density, EnthalpyOfVaporization
from openff.evaluator.server.server import Batch, EvaluatorServer
from openff.evaluator.substances import Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.utils import temporarily_change_directory


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
        cls, calculation_backend, storage_backend, layer_directory, batch
    ):
        futures = [
            calculation_backend.submit_task(
                QuickCalculationLayer.process_property,
                batch.queued_properties[0],
            ),
        ]

        return futures

    @staticmethod
    def process_property(physical_property, **_):
        """Return a result as if the property had been successfully estimated."""
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

    batch = Batch()
    batch.force_field_id = ""
    batch.options = RequestOptions()
    batch.options.calculation_layers = ["QuickCalculationLayer"]
    batch.options.calculation_schemas = {
        "Density": {"QuickCalculationLayer": CalculationLayerSchema()}
    }
    batch.parameter_gradient_keys = []
    batch.queued_properties = [*data_set]
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


@pytest.fixture
def c_o_dataset():
    thermodynamic_state = ThermodynamicState(
        temperature=1.0 * unit.kelvin, pressure=1.0 * unit.atmosphere
    )

    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(
        Density(
            thermodynamic_state=thermodynamic_state,
            substance=Substance.from_components("O", "C"),
            value=0.0 * unit.kilogram / unit.meter**3,
        ),
        EnthalpyOfVaporization(
            thermodynamic_state=thermodynamic_state,
            substance=Substance.from_components("O", "C"),
            value=0.0 * unit.kilojoule / unit.mole,
        ),
        Density(
            thermodynamic_state=thermodynamic_state,
            substance=Substance.from_components("O", "CO"),
            value=0.0 * unit.kilogram / unit.meter**3,
        ),
        EnthalpyOfVaporization(
            thermodynamic_state=thermodynamic_state,
            substance=Substance.from_components("O", "CO"),
            value=0.0 * unit.kilojoule / unit.mole,
        ),
    )
    return data_set


@pytest.fixture
def dataset_submission(c_o_dataset):
    options = RequestOptions()

    submission = EvaluatorClient._Submission()
    submission.dataset = c_o_dataset
    submission.options = options

    return submission


def test_same_component_batching(dataset_submission, tmp_path):
    os.chdir(tmp_path)
    with DaskLocalCluster() as calculation_backend:
        server = EvaluatorServer(calculation_backend)
        batches = server._batch_by_same_component(dataset_submission, "")

    assert len(batches) == 2

    assert len(batches[0].queued_properties) == 2
    assert len(batches[1].queued_properties) == 2


def test_shared_component_batching(dataset_submission, tmp_path):
    os.chdir(tmp_path)
    with DaskLocalCluster() as calculation_backend:
        server = EvaluatorServer(calculation_backend)
        batches = server._batch_by_shared_component(dataset_submission, "")

    assert len(batches) == 1
    assert len(batches[0].queued_properties) == 4


def test_nobatching(dataset_submission, tmp_path):
    os.chdir(tmp_path)
    with DaskLocalCluster() as calculation_backend:
        server = EvaluatorServer(calculation_backend)
        batches = server._no_batch(dataset_submission, "")

    assert len(batches) == 1
    assert len(batches[0].queued_properties) == 4
    assert batches[0].id == "batch_0000"
