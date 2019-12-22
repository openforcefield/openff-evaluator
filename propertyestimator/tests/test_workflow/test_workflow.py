"""
Units tests for propertyestimator.layers.simulation
"""
import tempfile

from propertyestimator import unit
from propertyestimator.backends import ComputeResources, DaskLocalCluster
from propertyestimator.layers.layers import CalculationLayerResult
from propertyestimator.properties.density import Density
from propertyestimator.protocols.groups import ConditionalGroup
from propertyestimator.tests.test_workflow.utils import DummyInputOutputProtocol
from propertyestimator.tests.utils import create_dummy_property
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow import Workflow, WorkflowGraph, WorkflowSchema
from propertyestimator.workflow.schemas import ProtocolReplicator
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue


def test_simple_workflow_graph():
    dummy_schema = WorkflowSchema()

    dummy_protocol_a = DummyInputOutputProtocol("protocol_a")
    dummy_protocol_a.input_value = EstimatedQuantity(
        1 * unit.kelvin, 0.1 * unit.kelvin, "dummy_source"
    )

    dummy_schema.protocols[dummy_protocol_a.id] = dummy_protocol_a.schema

    dummy_protocol_b = DummyInputOutputProtocol("protocol_b")
    dummy_protocol_b.input_value = ProtocolPath("output_value", dummy_protocol_a.id)

    dummy_schema.protocols[dummy_protocol_b.id] = dummy_protocol_b.schema

    dummy_schema.final_value_source = ProtocolPath("output_value", dummy_protocol_b.id)

    dummy_schema.validate()

    dummy_property = create_dummy_property(Density)

    dummy_workflow = Workflow(dummy_property, {})
    dummy_workflow.schema = dummy_schema

    with tempfile.TemporaryDirectory() as temporary_directory:

        workflow_graph = WorkflowGraph(temporary_directory)
        workflow_graph.add_workflow(dummy_workflow)

        dask_local_backend = DaskLocalCluster(1, ComputeResources(1))
        dask_local_backend.start()

        results_futures = workflow_graph.submit(dask_local_backend)

        assert len(results_futures) == 1

        result = results_futures[0].result()
        assert isinstance(result, CalculationLayerResult)
        assert result.calculated_property.value == 1 * unit.kelvin


def test_simple_workflow_graph_with_groups():
    dummy_schema = WorkflowSchema()

    dummy_protocol_a = DummyInputOutputProtocol("protocol_a")
    dummy_protocol_a.input_value = EstimatedQuantity(
        1 * unit.kelvin, 0.1 * unit.kelvin, "dummy_source"
    )

    dummy_protocol_b = DummyInputOutputProtocol("protocol_b")
    dummy_protocol_b.input_value = ProtocolPath("output_value", dummy_protocol_a.id)

    conditional_group = ConditionalGroup("conditional_group")
    conditional_group.add_protocols(dummy_protocol_a, dummy_protocol_b)

    condition = ConditionalGroup.Condition()
    condition.right_hand_value = 2 * unit.kelvin
    condition.type = ConditionalGroup.ConditionType.LessThan

    condition.left_hand_value = ProtocolPath(
        "output_value.value", conditional_group.id, dummy_protocol_b.id
    )

    conditional_group.add_condition(condition)

    dummy_schema.protocols[conditional_group.id] = conditional_group.schema

    dummy_schema.final_value_source = ProtocolPath(
        "output_value", conditional_group.id, dummy_protocol_b.id
    )

    dummy_schema.validate()

    dummy_property = create_dummy_property(Density)

    dummy_workflow = Workflow(dummy_property, {})
    dummy_workflow.schema = dummy_schema

    with tempfile.TemporaryDirectory() as temporary_directory:

        workflow_graph = WorkflowGraph(temporary_directory)
        workflow_graph.add_workflow(dummy_workflow)

        dask_local_backend = DaskLocalCluster(1, ComputeResources(1))
        dask_local_backend.start()

        results_futures = workflow_graph.submit(dask_local_backend)

        assert len(results_futures) == 1

        result = results_futures[0].result()
        assert isinstance(result, CalculationLayerResult)
        assert result.calculated_property.value == 1 * unit.kelvin


def test_nested_input():

    dummy_schema = WorkflowSchema()

    dict_protocol = DummyInputOutputProtocol("dict_protocol")
    dict_protocol.input_value = {"a": ThermodynamicState(temperature=1 * unit.kelvin)}
    dummy_schema.protocols[dict_protocol.id] = dict_protocol.schema

    quantity_protocol = DummyInputOutputProtocol("quantity_protocol")
    quantity_protocol.input_value = ProtocolPath(
        "output_value[a].temperature", dict_protocol.id
    )
    dummy_schema.protocols[quantity_protocol.id] = quantity_protocol.schema

    dummy_schema.validate()

    dummy_property = create_dummy_property(Density)

    dummy_workflow = Workflow(dummy_property, {})
    dummy_workflow.schema = dummy_schema

    with tempfile.TemporaryDirectory() as temporary_directory:

        workflow_graph = WorkflowGraph(temporary_directory)
        workflow_graph.add_workflow(dummy_workflow)

        dask_local_backend = DaskLocalCluster(1, ComputeResources(1))
        dask_local_backend.start()

        results_futures = workflow_graph.submit(dask_local_backend)

        assert len(results_futures) == 1

        result = results_futures[0].result()
        assert isinstance(result, CalculationLayerResult)


def test_index_replicated_protocol():

    dummy_schema = WorkflowSchema()

    dummy_replicator = ProtocolReplicator("dummy_replicator")
    dummy_replicator.template_values = ["a", "b", "c", "d"]
    dummy_schema.replicators = [dummy_replicator]

    replicated_protocol = DummyInputOutputProtocol(
        f"protocol_{dummy_replicator.placeholder_id}"
    )
    replicated_protocol.input_value = ReplicatorValue(dummy_replicator.id)
    dummy_schema.protocols[replicated_protocol.id] = replicated_protocol.schema

    for index in range(len(dummy_replicator.template_values)):

        indexing_protocol = DummyInputOutputProtocol(f"indexing_protocol_{index}")
        indexing_protocol.input_value = ProtocolPath(
            "output_value", f"protocol_{index}"
        )
        dummy_schema.protocols[indexing_protocol.id] = indexing_protocol.schema

    dummy_schema.validate()

    dummy_property = create_dummy_property(Density)

    dummy_workflow = Workflow(dummy_property, {})
    dummy_workflow.schema = dummy_schema
