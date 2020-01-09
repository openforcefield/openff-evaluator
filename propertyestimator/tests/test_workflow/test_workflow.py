"""
Units tests for propertyestimator.layers.simulation
"""
import tempfile

from propertyestimator import unit
from propertyestimator.backends import DaskLocalCluster
from propertyestimator.protocols.groups import ConditionalGroup
from propertyestimator.tests.test_workflow.utils import DummyInputOutputProtocol
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.workflow import Workflow, WorkflowResult, WorkflowSchema
from propertyestimator.workflow.schemas import ProtocolReplicator
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue


def test_simple_workflow_graph():

    expected_value = (1 * unit.kelvin).plus_minus(0.1 * unit.kelvin)

    protocol_a = DummyInputOutputProtocol("protocol_a")
    protocol_a.input_value = expected_value
    protocol_b = DummyInputOutputProtocol("protocol_b")
    protocol_b.input_value = ProtocolPath("output_value", protocol_a.id)

    schema = WorkflowSchema()
    schema.protocol_schemas = [protocol_a.schema, protocol_b.schema]
    schema.final_value_source = ProtocolPath("output_value", protocol_b.id)
    schema.validate()

    workflow = Workflow({})
    workflow.schema = schema

    workflow_graph = workflow.to_graph()

    with tempfile.TemporaryDirectory() as directory:

        with DaskLocalCluster() as calculation_backend:

            results_futures = workflow_graph.execute(directory, calculation_backend)

            assert len(results_futures) == 1
            result = results_futures[0].result()

        assert isinstance(result, WorkflowResult)
        assert result.value == expected_value


def test_workflow_with_groups():

    expected_value = (1 * unit.kelvin).plus_minus(0.1 * unit.kelvin)

    protocol_a = DummyInputOutputProtocol("protocol_a")
    protocol_a.input_value = expected_value
    protocol_b = DummyInputOutputProtocol("protocol_b")
    protocol_b.input_value = ProtocolPath("output_value", protocol_a.id)

    conditional_group = ConditionalGroup("conditional_group")
    conditional_group.add_protocols(protocol_a, protocol_b)

    condition = ConditionalGroup.Condition()
    condition.right_hand_value = 2 * unit.kelvin
    condition.type = ConditionalGroup.Condition.Type.LessThan
    condition.left_hand_value = ProtocolPath(
        "output_value.value", conditional_group.id, protocol_b.id
    )
    conditional_group.add_condition(condition)

    schema = WorkflowSchema()
    schema.protocol_schemas = [conditional_group.schema]
    schema.final_value_source = ProtocolPath(
        "output_value", conditional_group.id, protocol_b.id
    )
    schema.validate()

    workflow = Workflow({})
    workflow.schema = schema

    workflow_graph = workflow.to_graph()

    with tempfile.TemporaryDirectory() as directory:

        with DaskLocalCluster() as calculation_backend:

            results_futures = workflow_graph.execute(directory, calculation_backend)
            assert len(results_futures) == 1

            result = results_futures[0].result()

        assert isinstance(result, WorkflowResult)
        assert result.value == expected_value


def test_nested_input():

    dict_protocol = DummyInputOutputProtocol("dict_protocol")
    dict_protocol.input_value = {"a": ThermodynamicState(1.0 * unit.kelvin)}

    quantity_protocol = DummyInputOutputProtocol("quantity_protocol")
    quantity_protocol.input_value = ProtocolPath(
        "output_value[a].temperature", dict_protocol.id
    )

    schema = WorkflowSchema()
    schema.protocol_schemas = [dict_protocol.schema, quantity_protocol.schema]
    schema.validate()

    workflow = Workflow({})
    workflow.schema = schema

    workflow_graph = workflow.to_graph()

    with tempfile.TemporaryDirectory() as temporary_directory:

        with DaskLocalCluster() as calculation_backend:

            results_futures = workflow_graph.execute(
                temporary_directory, calculation_backend
            )

            assert len(results_futures) == 1
            result = results_futures[0].result()

    assert isinstance(result, WorkflowResult)


def test_index_replicated_protocol():

    replicator = ProtocolReplicator("replicator")
    replicator.template_values = ["a", "b", "c", "d"]

    replicated_protocol = DummyInputOutputProtocol(
        f"protocol_{replicator.placeholder_id}"
    )
    replicated_protocol.input_value = ReplicatorValue(replicator.id)

    schema = WorkflowSchema()
    schema.protocol_replicators = [replicator]
    schema.protocol_schemas = [replicated_protocol.schema]

    for index in range(len(replicator.template_values)):

        indexing_protocol = DummyInputOutputProtocol(f"indexing_protocol_{index}")
        indexing_protocol.input_value = ProtocolPath(
            "output_value", f"protocol_{index}"
        )
        schema.protocol_schemas.append(indexing_protocol.schema)

    schema.validate()

    workflow = Workflow({})
    workflow.schema = schema
