"""
Units tests for propertyestimator.layers.simulation
"""

from propertyestimator import unit
from propertyestimator.properties.density import Density
from propertyestimator.protocols.groups import ProtocolGroup
from propertyestimator.protocols.miscellaneous import AddValues
from propertyestimator.tests.test_workflow.utils import (
    DummyInputOutputProtocol,
    DummyReplicableProtocol,
)
from propertyestimator.tests.utils import create_dummy_property
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow import Workflow, WorkflowSchema
from propertyestimator.workflow.schemas import ProtocolReplicator
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue


def test_simple_replicators():

    dummy_schema = WorkflowSchema()

    replicator_id = "replicator"

    dummy_replicated_protocol = DummyInputOutputProtocol(f"dummy_$({replicator_id})")
    dummy_replicated_protocol.input_value = ReplicatorValue(replicator_id)

    dummy_protocol_single_value = DummyInputOutputProtocol(
        f"dummy_single_$({replicator_id})"
    )
    dummy_protocol_single_value.input_value = ProtocolPath(
        "output_value", dummy_replicated_protocol.id
    )

    dummy_protocol_list_value = AddValues(f"dummy_list")
    dummy_protocol_list_value.values = ProtocolPath(
        "output_value", dummy_replicated_protocol.id
    )

    dummy_schema.protocol_schemas = [
        dummy_replicated_protocol.schema,
        dummy_protocol_single_value.schema,
        dummy_protocol_list_value.schema,
    ]

    replicator = ProtocolReplicator(replicator_id)

    replicator.template_values = [
        EstimatedQuantity(1.0 * unit.kelvin, 1.0 * unit.kelvin, "dummy_source"),
        EstimatedQuantity(2.0 * unit.kelvin, 2.0 * unit.kelvin, "dummy_source"),
    ]

    dummy_schema.protocol_replicators = [replicator]
    dummy_schema.validate()

    dummy_property = create_dummy_property(Density)

    dummy_metadata = Workflow.generate_default_metadata(
        dummy_property, "smirnoff99Frosst-1.1.0.offxml", []
    )

    dummy_workflow = Workflow(dummy_metadata, "")
    dummy_workflow.schema = dummy_schema

    assert len(dummy_workflow.protocols) == 5

    assert (
        dummy_workflow.protocols["dummy_0"].input_value == replicator.template_values[0]
    )
    assert (
        dummy_workflow.protocols["dummy_1"].input_value == replicator.template_values[1]
    )

    assert dummy_workflow.protocols["dummy_single_0"].input_value == ProtocolPath(
        "output_value", "dummy_0"
    )
    assert dummy_workflow.protocols["dummy_single_1"].input_value == ProtocolPath(
        "output_value", "dummy_1"
    )

    assert len(dummy_workflow.protocols["dummy_list"].values) == 2

    assert dummy_workflow.protocols["dummy_list"].values[0] == ProtocolPath(
        "output_value", "dummy_0"
    )
    assert dummy_workflow.protocols["dummy_list"].values[1] == ProtocolPath(
        "output_value", "dummy_1"
    )


def test_group_replicators():

    dummy_schema = WorkflowSchema()

    replicator_id = "replicator"

    dummy_replicated_protocol = DummyInputOutputProtocol(f"dummy_$({replicator_id})")
    dummy_replicated_protocol.input_value = ReplicatorValue(replicator_id)

    dummy_group = ProtocolGroup("dummy_group")
    dummy_group.add_protocols(dummy_replicated_protocol)

    dummy_protocol_single_value = DummyInputOutputProtocol(
        f"dummy_single_$({replicator_id})"
    )
    dummy_protocol_single_value.input_value = ProtocolPath(
        "output_value", dummy_group.id, dummy_replicated_protocol.id
    )

    dummy_protocol_list_value = AddValues(f"dummy_list")
    dummy_protocol_list_value.values = ProtocolPath(
        "output_value", dummy_group.id, dummy_replicated_protocol.id
    )

    dummy_schema.protocol_schemas = [
        dummy_group.schema,
        dummy_protocol_single_value.schema,
        dummy_protocol_list_value.schema,
    ]

    replicator = ProtocolReplicator(replicator_id)

    replicator.template_values = [
        EstimatedQuantity(1.0 * unit.kelvin, 1.0 * unit.kelvin, "dummy_source"),
        EstimatedQuantity(2.0 * unit.kelvin, 2.0 * unit.kelvin, "dummy_source"),
    ]

    dummy_schema.protocol_replicators = [replicator]
    dummy_schema.validate()

    dummy_property = create_dummy_property(Density)

    dummy_metadata = Workflow.generate_default_metadata(
        dummy_property, "smirnoff99Frosst-1.1.0.offxml", []
    )

    dummy_workflow = Workflow(dummy_metadata, "")
    dummy_workflow.schema = dummy_schema

    assert len(dummy_workflow.protocols) == 4

    assert (
        dummy_workflow.protocols[dummy_group.id].protocols["dummy_0"].input_value
        == replicator.template_values[0]
    )
    assert (
        dummy_workflow.protocols[dummy_group.id].protocols["dummy_1"].input_value
        == replicator.template_values[1]
    )

    assert dummy_workflow.protocols["dummy_single_0"].input_value == ProtocolPath(
        "output_value", dummy_group.id, "dummy_0"
    )
    assert dummy_workflow.protocols["dummy_single_1"].input_value == ProtocolPath(
        "output_value", dummy_group.id, "dummy_1"
    )

    assert len(dummy_workflow.protocols["dummy_list"].values) == 2

    assert dummy_workflow.protocols["dummy_list"].values[0] == ProtocolPath(
        "output_value", dummy_group.id, "dummy_0"
    )
    assert dummy_workflow.protocols["dummy_list"].values[1] == ProtocolPath(
        "output_value", dummy_group.id, "dummy_1"
    )


def test_advanced_group_replicators():

    dummy_schema = WorkflowSchema()

    replicator_id = "replicator"

    dummy_replicated_protocol = DummyInputOutputProtocol(f"dummy_$({replicator_id})")
    dummy_replicated_protocol.input_value = ReplicatorValue(replicator_id)

    dummy_replicated_protocol_2 = DummyInputOutputProtocol(
        f"dummy_2_$({replicator_id})"
    )
    dummy_replicated_protocol_2.input_value = ReplicatorValue(replicator_id)

    dummy_group_2 = ProtocolGroup(f"dummy_group_2_$({replicator_id})")
    dummy_group_2.add_protocols(dummy_replicated_protocol_2)

    dummy_group = ProtocolGroup(f"dummy_group_$({replicator_id})")
    dummy_group.add_protocols(dummy_replicated_protocol, dummy_group_2)

    dummy_protocol_single_value = DummyInputOutputProtocol(
        f"dummy_single_$({replicator_id})"
    )
    dummy_protocol_single_value.input_value = ProtocolPath(
        "output_value", dummy_group.id, dummy_replicated_protocol.id
    )

    dummy_protocol_2_single_value = DummyInputOutputProtocol(
        f"dummy_single_2_$({replicator_id})"
    )
    dummy_protocol_2_single_value.input_value = ProtocolPath(
        "output_value", dummy_group.id, dummy_group_2.id, dummy_replicated_protocol_2.id
    )

    dummy_schema.protocol_schemas = [
        dummy_group.schema,
        dummy_protocol_single_value.schema,
        dummy_protocol_2_single_value.schema,
    ]

    replicator = ProtocolReplicator(replicator_id)

    replicator.template_values = [
        EstimatedQuantity(1.0 * unit.kelvin, 1.0 * unit.kelvin, "dummy_source"),
        EstimatedQuantity(2.0 * unit.kelvin, 2.0 * unit.kelvin, "dummy_source"),
    ]

    dummy_schema.protocol_replicators = [replicator]
    dummy_schema.validate()

    dummy_property = create_dummy_property(Density)

    dummy_metadata = Workflow.generate_default_metadata(
        dummy_property, "smirnoff99Frosst-1.1.0.offxml", []
    )

    dummy_workflow = Workflow(dummy_metadata, "")
    dummy_workflow.schema = dummy_schema

    assert len(dummy_workflow.protocols) == 6

    assert (
        dummy_workflow.protocols["dummy_group_0"].protocols["dummy_0"].input_value
        == replicator.template_values[0]
    )
    assert "dummy_1" not in dummy_workflow.protocols["dummy_group_0"].protocols

    assert (
        dummy_workflow.protocols["dummy_group_1"].protocols["dummy_1"].input_value
        == replicator.template_values[1]
    )
    assert "dummy_0" not in dummy_workflow.protocols["dummy_group_1"].protocols

    assert dummy_workflow.protocols["dummy_single_0"].input_value == ProtocolPath(
        "output_value", "dummy_group_0", "dummy_0"
    )
    assert dummy_workflow.protocols["dummy_single_1"].input_value == ProtocolPath(
        "output_value", "dummy_group_1", "dummy_1"
    )

    assert dummy_workflow.protocols["dummy_single_2_0"].input_value == ProtocolPath(
        "output_value", "dummy_group_0", "dummy_group_2_0", "dummy_2_0"
    )
    assert dummy_workflow.protocols["dummy_single_2_1"].input_value == ProtocolPath(
        "output_value", "dummy_group_1", "dummy_group_2_1", "dummy_2_1"
    )


def test_nested_replicators():

    dummy_schema = WorkflowSchema()

    dummy_protocol = DummyReplicableProtocol("dummy_$(rep_a)_$(rep_b)")

    dummy_protocol.replicated_value_a = ReplicatorValue("rep_a")
    dummy_protocol.replicated_value_b = ReplicatorValue("rep_b")

    dummy_schema.protocol_schemas = [dummy_protocol.schema]

    replicator_a = ProtocolReplicator(replicator_id="rep_a")
    replicator_a.template_values = ["a", "b"]

    replicator_b = ProtocolReplicator(replicator_id="rep_b")
    replicator_b.template_values = [1, 2]

    dummy_schema.protocol_replicators = [replicator_a, replicator_b]

    dummy_schema.validate()

    dummy_property = create_dummy_property(Density)

    dummy_metadata = Workflow.generate_default_metadata(
        dummy_property, "smirnoff99Frosst-1.1.0.offxml", []
    )

    dummy_workflow = Workflow(dummy_metadata, "")
    dummy_workflow.schema = dummy_schema

    assert len(dummy_workflow.protocols) == 4

    assert dummy_workflow.protocols["dummy_0_0"].replicated_value_a == "a"
    assert dummy_workflow.protocols["dummy_0_1"].replicated_value_a == "a"

    assert dummy_workflow.protocols["dummy_1_0"].replicated_value_a == "b"
    assert dummy_workflow.protocols["dummy_1_1"].replicated_value_a == "b"

    assert dummy_workflow.protocols["dummy_0_0"].replicated_value_b == 1
    assert dummy_workflow.protocols["dummy_0_1"].replicated_value_b == 2

    assert dummy_workflow.protocols["dummy_1_0"].replicated_value_b == 1
    assert dummy_workflow.protocols["dummy_1_1"].replicated_value_b == 2

    print(dummy_workflow.schema)


def test_advanced_nested_replicators():

    dummy_schema = WorkflowSchema()

    replicator_a = ProtocolReplicator(replicator_id="replicator_a")
    replicator_a.template_values = ["a", "b"]

    replicator_b = ProtocolReplicator(
        replicator_id=f"replicator_b_{replicator_a.placeholder_id}"
    )
    replicator_b.template_values = ProtocolPath(
        f"dummy_list[{replicator_a.placeholder_id}]", "global"
    )

    dummy_protocol = DummyReplicableProtocol(
        f"dummy_" f"{replicator_a.placeholder_id}_" f"{replicator_b.placeholder_id}"
    )

    dummy_protocol.replicated_value_a = ReplicatorValue(replicator_a.id)
    dummy_protocol.replicated_value_b = ReplicatorValue(replicator_b.id)

    dummy_schema.protocol_schemas = [dummy_protocol.schema]
    dummy_schema.protocol_replicators = [replicator_a, replicator_b]

    dummy_schema.validate()

    dummy_property = create_dummy_property(Density)
    dummy_metadata = Workflow.generate_default_metadata(
        dummy_property, "smirnoff99Frosst-1.1.0.offxml", []
    )
    dummy_metadata["dummy_list"] = [[1], [2]]

    dummy_workflow = Workflow(dummy_metadata, "")
    dummy_workflow.schema = dummy_schema

    assert len(dummy_workflow.protocols) == 2

    assert dummy_workflow.protocols["dummy_0_0"].replicated_value_a == "a"
    assert dummy_workflow.protocols["dummy_0_0"].replicated_value_b == 1

    assert dummy_workflow.protocols["dummy_1_0"].replicated_value_a == "b"
    assert dummy_workflow.protocols["dummy_1_0"].replicated_value_b == 2
