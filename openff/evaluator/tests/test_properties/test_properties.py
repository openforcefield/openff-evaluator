"""
Units tests for the calculation schemas registered by
`evaluator.properties`.
"""
from collections import OrderedDict

import pytest

import openff.evaluator.properties
from openff.evaluator.layers import registered_calculation_schemas
from openff.evaluator.layers.workflow import WorkflowCalculationSchema
from openff.evaluator.tests.test_workflow.utils import create_dummy_metadata
from openff.evaluator.tests.utils import create_dummy_property
from openff.evaluator.utils import graph
from openff.evaluator.workflow import Workflow, WorkflowGraph, WorkflowSchema


def calculation_schema_generator():
    """A generator which loops over all registered calculation
    layers and the corresponding calculation schemas."""

    for calculation_layer in registered_calculation_schemas:
        for property_type in registered_calculation_schemas[calculation_layer]:
            yield calculation_layer, property_type


def workflow_merge_functions():
    """Returns functions which will merge two work flows into
    a single graph.
    """

    def function_a(workflow_a, workflow_b):
        workflow_graph = WorkflowGraph()

        workflow_graph.add_workflows(workflow_a)
        workflow_graph.add_workflows(workflow_b)

        return workflow_graph

    def function_b(workflow_a, workflow_b):
        workflow_graph = WorkflowGraph()
        workflow_graph.add_workflows(workflow_a, workflow_b)

        return workflow_graph

    return [function_a, function_b]


@pytest.mark.parametrize(
    "calculation_layer, property_type", calculation_schema_generator()
)
def test_validate_schemas(calculation_layer, property_type):
    """Tests that all registered calculation schemas are valid."""

    schema = registered_calculation_schemas[calculation_layer][property_type]

    if callable(schema):
        schema = schema()

    schema.validate()


@pytest.mark.parametrize(
    "calculation_layer, property_type", calculation_schema_generator()
)
def test_schema_serialization(calculation_layer, property_type):
    """Tests serialisation and deserialization of a calculation schema."""

    schema = registered_calculation_schemas[calculation_layer][property_type]

    if callable(schema):
        schema = schema()

    json_schema = schema.json()

    schema_from_json = WorkflowSchema.parse_json(json_schema)
    property_recreated_json = schema_from_json.json()

    assert json_schema == property_recreated_json


@pytest.mark.parametrize(
    "calculation_layer, property_type", calculation_schema_generator()
)
@pytest.mark.parametrize("workflow_merge_function", workflow_merge_functions())
def test_workflow_schema_merging(
    calculation_layer, property_type, workflow_merge_function
):
    """Tests that two of the exact the same calculations get merged into one
    by the `WorkflowGraph`."""

    if property_type == "HostGuestBindingAffinity":
        pytest.skip(
            "This test does not currently support host-guest binding affinities "
            "which usually require specialised property metadata."
        )

    schema = registered_calculation_schemas[calculation_layer][property_type]

    if callable(schema):
        schema = schema()

    if not isinstance(schema, WorkflowCalculationSchema):
        pytest.skip("Not a `WorkflowCalculationSchema`.")

    property_class = getattr(openff.evaluator.properties, property_type)

    dummy_property = create_dummy_property(property_class)

    global_metadata = create_dummy_metadata(dummy_property, calculation_layer)

    workflow_a = Workflow(global_metadata, "workflow_a")
    workflow_a.schema = schema.workflow_schema

    workflow_b = Workflow(global_metadata, "workflow_b")
    workflow_b.schema = schema.workflow_schema

    workflow_graph = workflow_merge_function(workflow_a, workflow_b)

    workflow_graph_a = workflow_a.to_graph()
    workflow_graph_b = workflow_b.to_graph()

    dependants_graph_a = workflow_graph_a._protocol_graph._build_dependants_graph(
        workflow_graph_a.protocols, False, apply_reduction=True
    )
    dependants_graph_b = workflow_graph_b._protocol_graph._build_dependants_graph(
        workflow_graph_b.protocols, False, apply_reduction=True
    )

    ordered_dict_a = OrderedDict(sorted(dependants_graph_a.items()))
    ordered_dict_a = {key: sorted(value) for key, value in ordered_dict_a.items()}
    ordered_dict_b = OrderedDict(sorted(dependants_graph_b.items()))
    ordered_dict_b = {key: sorted(value) for key, value in ordered_dict_b.items()}

    merge_order_a = graph.topological_sort(ordered_dict_a)
    merge_order_b = graph.topological_sort(ordered_dict_b)

    assert len(workflow_graph.protocols) == len(workflow_a.protocols)

    for protocol_id in workflow_a.protocols:
        assert protocol_id in workflow_graph.protocols

    for protocol_id_A, protocol_id_B in zip(merge_order_a, merge_order_b):
        assert protocol_id_A == protocol_id_B

        assert (
            workflow_a.protocols[protocol_id_A].schema.json()
            == workflow_b.protocols[protocol_id_B].schema.json()
        )
