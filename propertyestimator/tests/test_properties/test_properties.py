"""
Units tests for the calculation schemas registered by
`propertyestimator.properties`.
"""
from collections import OrderedDict

import pytest

import propertyestimator.properties
from propertyestimator import unit
from propertyestimator.datasets import PropertyPhase
from propertyestimator.layers import registered_calculation_schemas
from propertyestimator.layers.workflow import WorkflowCalculationSchema
from propertyestimator.substances import Component, MoleFraction, Substance
from propertyestimator.tests.test_workflow.utils import create_dummy_metadata
from propertyestimator.tests.utils import create_dummy_property
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import graph
from propertyestimator.workflow import Workflow, WorkflowGraph, WorkflowSchema


def calculation_schema_generator():
    """A generator which loops over all registered calculation
    layers and the corresponding calculation schemas."""

    for calculation_layer in registered_calculation_schemas:
        for property_type in registered_calculation_schemas[calculation_layer]:
            yield calculation_layer, property_type


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
def test_workflow_schema_merging(calculation_layer, property_type):
    """Tests that two of the exact the same calculations get merged into one
    by the `WorkflowGraph`."""

    schema = registered_calculation_schemas[calculation_layer][property_type]

    if callable(schema):
        schema = schema()

    if not isinstance(schema, WorkflowCalculationSchema):
        pytest.skip("Not a `WorkflowCalculationSchema`.")

    property_class = getattr(propertyestimator.properties, property_type)

    dummy_property = create_dummy_property(property_class)

    global_metadata = create_dummy_metadata(dummy_property, calculation_layer)

    workflow_a = Workflow(dummy_property, global_metadata)
    workflow_a.schema = schema.workflow_schema

    workflow_b = Workflow(dummy_property, global_metadata)
    workflow_b.schema = schema.workflow_schema

    workflow_graph = WorkflowGraph()

    workflow_graph.add_workflow(workflow_a)
    workflow_graph.add_workflow(workflow_b)

    ordered_dict_a = OrderedDict(sorted(workflow_a.dependants_graph.items()))
    ordered_dict_b = OrderedDict(sorted(workflow_b.dependants_graph.items()))

    merge_order_a = graph.topological_sort(ordered_dict_a)
    merge_order_b = graph.topological_sort(ordered_dict_b)

    assert len(workflow_graph._protocols_by_id) == len(workflow_a.protocols)

    for protocol_id in workflow_a.protocols:
        assert protocol_id in workflow_graph._protocols_by_id

    for protocol_id_A, protocol_id_B in zip(merge_order_a, merge_order_b):

        assert protocol_id_A == protocol_id_B

        assert (
            workflow_a.protocols[protocol_id_A].schema.json()
            == workflow_b.protocols[protocol_id_B].schema.json()
        )


def test_density_dielectric_merging():

    substance = Substance.from_components("C")

    density = propertyestimator.properties.Density(
        thermodynamic_state=ThermodynamicState(
            temperature=298 * unit.kelvin, pressure=1 * unit.atmosphere
        ),
        phase=PropertyPhase.Liquid,
        substance=substance,
        value=10 * unit.gram / unit.mole,
        uncertainty=1 * unit.gram / unit.mole,
    )

    dielectric = propertyestimator.properties.DielectricConstant(
        thermodynamic_state=ThermodynamicState(
            temperature=298 * unit.kelvin, pressure=1 * unit.atmosphere
        ),
        phase=PropertyPhase.Liquid,
        substance=substance,
        value=10 * unit.gram / unit.mole,
        uncertainty=1 * unit.gram / unit.mole,
    )

    density_schema = density.default_simulation_schema().workflow_schema
    dielectric_schema = dielectric.default_simulation_schema().workflow_schema

    density_metadata = Workflow.generate_default_metadata(
        density, "smirnoff99Frosst-1.1.0.offxml", []
    )

    dielectric_metadata = Workflow.generate_default_metadata(
        density, "smirnoff99Frosst-1.1.0.offxml", []
    )

    density_workflow = Workflow(density, density_metadata)
    density_workflow.schema = density_schema

    dielectric_workflow = Workflow(dielectric, dielectric_metadata)
    dielectric_workflow.schema = dielectric_schema

    workflow_graph = WorkflowGraph("")

    workflow_graph.add_workflow(density_workflow)
    workflow_graph.add_workflow(dielectric_workflow)

    merge_order_a = graph.topological_sort(density_workflow.dependants_graph)
    merge_order_b = graph.topological_sort(dielectric_workflow.dependants_graph)

    for protocol_id_A, protocol_id_B in zip(merge_order_a, merge_order_b):

        if (
            protocol_id_A.find("extract_traj") < 0
            and protocol_id_A.find("extract_stats") < 0
        ):

            assert (
                density_workflow.protocols[protocol_id_A].schema.json()
                == dielectric_workflow.protocols[protocol_id_B].schema.json()
            )

        else:

            assert (
                density_workflow.protocols[protocol_id_A].schema.json()
                != dielectric_workflow.protocols[protocol_id_B].schema.json()
            )
