"""
Units tests for propertyestimator.layers.simulation
"""
import tempfile
from collections import OrderedDict

import pytest

from propertyestimator import unit
from propertyestimator.backends import DaskLocalCluster, ComputeResources
from propertyestimator.layers import available_layers
from propertyestimator.layers.layers import CalculationLayerResult
from propertyestimator.layers.simulation import Workflow, WorkflowGraph
from propertyestimator.properties import PropertyPhase
from propertyestimator.properties.density import Density
from propertyestimator.properties.dielectric import DielectricConstant
from propertyestimator.properties.plugins import registered_properties
from propertyestimator.protocols.groups import ConditionalGroup
from propertyestimator.substances import Substance
from propertyestimator.tests.test_workflow.utils import create_dummy_metadata, \
    DummyInputOutputProtocol
from propertyestimator.tests.utils import create_dummy_property
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import graph
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow import WorkflowOptions, WorkflowSchema
from propertyestimator.workflow.schemas import ProtocolReplicator
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue


@pytest.mark.parametrize("registered_property_name", registered_properties)
@pytest.mark.parametrize("available_layer", available_layers)
def test_workflow_schema_simulation(registered_property_name, available_layer):
    """Tests serialisation and deserialization of a calculation schema."""

    registered_property = registered_properties[registered_property_name]

    schema = registered_property.get_default_workflow_schema(available_layer, WorkflowOptions())

    if schema is None:
        return

    schema.validate_interfaces()

    json_schema = schema.json()
    print(json_schema)

    schema_from_json = WorkflowSchema.parse_json(json_schema)
    print(schema_from_json)

    property_recreated_json = schema_from_json.json()
    assert json_schema == property_recreated_json


@pytest.mark.parametrize("registered_property_name", registered_properties)
@pytest.mark.parametrize("available_layer", available_layers)
def test_cloned_schema_merging_simulation(registered_property_name, available_layer):
    """Tests that two, the exact the same, calculations get merged into one
    by the `WorkflowGraph`."""

    registered_property = registered_properties[registered_property_name]

    dummy_property = create_dummy_property(registered_property)

    workflow_schema = dummy_property.get_default_workflow_schema(available_layer, WorkflowOptions())

    if workflow_schema is None:
        return

    global_metadata = create_dummy_metadata(dummy_property, available_layer)

    workflow_a = Workflow(dummy_property, global_metadata)
    workflow_a.schema = workflow_schema

    workflow_b = Workflow(dummy_property, global_metadata)
    workflow_b.schema = workflow_schema

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

        assert workflow_a.protocols[protocol_id_A].schema.json() == \
               workflow_b.protocols[protocol_id_B].schema.json()


def test_density_dielectric_merging():

    substance = Substance()
    substance.add_component(Substance.Component(smiles='C'),
                            Substance.MoleFraction())

    density = Density(thermodynamic_state=ThermodynamicState(temperature=298*unit.kelvin,
                                                             pressure=1*unit.atmosphere),
                      phase=PropertyPhase.Liquid,
                      substance=substance,
                      value=10*unit.gram/unit.mole,
                      uncertainty=1*unit.gram/unit.mole)

    dielectric = DielectricConstant(thermodynamic_state=ThermodynamicState(temperature=298*unit.kelvin,
                                                                           pressure=1*unit.atmosphere),
                                    phase=PropertyPhase.Liquid,
                                    substance=substance,
                                    value=10*unit.gram/unit.mole,
                                    uncertainty=1*unit.gram/unit.mole)

    density_schema = density.get_default_workflow_schema('SimulationLayer', WorkflowOptions())
    dielectric_schema = dielectric.get_default_workflow_schema('SimulationLayer', WorkflowOptions())

    density_metadata = Workflow.generate_default_metadata(density,
                                                          'smirnoff99Frosst-1.1.0.offxml',
                                                          [])

    dielectric_metadata = Workflow.generate_default_metadata(density,
                                                             'smirnoff99Frosst-1.1.0.offxml',
                                                             [])

    density_workflow = Workflow(density, density_metadata)
    density_workflow.schema = density_schema

    dielectric_workflow = Workflow(dielectric, dielectric_metadata)
    dielectric_workflow.schema = dielectric_schema

    workflow_graph = WorkflowGraph('')

    workflow_graph.add_workflow(density_workflow)
    workflow_graph.add_workflow(dielectric_workflow)

    merge_order_a = graph.topological_sort(density_workflow.dependants_graph)
    merge_order_b = graph.topological_sort(dielectric_workflow.dependants_graph)

    for protocol_id_A, protocol_id_B in zip(merge_order_a, merge_order_b):

        if protocol_id_A.find('extract_traj') < 0 and protocol_id_A.find('extract_stats') < 0:

            assert density_workflow.protocols[protocol_id_A].schema.json() == \
                   dielectric_workflow.protocols[protocol_id_B].schema.json()

        else:

            assert density_workflow.protocols[protocol_id_A].schema.json() != \
                   dielectric_workflow.protocols[protocol_id_B].schema.json()


def test_simple_workflow_graph():
    dummy_schema = WorkflowSchema()

    dummy_protocol_a = DummyInputOutputProtocol('protocol_a')
    dummy_protocol_a.input_value = EstimatedQuantity(1 * unit.kelvin, 0.1 * unit.kelvin, 'dummy_source')

    dummy_schema.protocols[dummy_protocol_a.id] = dummy_protocol_a.schema

    dummy_protocol_b = DummyInputOutputProtocol('protocol_b')
    dummy_protocol_b.input_value = ProtocolPath('output_value', dummy_protocol_a.id)

    dummy_schema.protocols[dummy_protocol_b.id] = dummy_protocol_b.schema

    dummy_schema.final_value_source = ProtocolPath('output_value', dummy_protocol_b.id)

    dummy_schema.validate_interfaces()

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

    dummy_protocol_a = DummyInputOutputProtocol('protocol_a')
    dummy_protocol_a.input_value = EstimatedQuantity(1 * unit.kelvin, 0.1 * unit.kelvin, 'dummy_source')

    dummy_protocol_b = DummyInputOutputProtocol('protocol_b')
    dummy_protocol_b.input_value = ProtocolPath('output_value', dummy_protocol_a.id)

    conditional_group = ConditionalGroup('conditional_group')
    conditional_group.add_protocols(dummy_protocol_a, dummy_protocol_b)

    condition = ConditionalGroup.Condition()
    condition.right_hand_value = 2*unit.kelvin
    condition.type = ConditionalGroup.ConditionType.LessThan

    condition.left_hand_value = ProtocolPath('output_value.value', conditional_group.id,
                                                                   dummy_protocol_b.id)

    conditional_group.add_condition(condition)

    dummy_schema.protocols[conditional_group.id] = conditional_group.schema

    dummy_schema.final_value_source = ProtocolPath('output_value', conditional_group.id,
                                                                   dummy_protocol_b.id)

    dummy_schema.validate_interfaces()

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

    dict_protocol = DummyInputOutputProtocol('dict_protocol')
    dict_protocol.input_value = {'a': ThermodynamicState(temperature=1*unit.kelvin)}
    dummy_schema.protocols[dict_protocol.id] = dict_protocol.schema

    quantity_protocol = DummyInputOutputProtocol('quantity_protocol')
    quantity_protocol.input_value = ProtocolPath('output_value[a].temperature', dict_protocol.id)
    dummy_schema.protocols[quantity_protocol.id] = quantity_protocol.schema

    dummy_schema.validate_interfaces()

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

    dummy_replicator = ProtocolReplicator('dummy_replicator')
    dummy_replicator.template_values = ['a', 'b', 'c', 'd']
    dummy_schema.replicators = [dummy_replicator]

    replicated_protocol = DummyInputOutputProtocol(f'protocol_{dummy_replicator.placeholder_id}')
    replicated_protocol.input_value = ReplicatorValue(dummy_replicator.id)
    dummy_schema.protocols[replicated_protocol.id] = replicated_protocol.schema

    for index in range(len(dummy_replicator.template_values)):

        indexing_protocol = DummyInputOutputProtocol(f'indexing_protocol_{index}')
        indexing_protocol.input_value = ProtocolPath('output_value', f'protocol_{index}')
        dummy_schema.protocols[indexing_protocol.id] = indexing_protocol.schema

    dummy_schema.validate_interfaces()

    dummy_property = create_dummy_property(Density)

    dummy_workflow = Workflow(dummy_property, {})
    dummy_workflow.schema = dummy_schema
