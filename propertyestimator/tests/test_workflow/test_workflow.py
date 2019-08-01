"""
Units tests for propertyestimator.layers.simulation
"""
import tempfile
from collections import OrderedDict

import pytest
from simtk import unit

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
from propertyestimator.tests.test_workflow.utils import DummyReplicableProtocol, create_dummy_metadata, \
    DummyEstimatedQuantityProtocol
from propertyestimator.tests.utils import create_dummy_property
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import get_data_filename, graph
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow import WorkflowOptions, WorkflowSchema
from propertyestimator.workflow.schemas import ProtocolReplicator
from propertyestimator.workflow.utils import ReplicatorValue, ProtocolPath


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


def test_nested_replicators():

    dummy_schema = WorkflowSchema()

    dummy_protocol = DummyReplicableProtocol('dummy_$(rep_a)_$(rep_b)')

    dummy_protocol.replicated_value_a = ReplicatorValue('rep_a')
    dummy_protocol.replicated_value_b = ReplicatorValue('rep_b')

    dummy_schema.protocols[dummy_protocol.id] = dummy_protocol.schema

    dummy_schema.final_value_source = ProtocolPath('final_value', dummy_protocol.id)

    replicator_a = ProtocolReplicator(replicator_id='rep_a')

    replicator_a.template_values = ['a', 'b']
    replicator_a.protocols_to_replicate = [ProtocolPath('', dummy_protocol.id)]

    replicator_b = ProtocolReplicator(replicator_id='rep_b')

    replicator_b.template_values = [1, 2]
    replicator_b.protocols_to_replicate = [ProtocolPath('', dummy_protocol.id)]

    dummy_schema.replicators = [
        replicator_a,
        replicator_b
    ]

    dummy_schema.validate_interfaces()

    dummy_property = create_dummy_property(Density)

    dummy_metadata = Workflow.generate_default_metadata(dummy_property,
                                                        'smirnoff99Frosst-1.1.0.offxml',
                                                        [])

    dummy_workflow = Workflow(dummy_property, dummy_metadata)
    dummy_workflow.schema = dummy_schema

    assert len(dummy_workflow.protocols) == 4

    assert dummy_workflow.protocols[dummy_workflow.uuid + '|dummy_0_0'].replicated_value_a == 'a'
    assert dummy_workflow.protocols[dummy_workflow.uuid + '|dummy_0_1'].replicated_value_a == 'a'

    assert dummy_workflow.protocols[dummy_workflow.uuid + '|dummy_1_0'].replicated_value_a == 'b'
    assert dummy_workflow.protocols[dummy_workflow.uuid + '|dummy_1_1'].replicated_value_a == 'b'

    assert dummy_workflow.protocols[dummy_workflow.uuid + '|dummy_0_0'].replicated_value_b == 1
    assert dummy_workflow.protocols[dummy_workflow.uuid + '|dummy_0_1'].replicated_value_b == 2

    assert dummy_workflow.protocols[dummy_workflow.uuid + '|dummy_1_0'].replicated_value_b == 1
    assert dummy_workflow.protocols[dummy_workflow.uuid + '|dummy_1_1'].replicated_value_b == 2

    print(dummy_workflow.schema)


def test_simple_workflow_graph():
    dummy_schema = WorkflowSchema()

    dummy_protocol_a = DummyEstimatedQuantityProtocol('protocol_a')
    dummy_protocol_a.input_value = EstimatedQuantity(1 * unit.kelvin, 0.1 * unit.kelvin, 'dummy_source')

    dummy_schema.protocols[dummy_protocol_a.id] = dummy_protocol_a.schema

    dummy_protocol_b = DummyEstimatedQuantityProtocol('protocol_b')
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

    dummy_protocol_a = DummyEstimatedQuantityProtocol('protocol_a')
    dummy_protocol_a.input_value = EstimatedQuantity(1 * unit.kelvin, 0.1 * unit.kelvin, 'dummy_source')

    dummy_protocol_b = DummyEstimatedQuantityProtocol('protocol_b')
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
