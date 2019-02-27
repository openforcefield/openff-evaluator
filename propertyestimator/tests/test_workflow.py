"""
Units tests for propertyestimator.layers.simulation
"""
import inspect
import re
import uuid
from collections import OrderedDict

import pytest
from simtk import unit

from propertyestimator.client import PropertyEstimatorOptions
from propertyestimator.layers.simulation import Workflow, WorkflowGraph
from propertyestimator.properties import PropertyPhase
from propertyestimator.properties.density import Density
from propertyestimator.properties.dielectric import DielectricConstant
from propertyestimator.properties.plugins import registered_properties
from propertyestimator.substances import Mixture
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import get_data_filename, graph
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow import WorkflowSchema
from propertyestimator.workflow.decorators import protocol_input
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol
from propertyestimator.workflow.schemas import ProtocolReplicator
from propertyestimator.workflow.utils import ReplicatorValue, ProtocolPath


def create_dummy_property(property_class):

    substance = Mixture()
    substance.add_component('C', 0.5)
    substance.add_component('CO', 0.5)

    dummy_property = property_class(thermodynamic_state=ThermodynamicState(temperature=298 * unit.kelvin,
                                                                           pressure=1 * unit.atmosphere),
                                    phase=PropertyPhase.Liquid,
                                    substance=substance,
                                    value=10 * unit.gram,
                                    uncertainty=1 * unit.gram,
                                    id=str(uuid.uuid4()))

    return dummy_property


@register_calculation_protocol()
class DummyReplicableProtocol(BaseProtocol):

    @protocol_input(value_type=list)
    def replicated_value_a(self):
        pass

    @protocol_input(value_type=list)
    def replicated_value_b(self):
        pass

    @protocol_input(value_type=EstimatedQuantity)
    def final_value(self):
        pass

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._replicated_value_a = None
        self._replicated_value_b = None

        self._final_value = EstimatedQuantity(1 * unit.kelvin, 0.1 * unit.kelvin, 'dummy')


@pytest.mark.parametrize("registered_property_name", registered_properties)
def test_workflow_schema(registered_property_name):
    """Tests serialisation and deserialization of a calculation schema."""

    registered_property = registered_properties[registered_property_name]

    schema = registered_property.get_default_workflow_schema()
    schema.validate_interfaces()

    json_schema = schema.json()
    print(json_schema)

    schema_from_json = WorkflowSchema.parse_raw(json_schema)
    print(schema_from_json)

    property_recreated_json = schema_from_json.json()
    assert json_schema == property_recreated_json


@pytest.mark.parametrize("registered_property_name", registered_properties)
def test_cloned_schema_merging(registered_property_name):
    """Tests that two, the exact the same, calculations get merged into one
    by the `WorkflowGraph`."""

    registered_property = registered_properties[registered_property_name]

    substance = Mixture()
    substance.add_component('C', 1.0)

    dummy_property = create_dummy_property(registered_property)

    workflow_schema = dummy_property.get_default_workflow_schema()

    global_metadata = Workflow.generate_default_metadata(dummy_property,
                                                         get_data_filename('forcefield/smirnoff99Frosst.offxml'),
                                                         PropertyEstimatorOptions())

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

    substance = Mixture()
    substance.add_component('C', 1.0)

    density = Density(thermodynamic_state=ThermodynamicState(temperature=298*unit.kelvin,
                                                             pressure=1*unit.atmosphere),
                      phase=PropertyPhase.Liquid,
                      substance=substance,
                      value=10*unit.gram/unit.mole,
                      uncertainty=1*unit.gram/unit.mole,
                      id=str(uuid.uuid4()))

    dielectric = DielectricConstant(thermodynamic_state=ThermodynamicState(temperature=298*unit.kelvin,
                                                                           pressure=1*unit.atmosphere),
                                    phase=PropertyPhase.Liquid,
                                    substance=substance,
                                    value=10*unit.gram/unit.mole,
                                    uncertainty=1*unit.gram/unit.mole,
                                    id=str(uuid.uuid4()))

    density_schema = density.get_default_workflow_schema()
    dielectric_schema = dielectric.get_default_workflow_schema()

    density_metadata = Workflow.generate_default_metadata(density,
                                                          get_data_filename('forcefield/smirnoff99Frosst.offxml'),
                                                          PropertyEstimatorOptions())

    dielectric_metadata = Workflow.generate_default_metadata(density,
                                                             get_data_filename('forcefield/smirnoff99Frosst.offxml'),
                                                             PropertyEstimatorOptions())

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

    replicator_a = ProtocolReplicator(id='rep_a')

    replicator_a.template_values = ['a', 'b']
    replicator_a.protocols_to_replicate = [ProtocolPath('', dummy_protocol.id)]

    replicator_b = ProtocolReplicator(id='rep_b')

    replicator_b.template_values = [1, 2]
    replicator_b.protocols_to_replicate = [ProtocolPath('', dummy_protocol.id)]

    dummy_schema.replicators = [
        replicator_a,
        replicator_b
    ]

    dummy_schema.validate_interfaces()

    dummy_property = create_dummy_property(Density)

    dummy_metadata = Workflow.generate_default_metadata(dummy_property,
                                                        get_data_filename('forcefield/smirnoff99Frosst.offxml'),
                                                        PropertyEstimatorOptions())

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
