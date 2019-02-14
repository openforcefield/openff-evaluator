"""
Units tests for propertyestimator.layers.simulation
"""

import uuid
from collections import OrderedDict

import pytest
from simtk import unit

from propertyestimator.client import PropertyEstimatorOptions
from propertyestimator.layers.simulation import DirectCalculation, DirectCalculationGraph
from propertyestimator.properties import PropertyPhase
from propertyestimator.properties.density import Density
from propertyestimator.properties.dielectric import DielectricConstant
from propertyestimator.properties.plugins import registered_properties
from propertyestimator.substances import Mixture
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import get_data_filename, graph
from propertyestimator.workflow import WorkflowSchema


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


@pytest.mark.parametrize("registered_property_name", registered_properties)
def test_calculation_schema(registered_property_name):
    """Tests serialisation and deserialization of a calculation schema."""

    registered_property = registered_properties[registered_property_name]

    schema = registered_property.get_default_calculation_schema()
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
    by the `DirectCalculationGraph`."""

    registered_property = registered_properties[registered_property_name]

    substance = Mixture()
    substance.add_component('C', 1.0)

    dummy_property = create_dummy_property(registered_property)

    calculation_schema = dummy_property.get_default_calculation_schema()

    calculation_a = DirectCalculation(dummy_property,
                                      get_data_filename('forcefield/smirnoff99Frosst.offxml'),
                                      calculation_schema,
                                      PropertyEstimatorOptions())

    calculation_b = DirectCalculation(dummy_property,
                                      get_data_filename('forcefield/smirnoff99Frosst.offxml'),
                                      calculation_schema,
                                      PropertyEstimatorOptions())

    calculation_graph = DirectCalculationGraph()

    calculation_graph.add_calculation(calculation_a)
    calculation_graph.add_calculation(calculation_b)

    ordered_dict_a = OrderedDict(sorted(calculation_a.dependants_graph.items()))
    ordered_dict_b = OrderedDict(sorted(calculation_b.dependants_graph.items()))

    merge_order_a = graph.topological_sort(ordered_dict_a)
    merge_order_b = graph.topological_sort(ordered_dict_b)

    assert len(calculation_graph._nodes_by_id) == len(calculation_a.protocols)

    for protocol_id in calculation_a.protocols:
        assert protocol_id in calculation_graph._nodes_by_id

    for protocol_id_A, protocol_id_B in zip(merge_order_a, merge_order_b):

        assert protocol_id_A == protocol_id_B

        assert calculation_a.protocols[protocol_id_A].schema.json() == \
               calculation_b.protocols[protocol_id_B].schema.json()


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

    density_schema = density.get_default_calculation_schema()
    dielectric_schema = dielectric.get_default_calculation_schema()

    density_calculation = DirectCalculation(density,
                                            get_data_filename('forcefield/smirnoff99Frosst.offxml'),
                                            density_schema,
                                            PropertyEstimatorOptions())

    dielectric_calculation = DirectCalculation(dielectric,
                                               get_data_filename('forcefield/smirnoff99Frosst.offxml'),
                                               dielectric_schema,
                                               PropertyEstimatorOptions())

    calculation_graph = DirectCalculationGraph('')

    calculation_graph.add_calculation(density_calculation)
    calculation_graph.add_calculation(dielectric_calculation)

    merge_order_a = graph.topological_sort(density_calculation.dependants_graph)
    merge_order_b = graph.topological_sort(dielectric_calculation.dependants_graph)

    for protocol_id_A, protocol_id_B in zip(merge_order_a, merge_order_b):

        if protocol_id_A.find('extract_traj') < 0 and protocol_id_A.find('extract_stats') < 0:

            assert density_calculation.protocols[protocol_id_A].schema.json() == \
                   dielectric_calculation.protocols[protocol_id_B].schema.json()

        else:

            assert density_calculation.protocols[protocol_id_A].schema.json() != \
                   dielectric_calculation.protocols[protocol_id_B].schema.json()


def test_simulation_layer():
    """Test the simulation estimation layer."""

    # if path.isdir('property-data'):
    #     shutil.rmtree('property-data')
    #
    # # Set up time-based logging to help debug threading issues.
    # formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
    #                               datefmt='%H:%M:%S')
    #
    # screen_handler = logging.StreamHandler(stream=sys.stdout)
    # screen_handler.setFormatter(formatter)
    #
    # logger = logging.getLogger()
    # logger.setLevel(logging.INFO)
    # logger.addHandler(screen_handler)
    #
    # dummy_pickle = b''
    # data_model = pickle.loads(dummy_pickle)
    #
    # backend = DaskLocalClusterBackend(1, 1)
    # backend.start()
    #
    # def dummy_callback(*args, **kwargs):
    #     print(args, kwargs)
    #     pass
    #
    # simulation_layer = SimulationLayer()
    # simulation_layer.schedule_calculation(backend, data_model, {}, dummy_callback, True)
    pass
