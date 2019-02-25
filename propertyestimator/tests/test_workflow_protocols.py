"""
Units tests for propertyestimator.workflow
"""
import pytest
from simtk import unit

from propertyestimator.substances import Mixture
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow.decorators import protocol_input
from propertyestimator.workflow.protocols import BuildCoordinatesPackmol, BaseProtocol, AddQuantities
from propertyestimator.workflow.utils import ProtocolPath


class DummyEstimatedQuantityProtocol(BaseProtocol):

    @protocol_input(EstimatedQuantity)
    def input_value(self):
        pass

    @protocol_input(EstimatedQuantity)
    def output_value(self):
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._input_value = None
        self._output_value = None


class DummyProtocolWithDictInput(BaseProtocol):

    @protocol_input(dict)
    def input_value(self):
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)
        self._input_value = None


def test_nested_protocol_paths():

    value_protocol_a = DummyEstimatedQuantityProtocol('protocol_a')
    value_protocol_a.input_value = EstimatedQuantity(1 * unit.kelvin, 0.1 * unit.kelvin, 'constant')

    assert value_protocol_a.get_value(ProtocolPath('input_value.value')) == value_protocol_a.input_value.value

    value_protocol_a.set_value(ProtocolPath('input_value._value'), 0.5*unit.kelvin)
    assert value_protocol_a.input_value.value == 0.5*unit.kelvin

    value_protocol_b = DummyEstimatedQuantityProtocol('protocol_b')
    value_protocol_b.input_value = EstimatedQuantity(2 * unit.kelvin, 0.05 * unit.kelvin, 'constant')

    value_protocol_c = DummyEstimatedQuantityProtocol('protocol_c')
    value_protocol_c.input_value = EstimatedQuantity(4 * unit.kelvin, 0.01 * unit.kelvin, 'constant')

    add_values_protocol = AddQuantities('add_values')

    add_values_protocol.values = [
        ProtocolPath('output_value', value_protocol_a.id),
        ProtocolPath('output_value', value_protocol_b.id),
        ProtocolPath('output_value', value_protocol_b.id),
        5
    ]

    with pytest.raises(ValueError):
        add_values_protocol.get_value(ProtocolPath('valus[string]'))

    with pytest.raises(ValueError):
        add_values_protocol.get_value(ProtocolPath('values[string]'))

    input_values = add_values_protocol.get_value_references(ProtocolPath('values'))
    assert isinstance(input_values, dict) and len(input_values) == 3

    for index, value_reference in enumerate(input_values):

        input_value = add_values_protocol.get_value(value_reference)
        assert input_value.full_path == add_values_protocol.values[index].full_path

        add_values_protocol.set_value(value_reference, index)

    assert set(add_values_protocol.values) == {0, 1, 2, 5}

    dummy_dict_protocol = DummyProtocolWithDictInput('dict_protocol')

    dummy_dict_protocol.input_value = {
        'value_a': ProtocolPath('output_value', value_protocol_a.id),
        'value_b': ProtocolPath('output_value', value_protocol_b.id),
    }

    input_values = dummy_dict_protocol.get_value_references(ProtocolPath('input_value'))
    assert isinstance(input_values, dict) and len(input_values) == 2

    for index, value_reference in enumerate(input_values):

        input_value = dummy_dict_protocol.get_value(value_reference)

        dummy_dict_keys = list(dummy_dict_protocol.input_value.keys())
        assert input_value.full_path == dummy_dict_protocol.input_value[dummy_dict_keys[index]].full_path

        dummy_dict_protocol.set_value(value_reference, index)

    add_values_protocol_2 = AddQuantities('add_values')

    add_values_protocol_2.values = [
        [ProtocolPath('output_value', value_protocol_a.id)],
        [
            ProtocolPath('output_value', value_protocol_b.id),
            ProtocolPath('output_value', value_protocol_b.id)
        ]
    ]

    with pytest.raises(ValueError):
        add_values_protocol_2.get_value(ProtocolPath('valus[string]'))

    with pytest.raises(ValueError):
        add_values_protocol.get_value(ProtocolPath('values[string]'))

    pass

    # substance = Mixture()
    #
    # substance.add_component(smiles='C', mole_fraction=0.5)
    # substance.add_component(smiles='O', mole_fraction=0.5)
    #
    # build_coordinates = BuildCoordinatesPackmol('build_coordinates')
    # build_coordinates.substance = substance
    #
    # components = build_coordinates.get_value(ProtocolPath('substance.components', build_coordinates.id))
    # assert isinstance(components, list) and len(components) == 2
    #
    # fist_component = build_coordinates.get_value(ProtocolPath('substance.components[0]', build_coordinates.id))
    # assert isinstance(fist_component, Mixture.MixtureComponent) and fist_component.smiles == 'C'
    #
    # second_component = build_coordinates.get_value(ProtocolPath('substance.components[1]', build_coordinates.id))
    # assert isinstance(fist_component, Mixture.MixtureComponent) and fist_component.smiles == 'O'
