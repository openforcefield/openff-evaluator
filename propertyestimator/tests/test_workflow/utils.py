from propertyestimator import unit
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow import Workflow
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


def create_dummy_metadata(dummy_property, calculation_layer):

    global_metadata = Workflow.generate_default_metadata(dummy_property,
                                                         'smirnoff99Frosst-1.1.0.offxml',
                                                         [])

    if calculation_layer == 'ReweightingLayer':

        global_metadata['full_system_data'] = [
            ('data_path_0', 'ff_path_0'),
            ('data_path_1', 'ff_path_0'),
            ('data_path_2', 'ff_path_1')
        ]

        global_metadata['component_data'] = [
            [('data_path_3', 'ff_path_3'), ('data_path_4', 'ff_path_4')],
            [('data_path_5', 'ff_path_5'), ('data_path_6', 'ff_path_6')]
        ]

    return global_metadata


@register_calculation_protocol()
class DummyReplicableProtocol(BaseProtocol):

    replicated_value_a = protocol_input(docstring='', type_hint=list, default_value = protocol_input.UNDEFINED)
    replicated_value_b = protocol_input(docstring='', type_hint=list, default_value = protocol_input.UNDEFINED)
    final_value = protocol_output(docstring='', type_hint=EstimatedQuantity)

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._replicated_value_a = None
        self._replicated_value_b = None

        self._final_value = EstimatedQuantity(1 * unit.kelvin, 0.1 * unit.kelvin, 'dummy')


@register_calculation_protocol()
class DummyQuantityProtocol(BaseProtocol):

    input_value = protocol_input(docstring='', type_hint=unit.Quantity, default_value=protocol_input.UNDEFINED)
    output_value = protocol_output(docstring='', type_hint=unit.Quantity)

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._input_value = None
        self._output_value = None

    def execute(self, directory, available_resources):
        self._output_value = self._input_value
        return self._get_output_dictionary()


@register_calculation_protocol()
class DummyEstimatedQuantityProtocol(BaseProtocol):

    input_value = protocol_input(docstring='', type_hint=EstimatedQuantity, default_value=protocol_input.UNDEFINED)
    output_value = protocol_output(docstring='', type_hint=EstimatedQuantity)

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._input_value = None
        self._output_value = None

    def execute(self, directory, available_resources):
        self._output_value = self._input_value
        return self._get_output_dictionary()


@register_calculation_protocol()
class DummyProtocolWithDictInput(BaseProtocol):

    input_value = protocol_input(docstring='', type_hint=dict, default_value=protocol_input.UNDEFINED)
    output_value = protocol_output(docstring='', type_hint=dict)

    def __init__(self, protocol_id):

        super().__init__(protocol_id)
        self._input_value = None
        self._output_value = None

    def execute(self, directory, available_resources):
        self._output_value = self._input_value
        return self._get_output_dictionary()