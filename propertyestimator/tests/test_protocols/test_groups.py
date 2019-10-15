import random
import tempfile

from propertyestimator import unit
from propertyestimator.protocols.groups import ProtocolGroup, ConditionalGroup
from propertyestimator.protocols.miscellaneous import AddValues
from propertyestimator.tests.test_workflow.utils import DummyInputOutputProtocol
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.workflow.utils import ProtocolPath


def test_protocol_group():

    with tempfile.TemporaryDirectory() as directory:

        initial_value = random.random() * unit.kelvin

        protocol_group = ProtocolGroup('protocol_group')

        value_protocol_a = DummyInputOutputProtocol('protocol_a')
        value_protocol_a.input_value = initial_value

        value_protocol_b = DummyInputOutputProtocol('value_protocol_b')
        value_protocol_b.input_value = ProtocolPath('output_value', value_protocol_a.id)

        protocol_group.add_protocols(value_protocol_a, value_protocol_b)
        result = protocol_group.execute(directory, None)

        assert not isinstance(result, PropertyEstimatorException)
        assert protocol_group.get_value(ProtocolPath('output_value', value_protocol_b.id)) == initial_value


def test_conditional_protocol_group():

    with tempfile.TemporaryDirectory() as directory:

        initial_value = 2 * unit.kelvin

        value_protocol_a = DummyInputOutputProtocol('protocol_a')
        value_protocol_a.input_value = initial_value

        add_values = AddValues('add_values')
        add_values.values = [ProtocolPath('output_value', value_protocol_a.id),
                             ProtocolPath('output_value', value_protocol_a.id)]

        condition = ConditionalGroup.Condition()
        condition.left_hand_value = ProtocolPath('result', add_values.id)
        condition.right_hand_value = ProtocolPath('output_value', value_protocol_a.id)
        condition.type = ConditionalGroup.ConditionType.GreaterThan

        protocol_group = ConditionalGroup('protocol_group')
        protocol_group.conditions.append(condition)
        protocol_group.add_protocols(value_protocol_a, add_values)

        result = protocol_group.execute(directory, None)

        assert not isinstance(result, PropertyEstimatorException)
        assert protocol_group.get_value(ProtocolPath('result', add_values.id)) == 4 * unit.kelvin


def test_conditional_protocol_group_fail():

    with tempfile.TemporaryDirectory() as directory:

        initial_value = 2 * unit.kelvin

        value_protocol_a = DummyInputOutputProtocol('protocol_a')
        value_protocol_a.input_value = initial_value

        add_values = AddValues('add_values')
        add_values.values = [ProtocolPath('output_value', value_protocol_a.id),
                             ProtocolPath('output_value', value_protocol_a.id)]

        condition = ConditionalGroup.Condition()
        condition.left_hand_value = ProtocolPath('result', add_values.id)
        condition.right_hand_value = ProtocolPath('output_value', value_protocol_a.id)
        condition.type = ConditionalGroup.ConditionType.LessThan

        protocol_group = ConditionalGroup('protocol_group')
        protocol_group.conditions.append(condition)
        protocol_group.max_iterations = 10
        protocol_group.add_protocols(value_protocol_a, add_values)

        result = protocol_group.execute(directory, None)

        assert isinstance(result, PropertyEstimatorException)
