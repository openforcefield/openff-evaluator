import random
import tempfile

import pytest

from propertyestimator import unit
from propertyestimator.protocols.groups import ConditionalGroup, ProtocolGroup
from propertyestimator.protocols.miscellaneous import AddValues
from propertyestimator.tests.test_workflow.utils import DummyInputOutputProtocol
from propertyestimator.utils.exceptions import EvaluatorException
from propertyestimator.workflow.utils import ProtocolPath


def test_conditional_protocol_group():

    with tempfile.TemporaryDirectory() as directory:

        initial_value = 2 * unit.kelvin

        value_protocol_a = DummyInputOutputProtocol("protocol_a")
        value_protocol_a.input_value = initial_value

        add_values = AddValues("add_values")
        add_values.values = [
            ProtocolPath("output_value", value_protocol_a.id),
            ProtocolPath("output_value", value_protocol_a.id),
        ]

        condition = ConditionalGroup.Condition()
        condition.left_hand_value = ProtocolPath("result", add_values.id)
        condition.right_hand_value = ProtocolPath("output_value", value_protocol_a.id)
        condition.type = ConditionalGroup.ConditionType.GreaterThan

        protocol_group = ConditionalGroup("protocol_group")
        protocol_group.conditions.append(condition)
        protocol_group.add_protocols(value_protocol_a, add_values)

        result = protocol_group.execute(directory, None)

        assert not isinstance(result, EvaluatorException)
        assert (
            protocol_group.get_value(ProtocolPath("result", add_values.id))
            == 4 * unit.kelvin
        )


def test_conditional_protocol_group_fail():

    with tempfile.TemporaryDirectory() as directory:

        initial_value = 2 * unit.kelvin

        value_protocol_a = DummyInputOutputProtocol("protocol_a")
        value_protocol_a.input_value = initial_value

        add_values = AddValues("add_values")
        add_values.values = [
            ProtocolPath("output_value", value_protocol_a.id),
            ProtocolPath("output_value", value_protocol_a.id),
        ]

        condition = ConditionalGroup.Condition()
        condition.left_hand_value = ProtocolPath("result", add_values.id)
        condition.right_hand_value = ProtocolPath("output_value", value_protocol_a.id)
        condition.type = ConditionalGroup.ConditionType.LessThan

        protocol_group = ConditionalGroup("protocol_group")
        protocol_group.conditions.append(condition)
        protocol_group.max_iterations = 10
        protocol_group.add_protocols(value_protocol_a, add_values)

        with pytest.raises(RuntimeError):
            protocol_group.execute(directory, None)


def test_conditional_group_self_reference():
    """Tests that protocols within a conditional group
    can access the outputs of its parent, such as the
    current iteration of the group."""

    max_iterations = 10
    criteria = random.randint(1, max_iterations - 1)

    dummy_group = ConditionalGroup("conditional_group")
    dummy_group.max_iterations = max_iterations

    dummy_protocol = DummyInputOutputProtocol("protocol_a")
    dummy_protocol.input_value = ProtocolPath("current_iteration", dummy_group.id)

    dummy_condition_1 = ConditionalGroup.Condition()
    dummy_condition_1.left_hand_value = ProtocolPath(
        "output_value", dummy_group.id, dummy_protocol.id
    )
    dummy_condition_1.right_hand_value = criteria
    dummy_condition_1.type = ConditionalGroup.ConditionType.GreaterThan

    dummy_condition_2 = ConditionalGroup.Condition()
    dummy_condition_2.left_hand_value = ProtocolPath(
        "current_iteration", dummy_group.id
    )
    dummy_condition_2.right_hand_value = criteria
    dummy_condition_2.type = ConditionalGroup.ConditionType.GreaterThan

    dummy_group.add_protocols(dummy_protocol)
    dummy_group.add_condition(dummy_condition_1)
    dummy_group.add_condition(dummy_condition_2)

    with tempfile.TemporaryDirectory() as directory:

        assert not isinstance(dummy_group.execute(directory, None), EvaluatorException)
        assert dummy_protocol.output_value == criteria + 1
