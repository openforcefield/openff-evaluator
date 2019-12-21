"""
A collection of specialized workflow protocols, which serve to group together
multiple individual protocol building blocks, and apply special behaviours when
executing them.

Such behaviours may include for example running the grouped together
protocols until certain conditions have been met.
"""

import copy
import json
import logging
from enum import Enum, unique
from os import path

from propertyestimator import unit
from propertyestimator.workflow.attributes import (
    InequalityMergeBehaviour,
    InputAttribute,
    OutputAttribute,
)
from propertyestimator.workflow.plugins import workflow_protocol
from propertyestimator.workflow.protocols import ProtocolGroup, ProtocolPath


@workflow_protocol()
class ConditionalGroup(ProtocolGroup):
    """A collection of protocols which are to execute until
    a given condition is met.
    """

    @unique
    class ConditionType(Enum):
        """The acceptable conditions to place on the group"""

        LessThan = "lessthan"
        GreaterThan = "greaterthan"

        @classmethod
        def has_value(cls, value):
            """Checks whether an of the enum items matches a given value.

            Parameters
            ----------
            value: str
                The value to check for.

            Returns
            ---------
            bool
                True if the enum contains the value.
            """
            return any(value == item.value for item in cls)

    class Condition:
        def __init__(self):

            self.type = ConditionalGroup.ConditionType.LessThan

            self.left_hand_value = None
            self.right_hand_value = None

        def __getstate__(self):

            return {
                "type": self.type.value,
                "left_hand_value": self.left_hand_value,
                "right_hand_value": self.right_hand_value,
            }

        def __setstate__(self, state):

            self.type = ConditionalGroup.ConditionType(state["type"])

            self.left_hand_value = state["left_hand_value"]
            self.right_hand_value = state["right_hand_value"]

        def __eq__(self, other):

            return (
                self.left_hand_value == other.left_hand_value
                and self.right_hand_value == other.right_hand_value
                and self.type == other.type
            )

        def __ne__(self, other):
            return not self.__eq__(other)

        def __str__(self):
            return f"{self.left_hand_value} {self.type} {self.right_hand_value}"

    @property
    def conditions(self):
        return self._conditions

    max_iterations = InputAttribute(
        docstring="The maximum number of iterations to run for to try and satisfy the "
        "groups conditions.",
        type_hint=int,
        default_value=100,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
    )

    current_iteration = OutputAttribute(
        docstring="The current number of iterations this group has performed while "
        "attempting to satisfy the specified conditions. This value starts "
        "from one.",
        type_hint=int,
    )

    def __init__(self, protocol_id):
        """Constructs a new ConditionalGroup
        """
        self._conditions = []
        super().__init__(protocol_id)

    def _initialize(self):
        """Initialize the protocol."""

        super(ConditionalGroup, self)._initialize()
        self.required_inputs.append(ProtocolPath("conditions"))

    def _set_schema(self, schema_value):

        conditions = None

        if ".conditions" in schema_value.inputs:
            conditions = schema_value.inputs.pop(".conditions")

            for condition in conditions:
                self.add_condition(copy.deepcopy(condition))

        super(ConditionalGroup, self)._set_schema(schema_value)

        if conditions is not None:
            schema_value.inputs[".conditions"] = conditions

    def _evaluate_condition(self, condition):
        """Evaluates whether a condition has been successfully met.

        Parameters
        ----------
        condition: ConditionalGroup.Condition
            The condition to evaluate.

        Returns
        -------
        bool
            True if the condition has been met.
        """

        if not isinstance(condition.left_hand_value, ProtocolPath):
            left_hand_value = condition.left_hand_value
        else:
            left_hand_value = self.get_value(condition.left_hand_value)

        if not isinstance(condition.right_hand_value, ProtocolPath):
            right_hand_value = condition.right_hand_value
        else:
            right_hand_value = self.get_value(condition.right_hand_value)

        if left_hand_value is None or right_hand_value is None:
            return False

        right_hand_value_correct_units = right_hand_value

        if isinstance(right_hand_value, unit.Quantity) and isinstance(
            left_hand_value, unit.Quantity
        ):
            right_hand_value_correct_units = right_hand_value.to(left_hand_value.units)

        logging.info(
            f"Evaluating condition for protocol {self.id}: "
            f"{left_hand_value} {condition.type} {right_hand_value_correct_units}"
        )

        if condition.type == self.ConditionType.LessThan:
            return left_hand_value < right_hand_value
        elif condition.type == self.ConditionType.GreaterThan:
            return left_hand_value > right_hand_value

        raise NotImplementedError()

    @staticmethod
    def _write_checkpoint(directory, current_iteration):
        """Creates a checkpoint file for this group so that it can continue
        executing where it left off if it was killed for some reason (e.g the
        worker it was running on was killed).

        Parameters
        ----------
        directory: str
            The path to the working directory of this protocol
        current_iteration: int
            The number of iterations this group has performed so far.
        """

        checkpoint_path = path.join(directory, "checkpoint.json")

        with open(checkpoint_path, "w") as file:
            json.dump({"current_iteration": current_iteration}, file)

    @staticmethod
    def _read_checkpoint(directory):
        """Creates a checkpoint file for this group so that it can continue
        executing where it left off if it was killed for some reason (e.g the
        worker it was running on was killed).

        Parameters
        ----------
        directory: str
            The path to the working directory of this protocol

        Returns
        -------
        int
            The number of iterations this group has performed so far.
        """

        current_iteration = 0
        checkpoint_path = path.join(directory, "checkpoint.json")

        if not path.isfile(checkpoint_path):
            return current_iteration

        with open(checkpoint_path, "r") as file:

            checkpoint_dictionary = json.load(file)
            current_iteration = checkpoint_dictionary["current_iteration"]

        return current_iteration

    def execute(self, directory, available_resources):
        """Executes the protocols within this groups

        Parameters
        ----------
        directory : str
            The root directory in which to run the protocols
        available_resources: ComputeResources
            The resources available to execute on.

        Returns
        -------
        bool
            True if all the protocols execute correctly.
        """

        logging.info("Starting conditional while loop: {}".format(self.id))

        should_continue = True
        self.current_iteration = self._read_checkpoint(directory)

        while should_continue:

            # Create a checkpoint file so we can pick off where
            # we left off if this execution fails due to time
            # constraints for e.g.
            self._write_checkpoint(directory, self.current_iteration)
            self.current_iteration += 1

            return_value = super(ConditionalGroup, self).execute(
                directory, available_resources
            )

            conditions_met = True

            for condition in self._conditions:

                # Check to see if we have reached our goal.
                if not self._evaluate_condition(condition):
                    conditions_met = False

            if conditions_met:

                logging.info(
                    f"Conditional while loop finished after {self.current_iteration} iterations: {self.id}"
                )
                return return_value

            if self.current_iteration >= self.max_iterations:

                raise RuntimeError(
                    f"Conditional while loop failed to converge: {self.id}"
                )

            logging.info(
                f"Conditional criteria not yet met after {self.current_iteration} iterations"
            )

    def merge(self, other):
        """Merges another ProtocolGroup with this one. The id
        of this protocol will remain unchanged.

        It is assumed that can_merge has already returned that
        these protocol groups are compatible to be merged together.

        Parameters
        ----------
        other: ConditionalGroup
            The protocol to merge into this one.
        """
        merged_ids = super(ConditionalGroup, self).merge(other)

        for condition in other.conditions:

            if isinstance(condition.left_hand_value, ProtocolPath):
                condition.left_hand_value.replace_protocol(other.id, self.id)
            if isinstance(condition.right_hand_value, ProtocolPath):
                condition.right_hand_value.replace_protocol(other.id, self.id)

            for merged_id in merged_ids:

                if isinstance(condition.left_hand_value, ProtocolPath):
                    condition.left_hand_value.replace_protocol(
                        merged_id, merged_ids[merged_id]
                    )
                if isinstance(condition.right_hand_value, ProtocolPath):
                    condition.right_hand_value.replace_protocol(
                        merged_id, merged_ids[merged_id]
                    )

            self.add_condition(condition)

        return merged_ids

    def add_condition(self, condition_to_add):
        """Adds a condition to this groups list of conditions if it
        not already in the condition list.

        Parameters
        ----------
        condition_to_add: :obj:`ConditionalGroup.Condition`
            The condition to add.
        """

        for condition in self._conditions:

            if condition == condition_to_add:
                return

        self._conditions.append(condition_to_add)

    def set_uuid(self, value):
        """Store the uuid of the calculation this protocol belongs to

        Parameters
        ----------
        value : str
            The uuid of the parent calculation.
        """
        super(ConditionalGroup, self).set_uuid(value)

        for condition in self._conditions:

            if isinstance(condition.left_hand_value, ProtocolPath):
                condition.left_hand_value.append_uuid(value)

            if isinstance(condition.right_hand_value, ProtocolPath):
                condition.right_hand_value.append_uuid(value)

    def replace_protocol(self, old_id, new_id):
        """Finds each input which came from a given protocol
         and redirects it to instead take input from a different one.

        Parameters
        ----------
        old_id : str
            The id of the old input protocol.
        new_id : str
            The id of the new input protocol.
        """
        super(ConditionalGroup, self).replace_protocol(old_id, new_id)

        for condition in self._conditions:

            if isinstance(condition.left_hand_value, ProtocolPath):
                condition.left_hand_value.replace_protocol(old_id, new_id)

            if isinstance(condition.right_hand_value, ProtocolPath):
                condition.right_hand_value.replace_protocol(old_id, new_id)

    def get_class_attribute(self, reference_path):

        if reference_path.start_protocol is None or (
            reference_path.start_protocol == self.id
            and reference_path.last_protocol == self.id
        ):

            if (
                reference_path.property_name == "conditions"
                or reference_path.property_name.find("condition_") >= 0
            ):
                return None

        return super(ConditionalGroup, self).get_class_attribute(reference_path)

    def get_value(self, reference_path):
        """Returns the value of one of this protocols parameters / inputs.

        Parameters
        ----------
        reference_path: ProtocolPath
            The path pointing to the value to return.

        Returns
        ----------
        object:
            The value of the input
        """

        if reference_path.start_protocol is None or (
            reference_path.start_protocol == self.id
            and reference_path.last_protocol == self.id
        ):

            if reference_path.property_name == "conditions":
                return self._conditions

        return super(ConditionalGroup, self).get_value(reference_path)

    def set_value(self, reference_path, value):
        """Sets the value of one of this protocols parameters / inputs.

        Parameters
        ----------
        reference_path: ProtocolPath
            The path pointing to the value to return.
        value: Any
            The value to set.
        """

        if reference_path.start_protocol is None or (
            reference_path.start_protocol == self.id
            and reference_path.last_protocol == self.id
        ):

            if reference_path.property_name == "conditions":

                self._conditions = value
                return

        super(ConditionalGroup, self).set_value(reference_path, value)

    def get_value_references(self, input_path):

        if input_path.property_name != "conditions":
            return super(ConditionalGroup, self).get_value_references(input_path)

        value_references = {}

        for index, condition in enumerate(self.conditions):

            if isinstance(condition.left_hand_value, ProtocolPath):

                source_path = ProtocolPath(
                    "conditions[{}].left_hand_value".format(index)
                )
                value_references[source_path] = condition.left_hand_value

            if isinstance(condition.right_hand_value, ProtocolPath):

                source_path = ProtocolPath(
                    "conditions[{}].right_hand_value".format(index)
                )
                value_references[source_path] = condition.right_hand_value

        return value_references
