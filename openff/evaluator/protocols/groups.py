"""
A collection of specialized workflow protocols, which serve to group together
multiple individual protocol building blocks, and apply special behaviours when
executing them.

Such behaviours may include for example running the grouped together
protocols until certain conditions have been met.
"""

import json
import logging
import typing
from enum import Enum, unique
from os import path

from openff.units import unit

from openff.evaluator.attributes import UNDEFINED, Attribute, AttributeClass
from openff.evaluator.workflow import ProtocolGroup, workflow_protocol
from openff.evaluator.workflow.attributes import (
    InequalityMergeBehavior,
    InputAttribute,
    MergeBehavior,
    OutputAttribute,
)
from openff.evaluator.workflow.utils import ProtocolPath

logger = logging.getLogger(__name__)


@workflow_protocol()
class ConditionalGroup(ProtocolGroup):
    """A collection of protocols which are to execute until
    a given condition is met.
    """

    class Condition(AttributeClass):
        """Defines a specific condition which must be met of the form
        `left_hand_value` [TYPE] `right_hand_value`, where `[TYPE]` may
        be less than or greater than.
        """

        @unique
        class Type(Enum):
            """The available condition types."""

            LessThan = "lessthan"
            GreaterThan = "greaterthan"

        left_hand_value = Attribute(
            docstring="The left-hand value to compare.",
            type_hint=typing.Union[int, float, unit.Quantity],
        )
        right_hand_value = Attribute(
            docstring="The right-hand value to compare.",
            type_hint=typing.Union[int, float, unit.Quantity],
        )

        type = Attribute(
            docstring="The right-hand value to compare.",
            type_hint=Type,
            default_value=Type.LessThan,
        )

        def __eq__(self, other):
            return (
                type(self) is type(other)
                and self.left_hand_value == other.left_hand_value
                and self.right_hand_value == other.right_hand_value
                and self.type == other.type
            )

        def __ne__(self, other):
            return not self.__eq__(other)

        def __str__(self):
            return f"{self.left_hand_value} {self.type} {self.right_hand_value}"

        def __repr__(self):
            return f"<Condition {str(self)}>"

    conditions = InputAttribute(
        docstring="The conditions which must be satisfied before"
        "the group will cleanly exit.",
        type_hint=list,
        default_value=[],
        merge_behavior=MergeBehavior.Custom,
    )

    current_iteration = OutputAttribute(
        docstring="The current number of iterations this group has performed while "
        "attempting to satisfy the specified conditions. This value starts "
        "from one.",
        type_hint=int,
    )
    max_iterations = InputAttribute(
        docstring="The maximum number of iterations to run for to try and satisfy the "
        "groups conditions.",
        type_hint=int,
        default_value=100,
        merge_behavior=InequalityMergeBehavior.LargestValue,
    )

    def __init__(self, protocol_id):
        super(ConditionalGroup, self).__init__(protocol_id)

        # We disable checkpoint, as protocols may change their inputs
        # at each iteration and hence their checkpointed outputs may
        # be invalidated.
        self._enable_checkpointing = False

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

        left_hand_value = condition.left_hand_value
        right_hand_value = condition.right_hand_value

        if isinstance(condition.left_hand_value, ProtocolPath):
            left_hand_value = self.get_value(condition.left_hand_value)
        if isinstance(condition.right_hand_value, ProtocolPath):
            right_hand_value = self.get_value(condition.right_hand_value)

        if left_hand_value == UNDEFINED or right_hand_value == UNDEFINED:
            return False

        if isinstance(right_hand_value, unit.Quantity) and isinstance(
            left_hand_value, unit.Quantity
        ):
            right_hand_value = right_hand_value.to(left_hand_value.units)

        logger.info(
            f"Evaluating condition for protocol {self.id}: "
            f"{left_hand_value} {condition.type} {right_hand_value}"
        )

        if condition.type == self.Condition.Type.LessThan:
            return left_hand_value < right_hand_value
        elif condition.type == self.Condition.Type.GreaterThan:
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

    def _execute(self, directory, available_resources):
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

        should_continue = True
        self.current_iteration = self._read_checkpoint(directory)

        # Keep a track of the original protocol schemas
        original_schemas = [x.schema for x in self._protocols]

        while should_continue:
            # Create a checkpoint file so we can pick off where
            # we left off if this execution fails due to time
            # constraints for e.g.
            self._write_checkpoint(directory, self.current_iteration)
            self.current_iteration += 1

            # Reset the protocols from their schemas - this will ensure
            # that at each iteration protocols which take their inputs from
            # other protocols in the group get their inputs updated correctly.
            for protocol, schema in zip(self._protocols, original_schemas):
                protocol.schema = schema

            super(ConditionalGroup, self)._execute(directory, available_resources)

            conditions_met = True

            for condition in self._conditions:
                # Check to see if we have reached our goal.
                if not self._evaluate_condition(condition):
                    conditions_met = False

            if conditions_met:
                logger.info(
                    f"{self.id} loop finished after {self.current_iteration} iterations"
                )
                return

            if self.current_iteration >= self.max_iterations:
                raise RuntimeError(f"{self.id} failed to converge.")

            logger.info(
                f"{self.id} criteria not yet met after {self.current_iteration} "
                f"iterations"
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

        for condition in self.conditions:
            if condition == condition_to_add:
                return

        self.conditions.append(condition_to_add)

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
