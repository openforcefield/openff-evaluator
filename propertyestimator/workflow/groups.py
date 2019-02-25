"""
A collection of specialized workflow protocols, which serve to group together
multiple individual protocol building blocks, and apply special behaviours when
executing them.

Such behaviours may include for example running the grouped together
protocols until certain conditions have been met.
"""

import copy
import logging
from enum import Enum, unique
from os import path, makedirs

from propertyestimator.utils import graph, serialization
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow import plugins
from propertyestimator.workflow.decorators import MergeBehaviour, protocol_input
from propertyestimator.workflow.plugins import register_calculation_protocol, available_protocols
from .protocols import BaseProtocol, ProtocolPath
from .schemas import ProtocolGroupSchema


@register_calculation_protocol()
class ProtocolGroup(BaseProtocol):
    """A collection of protocols to be executed in one batch.

    This may be used for example to cluster together multiple protocols
    that will execute in a linear chain so that multiple scheduler
    execution calls are reduced into a single one.

    Additionally, a group may provide enhanced behaviour, for example
    running all protocols within the group self consistently until
    a given condition is met (e.g run a simulation until a given observable
    has converged).
    """

    @property
    def root_protocols(self):
        """List[str]: The ids of the protocols in the group which do not take
                      input from the other grouped protocols."""

        return self._root_protocols

    @property
    def execution_order(self):
        """List[str]: The ids of the protocols in the group, in the order in which
                      they will be internally executed."""

        return self._execution_order

    @property
    def dependants_graph(self):
        """Dict[str, str]: A dictionary of which stores which grouped protocols are
        dependant on other grouped protocols. Each key in the dictionary is the id of
        a grouped protocol, and each value is the id of a protocol which depends on the
        protocol by the key."""

        return self._dependants_graph

    @property
    def protocols(self):
        """Dict[str, BaseProtocol]: A dictionary of the protocols in this groups, where the dictionary
                                    key is the protocol id, and the value the protocol itself."""
        return self._protocols

    def __init__(self, protocol_id):
        """Constructs a new ProtocolGroup.
        """
        super().__init__(protocol_id)

        self._dependants_graph = {}

        self._root_protocols = []
        self._execution_order = []

        self._protocols = {}

    def _get_schema(self):

        base_schema = super(ProtocolGroup, self)._get_schema()
        # Convert the base schema to a group one.
        schema = ProtocolGroupSchema.parse_obj(base_schema.dict())

        for protocol_id in self._protocols:
            schema.grouped_protocol_schemas.append(self._protocols[protocol_id].schema)

        return schema

    def _set_schema(self, schema_value):

        super(ProtocolGroup, self)._set_schema(schema_value)

        protocols_to_create = []

        for protocol_schema in schema_value.grouped_protocol_schemas:

            if protocol_schema.id in self._protocols:

                self._protocols[protocol_schema.id].schema = protocol_schema
                continue

            # Recreate the protocol from scratch.
            protocol = available_protocols[protocol_schema.type](protocol_schema.id)
            protocol.schema = protocol_schema

            protocols_to_create.append(protocol)

        if len(protocols_to_create) > 0:
            self.add_protocols(*protocols_to_create)

    def add_protocols(self, *protocols):

        for protocol in protocols:

            if protocol.id in self._protocols:

                raise ValueError('The {} group already contains a protocol '
                                 'with id {}.'.format(self.id, protocol.id))

            self._protocols[protocol.id] = protocol
            self._dependants_graph[protocol.id] = []

        # Pull each of an individual protocols inputs up so that they
        # become a required input of the group.
        for protocol_id in self._protocols:

            protocol = self._protocols[protocol_id]

            for input_path in protocol.required_inputs:

                grouped_path = ProtocolPath.from_string(input_path.full_path)

                if grouped_path.start_protocol != protocol.id:
                    grouped_path.prepend_protocol_id(protocol.id)

                grouped_path.prepend_protocol_id(self.id)

                if grouped_path in self.required_inputs:
                    continue

                reference_values = protocol.get_value_references(input_path)

                if len(reference_values) == 0:
                    self.required_inputs.append(grouped_path)

                for source_path, reference_value in reference_values.items():

                    if reference_value.start_protocol not in self._protocols:

                        self.required_inputs.append(grouped_path)
                        continue

                    if protocol_id in self._dependants_graph[reference_value.start_protocol]:
                        continue

                    self._dependants_graph[reference_value.start_protocol].append(protocol_id)

        # Figure out the order in which grouped protocols should be executed.
        self._root_protocols = graph.find_root_nodes(self._dependants_graph)
        self._execution_order = graph.topological_sort(self._dependants_graph)

    def set_uuid(self, value):
        """Store the uuid of the calculation this protocol belongs to

        Parameters
        ----------
        value : str
            The uuid of the parent calculation.
        """
        for index in range(len(self._root_protocols)):
            self._root_protocols[index] = graph.append_uuid(self._root_protocols[index], value)

        for index in range(len(self._execution_order)):
            self._execution_order[index] = graph.append_uuid(self._execution_order[index], value)

        new_dependants_graph = {}

        for protocol_id in self._dependants_graph:

            new_protocol_id = graph.append_uuid(protocol_id, value)
            new_dependants_graph[new_protocol_id] = []

            for dependant in self._dependants_graph[protocol_id]:

                new_dependant_id = graph.append_uuid(dependant, value)
                new_dependants_graph[new_protocol_id].append(new_dependant_id)

        self._dependants_graph = new_dependants_graph

        new_protocols = {}

        for protocol_id in self._protocols:

            protocol = self._protocols[protocol_id]
            protocol.set_uuid(value)

            new_protocols[protocol.id] = protocol

        self._protocols = new_protocols
        super(ProtocolGroup, self).set_uuid(value)

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
        super(ProtocolGroup, self).replace_protocol(old_id, new_id)

        for index in range(len(self._root_protocols)):
            self._root_protocols[index] = self._root_protocols[index].replace(old_id, new_id)

        for index in range(len(self._execution_order)):
            self._execution_order[index] = self._execution_order[index].replace(old_id, new_id)

        new_dependants_graph = {}

        for protocol_id in self._dependants_graph:

            new_protocol_id = protocol_id.replace(old_id, new_id)
            new_dependants_graph[new_protocol_id] = []

            for dependant in self._dependants_graph[protocol_id]:

                new_dependant_id = dependant.replace(old_id, new_id)
                new_dependants_graph[new_protocol_id].append(new_dependant_id)

        self._dependants_graph = new_dependants_graph

        new_protocols = {}

        for protocol_id in self._protocols:

            protocol = self._protocols[protocol_id]
            protocol.replace_protocol(old_id, new_id)

            new_protocols[protocol_id.replace(old_id, new_id)] = protocol

        self._protocols = new_protocols

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

        output_dictionary = {}

        for protocol_id_to_execute in self._execution_order:

            protocol_to_execute = self._protocols[protocol_id_to_execute]
            working_directory = path.join(directory, protocol_to_execute.id)

            if not path.isdir(working_directory):
                makedirs(working_directory)

            for input_path in protocol_to_execute.required_inputs:

                value_references = protocol_to_execute.get_value_references(input_path)

                for source_path, value_reference in value_references.items():

                    if (value_reference.start_protocol == input_path.start_protocol or
                        value_reference.start_protocol == protocol_to_execute.id):

                        continue

                    value = self._protocols[value_reference.start_protocol].get_value(value_reference)
                    protocol_to_execute.set_value(source_path, value)

            return_value = protocol_to_execute.execute(working_directory, available_resources)

            if isinstance(return_value, PropertyEstimatorException):
                return return_value

            for output_path in return_value:

                output_path_prepended = ProtocolPath.from_string(output_path)

                if (output_path_prepended.start_protocol != self.id and
                    output_path_prepended.start_protocol != protocol_id_to_execute):

                    output_path_prepended.prepend_protocol_id(protocol_id_to_execute)

                if output_path_prepended.start_protocol != self.id:
                    output_path_prepended.prepend_protocol_id(self.id)

                    output_path_prepended.prepend_protocol_id(self.id)

                output_dictionary[output_path_prepended.full_path] = return_value[output_path]

        return output_dictionary

    def can_merge(self, other):
        """Determines whether this protocol group can be merged with another.

        Parameters
        ----------
        other : ProtocolGroup
            The protocol group to compare against.

        Returns
        ----------
        bool
            True if the two protocols are safe to merge.
        """

        if not super(ProtocolGroup, self).can_merge(other):
            return False

        if len(self._root_protocols) != len(other.root_protocols):
            # Only allow groups with the same number of root protocols
            # to merge.
            return False

        # Ensure that the starting points in each group can be
        # merged.
        # TODO: Is this too strict / too lenient / just right?
        for self_root_id in self._root_protocols:

            self_protocol = self._protocols[self_root_id]

            can_merge_with_root = False

            for other_root_id in other.root_protocols:

                other_protocol = other.protocols[other_root_id]

                if not self_protocol.can_merge(other_protocol):
                    continue

                can_merge_with_root = True
                break

            if not can_merge_with_root:
                return False

        return True

    def _try_merge_protocol(self, other_protocol_id, other_group, parent_ids, merged_ids):
        """Recursively inserts a protocol node into the group.

        Parameters
        ----------
        other_protocol_id : str
            The name of the other protocol to attempt to merge.
        other_group : ProtocolGroup
            The other protocol group which the protocol to merge belongs to.
        parent_ids : List[str]
            The ids of the new parents of the node to be inserted. If None,
            the protocol will be added as a new parent node.
        merged_ids : Dict[str, str]
            A map between any original protocol ids and their new merged values.
        """

        if other_protocol_id in self._dependants_graph:

            raise RuntimeError('A protocol with id {} has already been merged '
                               'into the group.'.format(other_protocol_id))

        protocol_ids = self._root_protocols if len(parent_ids) == 0 else []

        for parent_id in parent_ids:
            protocol_ids.extend(x for x in self._dependants_graph[parent_id] if x not in protocol_ids)

        protocol_to_merge = other_group.protocols[other_protocol_id]
        existing_protocol = None

        # Start by checking to see if the starting node of the calculation graph is
        # already present in the full graph.
        for protocol_id in protocol_ids:

            protocol = self._protocols[protocol_id]

            if not protocol.can_merge(protocol_to_merge):
                continue

            existing_protocol = protocol
            break

        if existing_protocol is not None:
            # Make a note that the existing node should be used in place
            # of this calculations version.

            other_group.protocols[other_protocol_id] = existing_protocol
            merged_ids[other_protocol_id] = existing_protocol.id

            for protocol_to_update in other_group.protocols:

                other_group.protocols[protocol_to_update].replace_protocol(other_protocol_id,
                                                                           existing_protocol.id)

        else:

            # Add the protocol as a new node in the graph.
            self._protocols[other_protocol_id] = protocol_to_merge
            existing_protocol = self._protocols[other_protocol_id]

            self._dependants_graph[other_protocol_id] = []

            if len(parent_ids) == 0:
                self._root_protocols.append(other_protocol_id)

            for parent_id in parent_ids:
                self._dependants_graph[parent_id].append(other_protocol_id)

        return existing_protocol.id

    def merge(self, other):
        """Merges another ProtocolGroup with this one. The id
        of this protocol will remain unchanged.

        It is assumed that can_merge has already returned that
        these protocol groups are compatible to be merged together.

        Parameters
        ----------
        other: ProtocolGroup
            The protocol to merge into this one.

        Returns
        -------
        Dict[str, str]
            A map between any original protocol ids and their new merged values.
        """

        merged_ids = super(ProtocolGroup, self).merge(other)

        other_execution_order = graph.topological_sort(other.dependants_graph)

        other_reduced_protocol_dependants = copy.deepcopy(other.dependants_graph)
        graph.apply_transitive_reduction(other_reduced_protocol_dependants)

        other_parent_protocol_ids = {}

        for protocol_id in other_execution_order:

            parent_ids = other_parent_protocol_ids.get(protocol_id) or []
            inserted_id = self._try_merge_protocol(protocol_id, other, parent_ids, merged_ids)

            for dependant in other_reduced_protocol_dependants[protocol_id]:

                if dependant not in other_parent_protocol_ids:
                    other_parent_protocol_ids[dependant] = []

                other_parent_protocol_ids[dependant].append(inserted_id)

        self._execution_order = graph.topological_sort(self._dependants_graph)

        return merged_ids

    def get_attribute_type(self, reference_path):
        """Returns the type of one of the protocol input/output attributes.

        Parameters
        ----------
        reference_path: ProtocolPath
            The path pointing to the value whose type to return.

        Returns
        ----------
        type:
            The type of the attribute.
        """

        reference_property, reference_ids = ProtocolPath.to_components(reference_path.full_path)

        if reference_path.start_protocol is None or (reference_path.start_protocol == self.id and
                                                     len(reference_ids) == 1):

            return super(ProtocolGroup, self).get_attribute_type(reference_path)

        # Make a copy of the path so we can alter it safely.
        reference_path_clone = copy.deepcopy(reference_path)

        if reference_path.start_protocol == self.id:
            reference_path_clone.pop_next_in_path()

        target_protocol_id = reference_path_clone.pop_next_in_path()

        if target_protocol_id not in self._protocols:
            raise ValueError('The reference path does not target this protocol'
                             'or any of its children.')

        return self._protocols[target_protocol_id].get_attribute_type(reference_path_clone)

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

        reference_property, reference_ids = ProtocolPath.to_components(reference_path.full_path)

        if reference_path.start_protocol is None or (reference_path.start_protocol == self.id and
                                                     len(reference_ids) == 1):

            return super(ProtocolGroup, self).get_value(reference_path)

        # Make a copy of the path so we can alter it safely.
        reference_path_clone = copy.deepcopy(reference_path)

        if reference_path.start_protocol == self.id:
            reference_path_clone.pop_next_in_path()

        target_protocol_id = reference_path_clone.pop_next_in_path()

        if target_protocol_id not in self._protocols:

            raise ValueError('The reference path does not target this protocol'
                             'or any of its children.')

        return self._protocols[target_protocol_id].get_value(reference_path_clone)

    def set_value(self, reference_path, value):
        """Sets the value of one of this protocols parameters / inputs.

        Parameters
        ----------
        reference_path: ProtocolPath
            The path pointing to the value to return.
        value: Any
            The value to set.
        """

        reference_property, reference_ids = ProtocolPath.to_components(reference_path.full_path)

        if reference_path.start_protocol is None or (reference_path.start_protocol == self.id and
                                                     len(reference_ids) == 1):

            return super(ProtocolGroup, self).set_value(reference_path, value)

        # Make a copy of the path so we can alter it safely.
        reference_path_clone = copy.deepcopy(reference_path)

        if reference_path.start_protocol == self.id:
            reference_path_clone.pop_next_in_path()

        target_protocol_id = reference_path_clone.pop_next_in_path()

        if target_protocol_id not in self._protocols:
            raise ValueError('The reference path does not target this protocol'
                             'or any of its children.')

        return self._protocols[target_protocol_id].set_value(reference_path_clone, value)

    def apply_replicator(self, replicator, template_values):
        """Applies a `ProtocolReplicator` to this groups protocols.

        Parameters
        ----------
        replicator: :obj:`ProtocolReplicator`
            The replicator to apply.
        template_values
            The values to pass to each of the replicated protocols.
        """
        for protocol_path in replicator.protocols_to_replicate:

            if protocol_path.full_path.find(self.id) < 0:
                continue

            # Start by coping the path, and removing the leading protocol ids
            # until the path starts at this group.
            protocol_path_copied = ProtocolPath.from_string(protocol_path.full_path)
            path_starting_protocol = None

            while path_starting_protocol != self.id:
                path_starting_protocol = protocol_path_copied.pop_next_in_path()

            # If the protocol to replicate is not in this group,
            # pass the call down the protocol chain.
            if protocol_path_copied.start_protocol != protocol_path_copied.last_protocol:

                self.protocols[protocol_path_copied.start_protocol].apply_replicator(replicator,
                                                                                     template_values)
                continue

            # Handle the case where the protocol to be replicated is within this group.
            for index, template_value in enumerate(template_values):

                protocol_schema = self.protocols[protocol_path_copied.start_protocol].schema
                protocol_schema.id = protocol_schema.id.replace('$index', str(index))

                protocol = plugins.available_protocols[protocol_schema.type](protocol_schema.id)
                protocol.schema = protocol_schema

                for other_path in replicator.protocols_to_replicate:

                    _, other_path_components = ProtocolPath.to_components(other_path.full_path)

                    for protocol_id_to_rename in other_path_components:

                        protocol.replace_protocol(protocol_id_to_rename,
                                                  protocol_id_to_rename.replace('$index', str(index)))

                self.protocols[protocol.id] = protocol

            self.protocols.pop(protocol_path_copied.start_protocol)


@register_calculation_protocol()
class ConditionalGroup(ProtocolGroup):
    """A collection of protocols which are to execute until
    a given condition is met.
    """
    @unique
    class ConditionType(Enum):
        """The acceptable conditions to place on the group"""
        LessThan = 'lessthan'
        GreaterThan = 'greaterthan'

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
                'type': self.type.value,
                'left_hand_value': serialization.PolymorphicDataType.serialize(self.left_hand_value),
                'right_hand_value': serialization.PolymorphicDataType.serialize(self.right_hand_value)
            }

        def __setstate__(self, state):

            self.type = ConditionalGroup.ConditionType(state['type'])

            self.left_hand_value = serialization.PolymorphicDataType.deserialize(state['left_hand_value']).value
            self.right_hand_value = serialization.PolymorphicDataType.deserialize(state['right_hand_value']).value

        def __eq__(self, other):

            return (self.left_hand_value == other.left_hand_value and
                    self.right_hand_value == other.right_hand_value and
                    self.type == other.type)

        def __ne__(self, other):
            return not self.__eq__(other)

    @protocol_input(int, merge_behavior=MergeBehaviour.GreatestValue)
    def max_iterations(self):
        """The maximum number of iterations to run for to try and satisfy the
         groups conditions."""
        pass

    @property
    def conditions(self):
        return self._conditions

    def __init__(self, protocol_id):
        """Constructs a new ConditionalGroup
        """
        super().__init__(protocol_id)

        self._max_iterations = 10
        self._conditions = []

        self.required_inputs.append(ProtocolPath('conditions'))

    def _set_schema(self, schema_value):

        conditions = None

        if '.conditions' in schema_value.inputs:
            conditions = schema_value.inputs.pop('.conditions')

            for condition in conditions.value:
                self.add_condition(copy.deepcopy(condition))

        super(ConditionalGroup, self)._set_schema(schema_value)

        if conditions is not None:
            schema_value.inputs['.conditions'] = conditions

    def _evaluate_condition(self, condition_type, left_hand_value, right_hand_value):

        if left_hand_value is None or right_hand_value is None:
            return False

        if condition_type == self.ConditionType.LessThan:
            return left_hand_value < right_hand_value
        elif condition_type == self.ConditionType.GreaterThan:
            return left_hand_value > right_hand_value

        raise NotImplementedError()

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

        logging.info('Starting conditional while loop: {}'.format(self.id))

        should_continue = True

        current_iteration = 0

        while should_continue:

            current_iteration += 1
            return_value = super(ConditionalGroup, self).execute(directory, available_resources)

            if isinstance(return_value, PropertyEstimatorException):
                # Exit on exceptions.
                return return_value

            for condition in self._conditions:

                evaluated_left_hand_value = None

                if not isinstance(condition.left_hand_value, ProtocolPath):
                    evaluated_left_hand_value = condition.left_hand_value
                else:
                    evaluated_left_hand_value = self.get_value(condition.left_hand_value)

                evaluated_right_hand_value = None

                if not isinstance(condition.right_hand_value, ProtocolPath):
                    evaluated_right_hand_value = condition.right_hand_value
                else:
                    evaluated_right_hand_value = self.get_value(condition.right_hand_value)

                # Check to see if we have reached our goal.
                if self._evaluate_condition(condition.type, evaluated_left_hand_value, evaluated_right_hand_value):

                    logging.info('Conditional while loop finished after {} iterations: {}'.format(current_iteration,
                                                                                                  self.id))
                    return return_value

            if current_iteration >= self._max_iterations:

                return PropertyEstimatorException(directory=directory,
                                                  message='Conditional while loop failed to '
                                                           'converge: {}'.format(self.id))

            logging.info('Conditional criteria not yet met after {} iterations'.format(current_iteration))

    def can_merge(self, other):
        return super(ConditionalGroup, self).can_merge(other)

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
                    condition.left_hand_value.replace_protocol(merged_id, merged_ids[merged_id])
                if isinstance(condition.right_hand_value, ProtocolPath):
                    condition.right_hand_value.replace_protocol(merged_id, merged_ids[merged_id])

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
        insertion_index = len(self._conditions) - 1

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

    def get_attribute_type(self, reference_path):

        if reference_path.start_protocol is None or (reference_path.start_protocol == self.id and
                                                     reference_path.last_protocol == self.id):

            if reference_path.property_name == 'conditions' or reference_path.property_name.find('condition_') >= 0:
                return None

        return super(ConditionalGroup, self).get_attribute_type(reference_path)

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

        if reference_path.start_protocol is None or (reference_path.start_protocol == self.id and
                                                     reference_path.last_protocol == self.id):

            if reference_path.property_name == 'conditions':
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

        if reference_path.start_protocol is None or (reference_path.start_protocol == self.id and
                                                     reference_path.last_protocol == self.id):

            if reference_path.property_name == 'conditions':

                self._conditions = value
                return

        super(ConditionalGroup, self).set_value(reference_path, value)

    def get_value_references(self, input_path):

        if input_path.property_name != 'conditions':
            return super(ConditionalGroup, self).get_value_references(input_path)

        value_references = {}

        for index, condition in enumerate(self.conditions):

            if isinstance(condition.left_hand_value, ProtocolPath):

                source_path = ProtocolPath('conditions[{}].left_hand_value'.format(index))
                value_references[source_path] = condition.left_hand_value

            if isinstance(condition.right_hand_value, ProtocolPath):

                source_path = ProtocolPath('conditions[{}].right_hand_value'.format(index))
                value_references[source_path] = condition.right_hand_value

        return value_references
