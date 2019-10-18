"""
A collection of specialized workflow building blocks, which when chained together,
form a larger property estimation workflow.
"""

import copy

from propertyestimator.utils import graph, utils
from propertyestimator.utils.utils import get_nested_attribute, set_nested_attribute
from propertyestimator.workflow.decorators import protocol_input, MergeBehaviour, InequalityMergeBehaviour, \
    protocol_output
from propertyestimator.workflow.schemas import ProtocolSchema
from propertyestimator.workflow.utils import ProtocolPath, PlaceholderInput


class BaseProtocol:
    """The base class for a protocol which would form one
    step of a larger property calculation workflow.

    A protocol may for example:

        * create the coordinates of a mixed simulation box
        * set up a bound ligand-protein system
        * build the simulation topology
        * perform an energy minimisation

    An individual protocol may require a set of inputs, which may either be
    set as constants

    >>> from propertyestimator.protocols.simulation import RunOpenMMSimulation
    >>>
    >>> npt_equilibration = RunOpenMMSimulation('npt_equilibration')
    >>> npt_equilibration.ensemble = RunOpenMMSimulation.Ensemble.NPT

    or from the output of another protocol, pointed to by a ProtocolPath

    >>> npt_production = RunOpenMMSimulation('npt_production')
    >>> # Use the coordinate file output by the npt_equilibration protocol
    >>> # as the input to the npt_production protocol
    >>> npt_production.input_coordinate_file = ProtocolPath('output_coordinate_file',
    >>>                                                     npt_equilibration.id)

    In this way protocols may be chained together, thus defining a larger property
    calculation workflow from simple, reusable building blocks.

    .. warning:: This class is still heavily under development and is subject to
                 rapid changes.
    """

    @property
    def id(self):
        """str: The unique id of this protocol."""
        return self._id

    @property
    def schema(self):
        """ProtocolSchema: A serializable schema for this object."""
        return self._get_schema()

    @schema.setter
    def schema(self, schema_value):
        self._set_schema(schema_value)

    @property
    def dependencies(self):
        """list of ProtocolPath: A list of pointers to the protocols which this
        protocol takes input from.
        """

        return_dependencies = []

        for input_path in self.required_inputs:

            value_references = self.get_value_references(input_path)

            if len(value_references) == 0:
                continue

            for value_reference in value_references.values():

                if value_reference in return_dependencies:
                    continue

                if (value_reference.start_protocol is None or
                    value_reference.start_protocol == self.id):

                    continue

                return_dependencies.append(value_reference)

        return return_dependencies

    allow_merging = protocol_input(docstring='Defines whether this protocols is allowed '
                                             'to merge with other protocols.',
                                   type_hint=bool,
                                   default_value=True)

    def __init__(self, protocol_id):

        # A unique identifier for this node.
        self._id = protocol_id

        # Defines whether a protocol is allowed to try and merge with other identical ones.
        self._allow_merging = True

        self.provided_outputs = []
        self.required_inputs = []

        self._initialize()

    def execute(self, directory, available_resources):
        """ Execute the protocol.

        Protocols may be chained together by passing the output
        of previous protocols as input to the current one.

        Parameters
        ----------
        directory: str
            The directory to store output data in.
        available_resources: ComputeResources
            The resources available to execute on.

        Returns
        ----------
        Dict[str, Any]
            The output of the execution.
        """

        return self._get_output_dictionary()

    def _initialize(self):
        """Initialize the protocol."""

        # Find the required inputs and outputs.
        self.provided_outputs = []
        self.required_inputs = []

        output_attributes = utils.find_types_with_decorator(type(self), protocol_output)
        input_attributes = utils.find_types_with_decorator(type(self), protocol_input)

        for output_attribute in output_attributes:
            self.provided_outputs.append(ProtocolPath(output_attribute))

        for input_attribute in input_attributes:
            self.required_inputs.append(ProtocolPath(input_attribute))

        # The directory in which to execute the protocol.
        self.directory = None

    def _get_schema(self):
        """Returns this protocols properties (i.e id and parameters)
        as a ProtocolSchema

        Returns
        -------
        ProtocolSchema
            The schema representation.
        """

        schema = ProtocolSchema()

        schema.id = self.id
        schema.type = type(self).__name__

        for input_path in self.required_inputs:

            if not (input_path.start_protocol is None or (input_path.start_protocol == self.id and
                                                          input_path.start_protocol == input_path.last_protocol)):

                continue

            # Always make sure to only pass a copy of the input. Changing the schema
            # should NOT change the protocol.
            schema.inputs[input_path.full_path] = copy.deepcopy(self.get_value(input_path))

        return schema

    def _set_schema(self, schema_value):
        """Sets this protocols properties (i.e id and parameters)
        from a ProtocolSchema

        Parameters
        ----------
        schema_value: ProtocolSchema
            The schema which will describe this protocol.
        """
        self._id = schema_value.id

        if type(self).__name__ != schema_value.type:
            # Make sure this object is the correct type.
            raise ValueError('Cannot convert a {} protocol to a {}.'
                             .format(str(type(self)), schema_value.type))

        for input_full_path in schema_value.inputs:

            value = copy.deepcopy(schema_value.inputs[input_full_path])

            input_path = ProtocolPath.from_string(input_full_path)
            self.set_value(input_path, value)

    def _get_output_dictionary(self):
        """Builds a dictionary of the output property names and their values.

        Returns
        -------
        Dict[str, Any]
            A dictionary whose keys are the output property names, and the
            values their associated values.
        """

        return_dictionary = {}

        for output_path in self.provided_outputs:
            return_dictionary[output_path.full_path] = self.get_value(output_path)

        return return_dictionary

    def set_uuid(self, value):
        """Store the uuid of the calculation this protocol belongs to

        Parameters
        ----------
        value : str
            The uuid of the parent calculation.
        """
        if self.id.find(value) >= 0:
            return

        self._id = graph.append_uuid(self.id, value)

        for input_path in self.required_inputs:

            input_path.append_uuid(value)

            value_references = self.get_value_references(input_path)

            for value_reference in value_references.values():
                value_reference.append_uuid(value)

        for output_path in self.provided_outputs:
            output_path.append_uuid(value)

    def replace_protocol(self, old_id, new_id):
        """Finds each input which came from a given protocol
         and redirects it to instead take input from a new one.

        Notes
        -----
        This method is mainly intended to be used only when merging
        multiple protocols into one.

        Parameters
        ----------
        old_id : str
            The id of the old input protocol.
        new_id : str
            The id of the new input protocol.
        """

        for input_path in self.required_inputs:

            input_path.replace_protocol(old_id, new_id)

            if input_path.start_protocol is not None or (input_path.start_protocol != input_path.last_protocol and
                                                         input_path.start_protocol != self.id):
                continue

            value_references = self.get_value_references(input_path)

            for value_reference in value_references.values():
                value_reference.replace_protocol(old_id, new_id)

        for output_path in self.provided_outputs:
            output_path.replace_protocol(old_id, new_id)

        if self._id == old_id:
            self._id = new_id

    def _find_inputs_to_merge(self):
        """Returns a list of those inputs which should
        be considered when attempting to merge two different
        protocols of the same type.

        Returns
        -------
        set of ProtocolPath
            References to those inputs which should be
            considered.
        """
        inputs_to_consider = set()

        for input_path in self.required_inputs:

            # Do not consider paths that point to child (e.g grouped) protocols.
            # These should be handled by the container classes themselves.
            if (input_path.start_protocol is not None and
                input_path.start_protocol != self.id):
                continue

            if not (input_path.start_protocol is None or (
                    input_path.start_protocol == input_path.last_protocol and
                    input_path.start_protocol == self.id)):

                continue

            # If no merge behaviour flag is present (for example in the case of
            # ConditionalGroup conditions), simply assume this is handled explicitly
            # elsewhere.
            if not hasattr(getattr(type(self), input_path.property_name), 'merge_behavior'):
                continue

            inputs_to_consider.add(input_path)

        return inputs_to_consider

    def can_merge(self, other, path_replacements=None):
        """Determines whether this protocol can be merged with another.

        Parameters
        ----------
        other : :obj:`BaseProtocol`
            The protocol to compare against.
        path_replacements: list of tuple of str, optional
            Replacements to make in any value reference protocol paths
            before comparing for equality.

        Returns
        ----------
        bool
            True if the two protocols are safe to merge.
        """

        if not self.allow_merging or not isinstance(self, type(other)):
            return False

        if path_replacements is None:
            path_replacements = []

        inputs_to_consider = self._find_inputs_to_merge()

        for input_path in inputs_to_consider:

            # Do a quick sanity check that the other protocol
            # does in fact also require this input.
            if input_path not in other.required_inputs:
                return False

            merge_behavior = getattr(type(self), input_path.property_name).merge_behavior

            self_value = self.get_value(input_path)
            other_value = other.get_value(input_path)

            if ((isinstance(self_value, PlaceholderInput) and not isinstance(other_value, PlaceholderInput)) or
                (isinstance(other_value, PlaceholderInput) and not isinstance(self_value, PlaceholderInput))):

                # We cannot safely merge inputs when only one of the values
                # is currently known.
                return False

            if merge_behavior == MergeBehaviour.ExactlyEqual:

                if self_value != other_value:
                    return False

            elif (isinstance(self_value, ProtocolPath) or
                  isinstance(other_value, ProtocolPath)):

                other_value_post_merge = ProtocolPath.from_string(other_value.full_path)

                for original_id, new_id in path_replacements:
                    other_value_post_merge.replace_protocol(original_id, new_id)

                # We cannot safely choose which value to take when the
                # values are not know ahead of time unless the two values
                # come from the exact same source.
                if self_value.protocol_path != other_value_post_merge.protocol_path:
                    return False

            elif (isinstance(self_value, PlaceholderInput) or
                  isinstance(other_value, PlaceholderInput)):

                return False

        return True

    def merge(self, other):
        """Merges another BaseProtocol with this one. The id
        of this protocol will remain unchanged.

        It is assumed that can_merge has already returned that
        these protocols are compatible to be merged together.

        Parameters
        ----------
        other: BaseProtocol
            The protocol to merge into this one.

        Returns
        -------
        Dict[str, str]
            A map between any original protocol ids and their new merged values.
        """

        if not self.can_merge(other):
            raise ValueError('These protocols can not be safely merged.')

        inputs_to_consider = self._find_inputs_to_merge()

        for input_path in inputs_to_consider:

            merge_behavior = getattr(type(self), input_path.property_name).merge_behavior

            if merge_behavior == MergeBehaviour.ExactlyEqual:
                continue

            if (isinstance(self.get_value(input_path), ProtocolPath) or
                isinstance(other.get_value(input_path), ProtocolPath)):

                continue

            if merge_behavior == InequalityMergeBehaviour.SmallestValue:
                value = min(self.get_value(input_path), other.get_value(input_path))
            elif merge_behavior == InequalityMergeBehaviour.LargestValue:
                value = max(self.get_value(input_path), other.get_value(input_path))
            else:
                raise NotImplementedError()

            self.set_value(input_path, value)

        return {}

    def get_value_references(self, input_path):
        """Returns a dictionary of references to the protocols which one of this
        protocols inputs (specified by `input_path`) takes its value from.

        Notes
        -----
        Currently this method only functions correctly for an input value which
        is either currently a :obj:`ProtocolPath`, or a `list` / `dict` which contains
        at least one :obj:`ProtocolPath`.

        Parameters
        ----------
        input_path: :obj:`propertyestimator.workflow.utils.ProtocolPath`
            The input value to check.

        Returns
        -------
        dict of ProtocolPath and ProtocolPath
            A dictionary of the protocol paths that the input targeted by `input_path` depends upon.
        """
        input_value = self.get_value(input_path)

        if isinstance(input_value, ProtocolPath):
            return {input_path: input_value}

        if (not isinstance(input_value, list) and
            not isinstance(input_value, tuple) and
            not isinstance(input_value, dict)):

            return {}

        property_name, protocols_ids = ProtocolPath.to_components(input_path.full_path)

        return_paths = {}

        if isinstance(input_value, list) or isinstance(input_value, tuple):

            for index, list_value in enumerate(input_value):

                if not isinstance(list_value, ProtocolPath):
                    continue

                path_index = ProtocolPath(property_name + '[{}]'.format(index), *protocols_ids)
                return_paths[path_index] = list_value

        else:

            for dict_key in input_value:

                if not isinstance(input_value[dict_key], ProtocolPath):
                    continue

                path_index = ProtocolPath(property_name + '[{}]'.format(dict_key), *protocols_ids)
                return_paths[path_index] = input_value[dict_key]

        return return_paths

    def get_class_attribute(self, reference_path):
        """Returns one of this protocols, or any of its children's,
        attributes directly (rather than its value).

        Parameters
        ----------
        reference_path: ProtocolPath
            The path pointing to the attribute to return.

        Returns
        ----------
        object:
            The class attribute.
        """

        if reference_path.start_protocol is not None and reference_path.start_protocol != self.id:
            raise ValueError('The reference path {} does not point to this protocol'.format(reference_path))

        if (reference_path.property_name.count(ProtocolPath.property_separator) >= 1 or
            reference_path.property_name.find('[') > 0):

            raise ValueError('The expected attribute cannot be found for '
                             'nested property names: {}'.format(reference_path.property_name))

        return getattr(type(self), reference_path.property_name)

    def get_value(self, reference_path):
        """Returns the value of one of this protocols inputs / outputs.

        Parameters
        ----------
        reference_path: ProtocolPath
            The path pointing to the value to return.

        Returns
        ----------
        Any:
            The value of the input / output
        """

        if (reference_path.start_protocol is not None and
            reference_path.start_protocol != self.id):

            raise ValueError('The reference path does not target this protocol.')

        if reference_path.property_name is None or reference_path.property_name == '':
            raise ValueError('The reference path does specify a property to return.')

        return get_nested_attribute(self, reference_path.property_name)

    def set_value(self, reference_path, value):
        """Sets the value of one of this protocols inputs.

        Parameters
        ----------
        reference_path: ProtocolPath
            The path pointing to the value to return.
        value: Any
            The value to set.
        """

        if (reference_path.start_protocol is not None and
            reference_path.start_protocol != self.id):

            raise ValueError('The reference path does not target this protocol.')

        if reference_path.property_name is None or reference_path.property_name == '':
            raise ValueError('The reference path does specify a property to set.')

        if reference_path in self.provided_outputs:
            raise ValueError('Output values cannot be set by this method.')

        set_nested_attribute(self, reference_path.property_name, value)

    def apply_replicator(self, replicator, template_values, template_index=-1,
                         template_value=None, update_input_references=False):
        """Applies a `ProtocolReplicator` to this protocol. This method
        should clone any protocols whose id contains the id of the
        replicator (in the format `$(replicator.id)`).

        Parameters
        ----------
        replicator: ProtocolReplicator
            The replicator to apply.
        template_values: list of Any
            A list of the values which will be inserted
            into the newly replicated protocols.

            This parameter is mutually exclusive with
            `template_index` and `template_value`
        template_index: int, optional
            A specific value which should be used for any
            protocols flagged as to be replicated by the
            replicator. This option is mainly used when
            replicating children of an already replicated
            protocol.

            This parameter is mutually exclusive with
            `template_values` and must be set along with
            a `template_value`.
        template_value: Any, optional
            A specific index which should be used for any
            protocols flagged as to be replicated by the
            replicator. This option is mainly used when
            replicating children of an already replicated
            protocol.

            This parameter is mutually exclusive with
            `template_values` and must be set along with
            a `template_index`.
        update_input_references: bool
            If true, any protocols which take their input from a protocol
            which was flagged for replication will be updated to take input
            from the actually replicated protocol. This should only be set
            to true if this protocol is not nested within a workflow or a
            protocol group.

            This option cannot be used when a specific `template_index` or
            `template_value` is providied.

        Returns
        -------
        dict of ProtocolPath and list of tuple of ProtocolPath and int
            A dictionary of references to all of the protocols which have
            been replicated, with keys of original protocol ids. Each value
            is comprised of a list of the replicated protocol ids, and their
            index into the `template_values` array.
        """
        return {}
