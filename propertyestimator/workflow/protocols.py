"""
A collection of specialized workflow building blocks, which when chained together,
form a larger property estimation workflow.
"""
import abc
import copy
from collections import defaultdict

from propertyestimator.attributes import Attribute, AttributeClass, PlaceholderValue
from propertyestimator.utils import graph
from propertyestimator.utils.utils import get_nested_attribute, set_nested_attribute
from propertyestimator.workflow import registered_workflow_protocols
from propertyestimator.workflow.attributes import (
    InequalityMergeBehaviour,
    InputAttribute,
    MergeBehaviour,
    OutputAttribute,
)
from propertyestimator.workflow.schemas import ProtocolGroupSchema, ProtocolSchema
from propertyestimator.workflow.utils import ProtocolPath


class Protocol(AttributeClass, abc.ABC):
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
    """

    id = Attribute(
        docstring="The unique id of this protocol.", type_hint=str, read_only=True
    )
    allow_merging = InputAttribute(
        docstring="Defines whether this protocols is allowed "
        "to merge with other protocols.",
        type_hint=bool,
        default_value=True,
    )

    @property
    def schema(self):
        """ProtocolSchema: A serializable schema for this object."""
        return self._get_schema()

    @schema.setter
    def schema(self, schema_value):
        self._set_schema(schema_value)

    @property
    def required_inputs(self):
        """list of ProtocolPath: The inputs which must be set on this protocol."""
        input_attributes = self.get_attributes(InputAttribute)
        return [ProtocolPath(x) for x in input_attributes]

    @property
    def outputs(self):
        """dict of ProtocolPath and Any: A dictionary of the outputs of this property."""

        outputs = {}

        for output_attribute in self.get_attributes(OutputAttribute):
            outputs[ProtocolPath(output_attribute)] = getattr(self, output_attribute)

        return outputs

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

                if (
                    value_reference.start_protocol is None
                    or value_reference.start_protocol == self.id
                ):

                    continue

                return_dependencies.append(value_reference)

        return return_dependencies

    def __init__(self, protocol_id):
        self._set_value("id", protocol_id)

    def _get_schema(self, schema_type=ProtocolSchema):
        """Returns the schema representation of this protocol.

        Parameters
        ----------
        schema_type: type of ProtocolSchema
            The type of schema to create.

        Returns
        -------
        schema_type
            The schema representation.
        """

        inputs = {}

        for input_path in self.required_inputs:

            if (
                input_path.protocol_path is not None
                and input_path.protocol_path != self.id
            ):
                continue

            # Always make sure to only pass a copy of the input.
            # Changing the schema should NOT change the protocol.
            inputs[input_path.full_path] = copy.deepcopy(self.get_value(input_path))

        schema = ProtocolSchema(self.id, self.__class__.__name__, inputs)
        return schema

    def _set_schema(self, schema):
        """Sets this protocols properties from a `ProtocolSchema`

        Parameters
        ----------
        schema: ProtocolSchema
            The schema to set.
        """

        # Make sure this protocol matches the schema type.
        if self.__class__.__name__ != schema.type:

            raise ValueError(
                f"The schema type {schema.type} does not match this protocol."
            )

        self._set_value("id", schema.id)

        for input_full_path in schema.inputs:

            value = copy.deepcopy(schema.inputs[input_full_path])

            input_path = ProtocolPath.from_string(input_full_path)
            self.set_value(input_path, value)

    @classmethod
    def from_schema(cls, schema):
        """Initializes a protocol from it's schema definition.

        Parameters
        ----------
        schema: ProtocolSchema
            The schema to initialize the protocol using.

        Returns
        -------
        cls
            The initialized protocol.
        """
        protocol = registered_workflow_protocols[schema.type](schema.id)
        protocol.schema = schema
        return protocol

    def set_uuid(self, value):
        """Prepend a unique identifier to this protocols id. If the id
        already has a prepended uuid, it will be overwritten by this value.

        Parameters
        ----------
        value : str
            The uuid to prepend.
        """
        if self.id.find(value) >= 0:
            return

        self._set_value("id", graph.append_uuid(self.id, value))

        for input_path in self.required_inputs:

            input_path.append_uuid(value)

            value_references = self.get_value_references(input_path)

            for value_reference in value_references.values():
                value_reference.append_uuid(value)

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

            if input_path.start_protocol is not None or (
                input_path.start_protocol != input_path.last_protocol
                and input_path.start_protocol != self.id
            ):
                continue

            value_references = self.get_value_references(input_path)

            for value_reference in value_references.values():
                value_reference.replace_protocol(old_id, new_id)

        if self.id == old_id:
            self._set_value("id", new_id)

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
            if (
                input_path.start_protocol is not None
                and input_path.start_protocol != self.id
            ):
                continue

            if not (
                input_path.start_protocol is None
                or (
                    input_path.start_protocol == input_path.last_protocol
                    and input_path.start_protocol == self.id
                )
            ):

                continue

            # If no merge behaviour flag is present (for example in the case of
            # ConditionalGroup conditions), simply assume this is handled explicitly
            # elsewhere.
            if not hasattr(
                getattr(type(self), input_path.property_name), "merge_behavior"
            ):
                continue

            inputs_to_consider.add(input_path)

        return inputs_to_consider

    def can_merge(self, other, path_replacements=None):
        """Determines whether this protocol can be merged with another.

        Parameters
        ----------
        other : :obj:`Protocol`
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

            merge_behavior = getattr(
                type(self), input_path.property_name
            ).merge_behavior

            self_value = self.get_value(input_path)
            other_value = other.get_value(input_path)

            if (
                isinstance(self_value, PlaceholderValue)
                and not isinstance(other_value, PlaceholderValue)
            ) or (
                isinstance(other_value, PlaceholderValue)
                and not isinstance(self_value, PlaceholderValue)
            ):

                # We cannot safely merge inputs when only one of the values
                # is currently known.
                return False

            if isinstance(self_value, ProtocolPath) and isinstance(
                other_value, ProtocolPath
            ):

                other_value_post_merge = ProtocolPath.from_string(other_value.full_path)

                for original_id, new_id in path_replacements:
                    other_value_post_merge.replace_protocol(original_id, new_id)

                # We cannot safely choose which value to take when the
                # values are not know ahead of time unless the two values
                # come from the exact same source.
                if self_value.protocol_path != other_value_post_merge.protocol_path:
                    return False

            elif isinstance(self_value, PlaceholderValue) and isinstance(
                other_value, PlaceholderValue
            ):
                return False

            elif (
                merge_behavior == MergeBehaviour.ExactlyEqual
                and self_value != other_value
            ):
                return False

        return True

    def merge(self, other):
        """Merges another Protocol with this one. The id
        of this protocol will remain unchanged.

        Parameters
        ----------
        other: Protocol
            The protocol to merge into this one.

        Returns
        -------
        Dict[str, str]
            A map between any original protocol ids and their new merged values.
        """

        if not self.can_merge(other):
            raise ValueError("These protocols cannot be safely merged.")

        inputs_to_consider = self._find_inputs_to_merge()

        for input_path in inputs_to_consider:

            merge_behavior = getattr(
                type(self), input_path.property_name
            ).merge_behavior

            if merge_behavior == MergeBehaviour.ExactlyEqual:
                continue

            if isinstance(self.get_value(input_path), ProtocolPath) or isinstance(
                other.get_value(input_path), ProtocolPath
            ):

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

        if (
            not isinstance(input_value, list)
            and not isinstance(input_value, tuple)
            and not isinstance(input_value, dict)
        ):

            return {}

        property_name, protocols_ids = ProtocolPath.to_components(input_path.full_path)

        return_paths = {}

        if isinstance(input_value, list) or isinstance(input_value, tuple):

            for index, list_value in enumerate(input_value):

                if not isinstance(list_value, ProtocolPath):
                    continue

                path_index = ProtocolPath(
                    property_name + "[{}]".format(index), *protocols_ids
                )
                return_paths[path_index] = list_value

        else:

            for dict_key in input_value:

                if not isinstance(input_value[dict_key], ProtocolPath):
                    continue

                path_index = ProtocolPath(
                    property_name + "[{}]".format(dict_key), *protocols_ids
                )
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

        if (
            reference_path.start_protocol is not None
            and reference_path.start_protocol != self.id
        ):
            raise ValueError(
                "The reference path {} does not point to this protocol".format(
                    reference_path
                )
            )

        if (
            reference_path.property_name.count(ProtocolPath.property_separator) >= 1
            or reference_path.property_name.find("[") > 0
        ):

            raise ValueError(
                "The expected attribute cannot be found for "
                "nested property names: {}".format(reference_path.property_name)
            )

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

        if (
            reference_path.start_protocol is not None
            and reference_path.start_protocol != self.id
        ):

            raise ValueError("The reference path does not target this protocol.")

        if reference_path.property_name is None or reference_path.property_name == "":
            raise ValueError("The reference path does specify a property to return.")

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

        if (
            reference_path.start_protocol is not None
            and reference_path.start_protocol != self.id
        ):

            raise ValueError("The reference path does not target this protocol.")

        if reference_path.property_name is None or reference_path.property_name == "":
            raise ValueError("The reference path does specify a property to set.")

        set_nested_attribute(self, reference_path.property_name, value)

    def apply_replicator(
        self,
        replicator,
        template_values,
        template_index=-1,
        template_value=None,
        update_input_references=False,
    ):
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

    @abc.abstractmethod
    def _execute(self, directory, available_resources):
        """The implementation of the public facing `execute`
        method.

        This method will be called by `execute` after all inputs
        have been validated.

        Parameters
        ----------
        directory: str
            The directory to store output data in.
        available_resources: ComputeResources
            The resources available to execute on.
        """

    def execute(self, directory, available_resources):
        """Execute the protocol.

        Parameters
        ----------
        directory: str
            The directory to store output data in.
        available_resources: ComputeResources
            The resources available to execute on.
        """
        self.validate(InputAttribute)
        self._execute(directory, available_resources)


class ProtocolGraph:
    """A graph of connected protocols.
    """

    @property
    def root_protocols(self):
        """list of str: The ids of the protocols in the group which do not
        take input from the other grouped protocols."""
        return self._root_protocols

    @property
    def dependants_graph(self):
        """dict of str and str: A dictionary of which stores which grouped protocols
        are dependant on other grouped protocols. Each key in the dictionary is the
        id of a grouped protocol, and each value is the id of a protocol which depends
        on the protocol by the key."""
        return self._dependants_graph

    def __init__(self):
        self._protocols_by_id = {}

        self._root_protocols = []
        self._dependants_graph = {}

    @staticmethod
    def _build_dependants_graph(protocols, allow_external_dependencies):
        """Builds a dictionary of key value pairs where each key
        is the id of a protocol in the graph and each value is a
        list ids of protocols which depend on this protocol.

        Parameters
        ----------
        dict of str and Protocol
            The protocols in the graph.
        allow_external_dependencies: bool
            If `False`, an exception will be raised if a protocol
            has a dependency outside of this graph.
        """
        dependants_graph = {}

        for protocol_id in protocols:
            dependants_graph[protocol_id] = set()

        for protocol in protocols:

            for dependency in protocol.dependencies:

                # Check for external dependencies.
                if dependency.start_protocol not in protocols:

                    if allow_external_dependencies:
                        continue
                    else:

                        raise ValueError(
                            f"The {dependency.start_protocol} dependency "
                            f"is outside of this graph."
                        )

                # Skip global or self dependencies.
                if dependency.is_global or dependency.start_protocol == protocol.id:
                    continue

                # Add the dependency
                dependants_graph[dependency.start_protocol].add(protocol.id)

        return dependants_graph

    def _add_protocol(
        self,
        protocol_id,
        protocols_to_add,
        dependants_graph,
        parent_protocol_ids,
        path_replacements=None,
    ):
        """Adds a protocol into the graph.

        Parameters
        ----------
        protocol_id : Protocol
            The id of the protocol to insert.
        protocols_to_add: dict of str and Protocol
            A dictionary of all of the protocols currently being
            added to the graph.
        dependants_graph: str and list of str
            The dependency graph of the protocols to add.
        parent_protocol_ids : `list` of str
            The ids of the new parents of the node to be inserted. If None,
            the protocol will be added as a new parent node.
        path_replacements: list of tuple of str, optional
            Replacements to make in any value reference protocol paths
            before comparing for equality.

        Returns
        -------
        str
            The id of the protocol which was inserted. This may not be
            the same as `protocol_id` if the protocol to insert was merged
            with an existing one.
        dict of str and str
            A mapping between all current protocol ids, and the new ids of
            protocols after the protocol has been inserted due to protocol
            merging.
        """

        # Build a list of protocols which have the same ancestors
        # as the protocols to insert. This will be used to check
        # if we are trying to add a redundant protocol to the graph.
        existing_protocols = (
            self._root_protocols if len(parent_protocol_ids) == 0 else []
        )

        for parent_protocol_id in parent_protocol_ids:

            existing_protocols.extend(
                x
                for x in self._dependants_graph[parent_protocol_id]
                if x not in existing_protocols
            )

        protocol_to_insert = protocols_to_add[protocol_id]
        existing_protocol = None

        # Start by checking to see if the starting protocol of the workflow graph is
        # already present in the full graph.
        for existing_id in existing_protocols:

            if existing_id in protocols_to_add:
                continue

            protocol = self._protocols_by_id[protocol_id]

            if not protocol.can_merge(protocol_to_insert, path_replacements):
                continue

            existing_protocol = protocol
            break

        # Store a mapping between original and merged protocols.
        merged_ids = {}

        if existing_protocol is not None:

            # Make a note that the existing protocol should be used in place
            # of this workflows version.
            protocols_to_add[protocol_id] = existing_protocol

            merged_ids = existing_protocol.merge(protocol_to_insert)
            merged_ids[protocol_to_insert.id] = existing_protocol.id

            for old_id, new_id in merged_ids.items():

                for protocol in protocols_to_add:
                    protocol.replace_protocol(old_id, new_id)

        else:

            # Add the protocol as a new protocol in the graph.
            self._protocols_by_id[protocol_id] = protocol_to_insert
            existing_protocol = self._protocols_by_id[protocol_id]

            self._dependants_graph[protocol_id] = []

            if len(parent_protocol_ids) == 0:
                self._root_protocols.append(protocol_id)

        if len(parent_protocol_ids) > 0:

            # Update the dependency graph.
            for protocol_id in dependants_graph:

                if (
                    existing_protocol.id not in dependants_graph[protocol_id]
                    or existing_protocol.id in self._dependants_graph[protocol_id]
                    or protocol_id in self._dependants_graph[existing_protocol.id]
                ):

                    continue

                self._dependants_graph[protocol_id].append(existing_protocol.id)

        return existing_protocol.id, merged_ids

    def add_protocols(self, *protocols, allow_external_dependencies):
        """Adds a set of protocols to the graph.

        Parameters
        ----------
        protocols : tuple of Protocol
            The protocols to add.
        allow_external_dependencies: bool
            If `False`, an exception will be raised if a protocol
            has a dependency outside of this graph.

        Returns
        -------
        dict of str and str
            A mapping between the original protocols and protocols which
            were merged over the course of adding the new protocols.
        """
        conflicting_ids = [x.id for x in protocols if x in self._protocols_by_id]

        # Make sure we aren't trying to add protocols with conflicting ids.
        if len(conflicting_ids) > 0:

            raise ValueError(
                f"The graph already contains protocols with ids {conflicting_ids}"
            )

        # Add the protocols to the graph
        protocols_by_id = {x.id: x for x in protocols}

        # Build a dependencies graph to check if the protocols
        # contain any cyclic dependencies
        dependants_graph = self._build_dependants_graph(
            protocols_by_id, allow_external_dependencies
        )

        if not graph.is_acyclic(dependants_graph):
            raise ValueError("The protocols to add contain cyclic dependencies.")

        # Determine the order in which the new protocols would execute.
        # This will be the order we attempt to insert them into the graph.
        protocol_execution_order = graph.topological_sort(dependants_graph)

        # Remove any redundant connections from the graph.
        reduced_graph = copy.deepcopy(dependants_graph)
        graph.apply_transitive_reduction(reduced_graph)

        parent_protocol_ids = defaultdict(set)
        # Store a mapping between original and merged protocols.
        merged_ids = {}

        for protocol_id in protocol_execution_order:

            parent_ids = parent_protocol_ids.get(protocol_id) or []
            inserted_id, new_ids = self._add_protocol(
                protocol_id, protocols, reduced_graph, parent_ids
            )

            # Keep track of any merged protocols
            merged_ids.update(new_ids)

            # Update the parent graph
            for dependant in reduced_graph[protocol_id]:
                parent_protocol_ids[dependant].add(inserted_id)

        return merged_ids


class ProtocolGroup(Protocol):
    """A group of workflow protocols to be executed in one batch.

    This may be used for example to cluster together multiple protocols
    that will execute in a linear chain so that multiple scheduler
    execution calls are reduced into a single one.

    Additionally, a group may provide enhanced behaviour, for example
    running all protocols within the group self consistently until
    a given condition is met (e.g run a simulation until a given observable
    has converged).
    """

    @property
    def required_inputs(self):
        """list of ProtocolPath: The inputs which must be set on this protocol."""
        required_inputs = super(ProtocolGroup, self).required_inputs

        # Pull each of an individual protocols inputs up so that they
        # become a required input of the group.
        for protocol in self._protocols:

            for input_path in protocol.required_inputs:

                input_path = ProtocolPath.from_string(input_path.full_path)

                if input_path.start_protocol != protocol.id:
                    input_path.prepend_protocol_id(protocol.id)

                input_path.prepend_protocol_id(self.id)

                required_inputs.append(input_path)

        return required_inputs

    @property
    def outputs(self):
        """dict of ProtocolPath and Any: A dictionary of the outputs of this property."""

        outputs = super(ProtocolGroup, self).outputs

        for protocol in self._protocols:

            for output_path in protocol.outputs:

                output_value = protocol.get_value(output_path)
                output_path = ProtocolPath.from_string(output_path.full_path)

                if output_path.start_protocol != protocol.id:
                    output_path.prepend_protocol_id(protocol.id)

                output_path.prepend_protocol_id(self.id)

                outputs[output_path] = output_value

        return outputs

    @property
    def protocols(self):
        """dict of str and Protocol: A dictionary of the protocols in
        this groups, where the dictionary key is the protocol id, and
        the value is the protocol itself.

        Notes
        -----
        This property should *not* be altered. Use `add_protocols` to
        add new protocols to the group.
        """
        return {protocol.id: protocol for protocol in self._protocols}

    def __init__(self, protocol_id):
        """Constructs a new ProtocolGroup.
        """
        super().__init__(protocol_id)

        self._protocols = []
        self._inner_graph = ProtocolGraph()

    def _get_schema(self, schema_type=ProtocolGroupSchema):

        schema = super(ProtocolGroup, self)._get_schema(schema_type)

        for protocol in self._protocols:
            schema.protocol_schemas[protocol.id] = protocol.schema

        return schema

    def _set_schema(self, schema_value):
        """
        Parameters
        ----------
        schema_value: ProtocolGroupSchema
            The schema from which this group should take its properties.
        """

        super(ProtocolGroup, self)._set_schema(schema_value)

        self._protocols = []
        self._inner_graph = ProtocolGraph()

        protocols_to_add = []

        for protocol_schema in schema_value.protocol_schemas.values():

            protocol = Protocol.from_schema(protocol_schema)
            protocols_to_add.append(protocol)

        self.add_protocols(protocols_to_add)

    def add_protocols(self, *protocols):
        """Add protocols to this group.

        Parameters
        ----------
        protocols: tuple of Protocol
            The protocols to add.
        """
        for protocol in protocols:

            if protocol.id in self.protocols:

                raise ValueError(
                    f"The {self.id} group already contains a protocol "
                    f"with id {protocol.id}."
                )

            self._protocols.append(protocol)

        self._inner_graph.add_protocols(
            *self._protocols, allow_external_dependencies=True
        )

    def set_uuid(self, value):
        """Store the uuid of the calculation this protocol belongs to

        Parameters
        ----------
        value : str
            The uuid of the parent calculation.
        """
        for protocol in self._protocols:
            protocol.set_uuid(value)

        super(ProtocolGroup, self).set_uuid(value)

        # Rebuild the inner graph
        self._inner_graph = ProtocolGraph()
        self._inner_graph.add_protocols(
            *self._protocols, allow_external_dependencies=True
        )

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
        for protocol in self._protocols:
            protocol.replace_protocol(old_id, new_id)

        super(ProtocolGroup, self).replace_protocol(old_id, new_id)

        # Rebuild the inner graph
        self._inner_graph = ProtocolGraph()
        self._inner_graph.add_protocols(
            *self._protocols, allow_external_dependencies=True
        )

    def _execute(self, directory, available_resources):
        raise NotImplementedError()

    def can_merge(self, other, path_replacements=None):

        if path_replacements is None:
            path_replacements = []

        path_replacements.append((other.id, self.id))

        if not isinstance(other, ProtocolGroup):
            return False

        if not super(ProtocolGroup, self).can_merge(other, path_replacements):
            return False

        # Ensure that the starting points in each group can be merged.
        for self_protocol in self._inner_graph.root_protocols:
            for other_protocol in other._inner_graph.root_protocols:
                if self_protocol.can_merge(other_protocol, path_replacements):
                    break
            else:
                return False

        return True

    def merge(self, other):

        assert isinstance(other, ProtocolGroup)
        merged_ids = super(ProtocolGroup, self).merge(other)

        new_merged_ids = self._inner_graph.add_protocols(
            *other.protocols.values(), allow_external_dependencies=True
        )

        # TODO: Fix.
        raise NotImplementedError()
        return merged_ids

    def _get_next_in_path(self, reference_path):
        """Returns the id of the next protocol in a protocol path,
        making sure that the targeted protocol is within this group.

        Parameters
        ----------
        reference_path: ProtocolPath
            The path being traversed.

        Returns
        -------
        str
            The id of the next protocol in the path.
        ProtocolPath
            The remainder of the path to be traversed.
        """

        # Make a copy of the path so we can alter it safely.
        reference_path_clone = copy.deepcopy(reference_path)

        if reference_path.start_protocol == self.id:
            reference_path_clone.pop_next_in_path()

        target_protocol_id = reference_path_clone.pop_next_in_path()

        if target_protocol_id not in self.protocols:

            raise ValueError(
                "The reference path does not target this protocol"
                "or any of its children."
            )

        return target_protocol_id, reference_path_clone

    def get_class_attribute(self, reference_path):

        if (
            reference_path.protocol_path is None
            or reference_path.protocol_path == self.id
        ):
            return super(ProtocolGroup, self).get_class_attribute(reference_path)

        target_protocol_id, truncated_path = self._get_next_in_path(reference_path)
        return self.protocols[target_protocol_id].get_class_attribute(truncated_path)

    def get_value(self, reference_path):

        if (
            reference_path.protocol_path is None
            or reference_path.protocol_path == self.id
        ):
            return super(ProtocolGroup, self).get_value(reference_path)

        target_protocol_id, truncated_path = self._get_next_in_path(reference_path)
        return self.protocols[target_protocol_id].get_value(truncated_path)

    def set_value(self, reference_path, value):

        if (
            reference_path.protocol_path is None
            or reference_path.protocol_path == self.id
        ):
            return super(ProtocolGroup, self).set_value(reference_path, value)

        target_protocol_id, truncated_path = self._get_next_in_path(reference_path)
        return self.protocols[target_protocol_id].set_value(truncated_path, value)

    def apply_replicator(
        self,
        replicator,
        template_values,
        template_index=-1,
        template_value=None,
        update_input_references=False,
    ):

        protocols, replication_map = replicator.apply(
            self.protocols, template_values, template_index, template_value
        )

        if (
            template_index >= 0 or template_value is not None
        ) and update_input_references is True:

            raise ValueError(
                "Specific template indices and values cannot be passed "
                "when `update_input_references` is True"
            )

        if update_input_references:
            replicator.update_references(protocols, replication_map, template_values)

        # Re-initialize the group using the replicated protocols.
        self._protocols = []
        self.add_protocols(*protocols.values())

        return replication_map
