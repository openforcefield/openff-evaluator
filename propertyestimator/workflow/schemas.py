"""
A collection of schemas which represent elements of a property calculation workflow.
"""
import re

from propertyestimator.attributes import UNDEFINED
from propertyestimator.attributes.typing import is_type_subclass_of_type
from propertyestimator.forcefield import ParameterGradient
from propertyestimator.storage.attributes import StorageAttribute
from propertyestimator.storage.data import BaseStoredData
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.serialization import TypedBaseModel
from propertyestimator.workflow.attributes import InputAttribute
from propertyestimator.workflow.plugins import registered_workflow_protocols
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue


class ProtocolSchema(TypedBaseModel):
    """A json serializable representation of a workflow protocol.
    """

    def __init__(self):
        """Constructs a new ProtocolSchema object.
        """
        self.id = None
        self.type = None

        self.inputs = {}

    def __getstate__(self):

        return {"id": self.id, "type": self.type, "inputs": self.inputs}

    def __setstate__(self, state):

        self.id = state["id"]
        self.type = state["type"]

        self.inputs = state["inputs"]


class ProtocolGroupSchema(ProtocolSchema):
    """A json serializable representation of a workflow protocol
    group.
    """

    def __init__(self):
        """Constructs a new ProtocolGroupSchema object.
        """
        super().__init__()

        self.grouped_protocol_schemas = []

    def __getstate__(self):

        state = super(ProtocolGroupSchema, self).__getstate__()
        state.update({"grouped_protocol_schemas": self.grouped_protocol_schemas})

        return state

    def __setstate__(self, state):

        super(ProtocolGroupSchema, self).__setstate__(state)
        self.grouped_protocol_schemas = state["grouped_protocol_schemas"]


class ProtocolReplicator(TypedBaseModel):
    """A protocol replicator contains the information necessary to replicate
    parts of a property estimation workflow.

    Any protocol whose id includes `$(replicator.id)` (where `replicator.id` is the
    id of a replicator) will be cloned for each value present in `template_values`.
    Protocols that are being replicated will also have any ReplicatorValue inputs replaced
    with the actual value taken from `template_values`.

    When the protocol is replicated, the `$(replicator.id)` placeholder in the protocol
    id will be replaced an integer which corresponds to the index of a value in the
    `template_values` array.

    Any protocols which take input from a replicated protocol will be updated to
    instead take a list of value, populated by the outputs of the replicated
    protocols.

    Notes
    -----
        * The `template_values` property must be a list of either constant values,
          or `ProtocolPath` objects which take their value from the `global` scope.
        * If children of replicated protocols are also flagged as to be replicated,
          they will only have their ids changed to match the index of the parent
          protocol, as opposed to being fully replicated.
    """

    @property
    def placeholder_id(self):
        """The string which protocols to be replicated should include in
        their ids."""
        return f"$({self.id})"

    def __init__(self, replicator_id=""):
        """Constructs a new ProtocolReplicator object.

        Parameters
        ----------
        replicator_id: str
            The id of this replicator.
        """
        self.id = replicator_id
        self.template_values = None

    def __getstate__(self):

        return {"id": self.id, "template_values": self.template_values}

    def __setstate__(self, state):

        self.id = state["id"]
        self.template_values = state["template_values"]

    def apply(
        self, protocols, template_values=None, template_index=-1, template_value=None
    ):
        """Applies this replicator to the provided set of protocols and any of
        their children.

        This protocol should be followed by a call to `update_references`
        to ensure that all protocols which take their input from a replicated
        protocol get correctly updated.

        Parameters
        ----------
        protocols: dict of str and WorkflowProtocol
            The protocols to apply the replicator to.
        template_values: list of Any
            A list of the values which will be inserted
            into the newly replicated protocols.

            This parameter is mutually exclusive with
            `template_index` and `template_value`
        template_index: int, optional
            A specific value which should be used for any
            protocols flagged as to be replicated by this
            replicator. This option is mainly used when
            replicating children of an already replicated
            protocol.

            This parameter is mutually exclusive with
            `template_values` and must be set along with
            a `template_value`.
        template_value: Any, optional
            A specific index which should be used for any
            protocols flagged as to be replicated by this
            replicator. This option is mainly used when
            replicating children of an already replicated
            protocol.

            This parameter is mutually exclusive with
            `template_values` and must be set along with
            a `template_index`.

        Returns
        -------
        dict of str and WorkflowProtocol
            The replicated protocols.
        dict of ProtocolPath and list of tuple of ProtocolPath and int
            A dictionary of references to all of the protocols which have
            been replicated, with keys of original protocol ids. Each value
            is comprised of a list of the replicated protocol ids, and their
            index into the `template_values` array.
        """

        if (
            template_values is not None
            and (template_index >= 0 or template_value is not None)
        ) or (
            template_values is None and (template_index < 0 or template_value is None)
        ):

            raise ValueError(
                f"Either the template values array must be set, or a specific "
                f"template index and value must be passed."
            )

        replicated_protocols = {}
        replicated_protocol_map = {}

        for protocol_id, protocol in protocols.items():

            should_replicate = self.placeholder_id in protocol_id

            # If this protocol should not be directly replicated then try and
            # replicate any child protocols...
            if not should_replicate:

                replicated_protocols[protocol_id] = protocol

                if template_index is not None and template_index >= 0:
                    # Make sure to include children of replicated protocols in the
                    # map to ensure correct behaviour when updating children of replicated
                    # protocols which have the replicator id in their name, and take input
                    # from another child protocol which doesn't have the replicator id in
                    # its name.
                    if ProtocolPath("", protocol_id) not in replicated_protocol_map:
                        replicated_protocol_map[ProtocolPath("", protocol_id)] = []

                    replicated_protocol_map[ProtocolPath("", protocol_id)].append(
                        (ProtocolPath("", protocol_id), template_index)
                    )

                self._apply_to_protocol_children(
                    protocol,
                    replicated_protocol_map,
                    template_values,
                    template_index,
                    template_value,
                )

                continue

            # ..otherwise, we need to replicate this protocol.
            replicated_protocols.update(
                self._apply_to_protocol(
                    protocol,
                    replicated_protocol_map,
                    template_values,
                    template_index,
                    template_value,
                )
            )

        return replicated_protocols, replicated_protocol_map

    def _apply_to_protocol(
        self,
        protocol,
        replicated_protocol_map,
        template_values=None,
        template_index=-1,
        template_value=None,
    ):

        replicated_protocol_map[ProtocolPath("", protocol.id)] = []
        replicated_protocols = {}

        template_values_dict = {template_index: template_value}

        if template_values is not None:

            template_values_dict = {
                index: template_value
                for index, template_value in enumerate(template_values)
            }

        for index, template_value in template_values_dict.items():

            protocol_schema = protocol.schema
            protocol_schema.id = protocol_schema.id.replace(
                self.placeholder_id, str(index)
            )

            replicated_protocol = registered_workflow_protocols[protocol_schema.type](
                protocol_schema.id
            )
            replicated_protocol.schema = protocol_schema

            replicated_protocol_map[ProtocolPath("", protocol.id)].append(
                (ProtocolPath("", replicated_protocol.id), index)
            )

            # Pass the template values to any inputs which require them.
            for required_input in replicated_protocol.required_inputs:

                input_value = replicated_protocol.get_value(required_input)

                if not isinstance(input_value, ReplicatorValue):
                    continue

                elif input_value.replicator_id != self.id:

                    input_value.replicator_id = input_value.replicator_id.replace(
                        self.placeholder_id, str(index)
                    )
                    continue

                replicated_protocol.set_value(required_input, template_value)

            self._apply_to_protocol_children(
                replicated_protocol,
                replicated_protocol_map,
                None,
                index,
                template_value,
            )

            replicated_protocols[replicated_protocol.id] = replicated_protocol

        return replicated_protocols

    def _apply_to_protocol_children(
        self,
        protocol,
        replicated_protocol_map,
        template_values=None,
        template_index=-1,
        template_value=None,
    ):

        replicated_child_ids = protocol.apply_replicator(
            self, template_values, template_index, template_value
        )

        # Append the id of this protocols to any replicated child protocols.
        for child_id, replicated_ids in replicated_child_ids.items():

            child_id.prepend_protocol_id(protocol.id)

            for replicated_id, _ in replicated_ids:
                replicated_id.prepend_protocol_id(protocol.id)

            replicated_protocol_map.update(replicated_child_ids)

    def update_references(self, protocols, replication_map, template_values):
        """Redirects the input references of protocols to the replicated
        versions.

        Parameters
        ----------
        protocols: dict of str and WorkflowProtocol
            The protocols which have had this replicator applied
            to them.
        replication_map: dict of ProtocolPath and list of tuple of ProtocolPath and int
            A dictionary of references to all of the protocols which have
            been replicated, with keys of original protocol ids. Each value
            is comprised of a list of the replicated protocol ids, and their
            index into the `template_values` array.
        template_values: List of Any
            A list of the values which will be inserted
            into the newly replicated protocols.
        """

        inverse_replication_map = {}

        for original_id, replicated_ids in replication_map.items():
            for replicated_id, index in replicated_ids:
                inverse_replication_map[replicated_id] = (original_id, index)

        for protocol_id, protocol in protocols.items():

            # Look at each of the protocols inputs and see if its value is either a ProtocolPath,
            # or a list of ProtocolPath's.
            for required_input in protocol.required_inputs:

                all_value_references = protocol.get_value_references(required_input)
                replicated_value_references = {}

                for source_path, value_reference in all_value_references.items():

                    if self.placeholder_id not in value_reference.full_path:
                        continue

                    replicated_value_references[source_path] = value_reference

                # If this protocol does not take input from one of the replicated protocols,
                # then we are done.
                if len(replicated_value_references) == 0:
                    continue

                for source_path, value_reference in replicated_value_references.items():

                    full_source_path = ProtocolPath.from_string(source_path.full_path)
                    full_source_path.prepend_protocol_id(protocol_id)

                    # If the protocol was not itself replicated by this replicator, its value
                    # is set to a list containing references to all newly replicated protocols.
                    # Otherwise, the value will be set to a reference to just the protocol which
                    # was replicated using the same index.
                    value_source = [
                        ProtocolPath.from_string(
                            value_reference.full_path.replace(
                                self.placeholder_id, str(index)
                            )
                        )
                        for index in range(len(template_values))
                    ]

                    for replicated_id, map_tuple in inverse_replication_map.items():

                        original_id, replicated_index = map_tuple

                        if (
                            full_source_path.protocol_path
                            != replicated_id.protocol_path
                        ):
                            continue

                        value_source = ProtocolPath.from_string(
                            value_reference.full_path.replace(
                                self.placeholder_id, str(replicated_index)
                            )
                        )

                        break

                    # Replace the input value with a list of ProtocolPath's that point to
                    # the newly generated protocols.
                    protocol.set_value(source_path, value_source)


class WorkflowSchema(TypedBaseModel):
    """Outlines the workflow which should be followed when calculating
    a certain property.
    """

    def __init__(self, property_type=None):
        """Constructs a new WorkflowSchema object.

        Parameters
        ----------
        property_type: str
            The type of property which this workflow aims to estimate.
        """
        self.property_type = property_type
        self.id = None

        self.protocols = {}
        self.replicators = []

        self.final_value_source = None
        self.gradients_sources = []

        self.outputs_to_store = {}

    def __getstate__(self):

        return {
            "property_type": self.property_type,
            "id": self.id,
            "protocols": self.protocols,
            "replicators": self.replicators,
            "final_value_source": self.final_value_source,
            "gradients_sources": self.gradients_sources,
            "outputs_to_store": self.outputs_to_store,
        }

    def __setstate__(self, state):

        self.property_type = state["property_type"]
        self.id = state["id"]

        self.protocols = state["protocols"]
        self.replicators = state["replicators"]

        self.final_value_source = state["final_value_source"]
        self.gradients_sources = state["gradients_sources"]

        self.outputs_to_store = state["outputs_to_store"]

    def _find_protocols_to_be_replicated(self, replicator, protocols=None):
        """Finds all protocols which have been flagged to be replicated
        by a specified replicator.

        Parameters
        ----------
        replicator: ProtocolReplicator
            The replicator of interest.
        protocols: dict of str and ProtocolSchema or list of ProtocolSchema, optional
            The protocols to search through. If None, then
            all protocols in this schema will be searched.

        Returns
        -------
        list of str
            The ids of the protocols to be replicated by the specified replicator
        """

        if protocols is None:
            protocols = self.protocols

        if isinstance(protocols, list):
            protocols = {protocol.id: protocol for protocol in protocols}

        protocols_to_replicate = []

        for protocol_id, protocol in protocols.items():

            if protocol_id.find(replicator.placeholder_id) >= 0:
                protocols_to_replicate.append(protocol_id)

            # Search through any children
            if not isinstance(protocol, ProtocolGroupSchema):
                continue

            protocols_to_replicate.extend(
                self._find_protocols_to_be_replicated(
                    replicator, protocol.grouped_protocol_schemas
                )
            )

        return protocols_to_replicate

    def _get_unreplicated_path(self, protocol_path):
        """Checks to see if the protocol pointed to by this path will only
        exist after a replicator has been applied, and if so, returns a
        path to the unreplicated protocol.

        Parameters
        ----------
        protocol_path: ProtocolPath
            The path to convert to an unreplicated path.

        Returns
        -------
        ProtocolPath
            The path which should point to only unreplicated protocols
        """

        full_unreplicated_path = str(protocol_path.full_path)

        for replicator in self.replicators:

            if replicator.placeholder_id in full_unreplicated_path:
                continue

            protocols_to_replicate = self._find_protocols_to_be_replicated(replicator)

            for protocol_id in protocols_to_replicate:

                match_pattern = re.escape(
                    protocol_id.replace(replicator.placeholder_id, r"\d+")
                )
                match_pattern = match_pattern.replace(re.escape(r"\d+"), r"\d+")

                full_unreplicated_path = re.sub(
                    match_pattern, protocol_id, full_unreplicated_path
                )

        return ProtocolPath.from_string(full_unreplicated_path)

    @staticmethod
    def _get_unnested_protocol_path(protocol_path):
        """Returns a protocol path whose nested property name
        has been truncated to only include the top level name,
        e.g:

        `some_protocol_id.value.uncertainty` would be truncated to `some_protocol_id.value`

        and

        `some_protocol_id.value[1]` would be truncated to `some_protocol_id.value`

        Parameters
        ----------
        protocol_path: ProtocolPath
            The path to truncate.

        Returns
        -------
        ProtocolPath
            The truncated path.
        """
        property_name, protocol_ids = ProtocolPath.to_components(
            protocol_path.full_path
        )

        # Remove any nested property names from the path
        if protocol_path.property_name.find(".") >= 0:
            property_name = property_name.split(".")[0]

        # Remove any array indices from the path
        if protocol_path.property_name.find("[") >= 0:
            property_name = property_name.split("[")[0]

        return ProtocolPath(property_name, *protocol_ids)

    def replace_protocol_types(self, protocol_replacements, protocol_group_schema=None):
        """Replaces protocols with given types with other protocols
        of specified replacements. This is useful when replacing
        the default protocols with custom ones, or swapping out base
        protocols with actual implementations

        Warnings
        --------
        This method is NOT fully implemented and is likely to fail in
        all but a few specific cases. This method should be used with
        extreme caution.

        Parameters
        ----------
        protocol_replacements: dict of str and str, None
            A dictionary with keys of the types of protocols which should be replaced
            with those protocols named by the values.
        protocol_group_schema: ProtocolGroupSchema
            The protocol group to apply the replacements to. This
            is mainly used when applying this method recursively.
        """

        if protocol_replacements is None:
            return

        if protocol_group_schema is None:
            protocol_schemas = self.protocols
        else:
            protocol_schemas = protocol_group_schema.grouped_protocol_schemas

        for protocol_schema_key in protocol_schemas:

            protocol_schema = protocol_schemas[protocol_schema_key]

            if protocol_schema.type not in protocol_replacements:
                continue

            protocol = registered_workflow_protocols[protocol_schema.type](
                protocol_schema.id
            )
            protocol.schema = protocol_schema

            new_protocol = registered_workflow_protocols[
                protocol_replacements[protocol_schema.type]
            ](protocol_schema.id)

            for input_path in new_protocol.required_inputs:

                if input_path not in protocol.required_inputs:
                    continue

                value = protocol.get_value(input_path)
                new_protocol.set_value(input_path, value)

            protocol_schemas[protocol_schema_key] = new_protocol.schema

            if isinstance(protocol_schemas[protocol_schema_key], ProtocolGroupSchema):
                self.replace_protocol_types(
                    protocol_replacements, protocol_schemas[protocol_schema_key]
                )

    def _validate_replicators(self):

        for replicator in self.replicators:

            assert replicator.id is not None and len(replicator.id) > 0

            # if len(replicator.protocols_to_replicate) == 0:
            #     raise ValueError('A replicator does not have any protocols to replicate.')

            if not isinstance(replicator.template_values, list) and not isinstance(
                replicator.template_values, ProtocolPath
            ):

                raise ValueError(
                    "The template values of a replicator must either be "
                    "a list of values, or a reference to a list of values."
                )

            if isinstance(replicator.template_values, list):

                for template_value in replicator.template_values:

                    if not isinstance(template_value, ProtocolPath):
                        continue

                    if template_value.start_protocol not in self.protocols:
                        raise ValueError(
                            "The value source {} does not exist.".format(template_value)
                        )

            elif isinstance(replicator.template_values, ProtocolPath):

                if not replicator.template_values.is_global:
                    raise ValueError(
                        "Template values must either be a constant, or come from the global "
                        "scope."
                    )

            if (
                self.final_value_source is not None
                and self.final_value_source.protocol_path.find(
                    replicator.placeholder_id
                )
                >= 0
            ):

                raise ValueError(
                    "The final value source cannot come from"
                    "a protocol which is being replicated."
                )

    def _validate_final_value(self):

        if self.final_value_source is None:
            return

        if self.final_value_source.start_protocol not in self.protocols:
            raise ValueError(
                "The value source {} does not exist.".format(self.final_value_source)
            )

        protocol_schema = self.protocols[self.final_value_source.start_protocol]

        protocol_object = registered_workflow_protocols[protocol_schema.type](
            protocol_schema.id
        )
        protocol_object.schema = protocol_schema

        protocol_object.get_value(self.final_value_source)

        attribute_type = protocol_object.get_class_attribute(
            self.final_value_source
        ).type_hint
        assert is_type_subclass_of_type(attribute_type, EstimatedQuantity)

    def _validate_gradients(self):

        for gradient_source in self.gradients_sources:

            if gradient_source.start_protocol not in self.protocols:
                raise ValueError(
                    "The gradient source {} does not exist.".format(gradient_source)
                )

            protocol_schema = self.protocols[gradient_source.start_protocol]

            protocol_object = registered_workflow_protocols[protocol_schema.type](
                protocol_schema.id
            )
            protocol_object.schema = protocol_schema

            protocol_object.get_value(gradient_source)

            attribute_type = protocol_object.get_class_attribute(
                gradient_source
            ).type_hint
            assert is_type_subclass_of_type(attribute_type, ParameterGradient)

    def _validate_output_to_store(self, output_to_store):
        """Validates that the references of a particular output to store
        are valid.

        Parameters
        ----------
        output_to_store: BaseStoredData
            The output to store to validate.
        """

        if not isinstance(output_to_store, BaseStoredData):

            raise ValueError(
                "Only `BaseStoredData` derived objects are allowed "
                "in the outputs_to_store dictionary at this time."
            )

        output_to_store.validate()

        for attribute_name in output_to_store.get_attributes(StorageAttribute):

            attribute_value = getattr(output_to_store, attribute_name)

            if isinstance(attribute_value, ReplicatorValue):

                if len(self.replicators) == 0:

                    raise ValueError(
                        "An output to store is trying to take its value from a "
                        "replicator, while this schema has no replicators."
                    )

                elif (
                    len(
                        [
                            replicator
                            for replicator in self.replicators
                            if attribute_value.replicator_id == replicator.id
                        ]
                    )
                    == 0
                ):

                    raise ValueError(
                        f"An output to store is trying to take its value from a "
                        f"replicator {attribute_value.replicator_id} which does not exist."
                    )

            if (
                not isinstance(attribute_value, ProtocolPath)
                or attribute_value.is_global
            ):
                continue

            if attribute_value.start_protocol not in self.protocols:
                raise ValueError(
                    "The value source {} does not exist.".format(attribute_value)
                )

            protocol_schema = self.protocols[attribute_value.start_protocol]

            protocol_object = registered_workflow_protocols[protocol_schema.type](
                protocol_schema.id
            )
            protocol_object.schema = protocol_schema

            protocol_object.get_value(attribute_value)

    def _validate_outputs_to_store(self):
        """Validates that the references to the outputs to store
        are valid.
        """
        for output_label in self.outputs_to_store:

            output_to_store = self.outputs_to_store[output_label]
            self._validate_output_to_store(output_to_store)

    def validate_interfaces(self):
        """Validates the flow of the data between protocols, ensuring
        that inputs and outputs correctly match up.
        """

        self._validate_final_value()
        self._validate_gradients()
        self._validate_replicators()
        self._validate_outputs_to_store()

        for protocol_id in self.protocols:

            protocol_schema = self.protocols[protocol_id]

            protocol_object = registered_workflow_protocols[protocol_schema.type](
                protocol_schema.id
            )
            protocol_object.schema = protocol_schema

            for input_path in protocol_object.required_inputs:

                input_value = protocol_object.get_value(input_path)
                input_attribute = protocol_object.get_class_attribute(input_path)

                if not isinstance(input_attribute, InputAttribute):
                    continue

                is_optional = input_attribute.optional

                if input_value == UNDEFINED and is_optional is False:

                    raise Exception(
                        "The {} required input of protocol {} in the {} schema was "
                        "not set.".format(input_path, protocol_id, self.id)
                    )

            for input_path in protocol_object.required_inputs:

                value_references = protocol_object.get_value_references(input_path)

                for source_path, value_reference in value_references.items():

                    if value_reference.is_global:
                        # We handle global input validation separately
                        continue

                    value_reference = self._get_unreplicated_path(value_reference)

                    # Make sure the other protocol whose output we are interested
                    # in actually exists.
                    if value_reference.start_protocol not in self.protocols:

                        raise Exception(
                            f"The {protocol_object.id} protocol of the {self.id} schema "
                            f"tries to take input from a non-existent "
                            f"protocol: {value_reference.full_path}"
                        )

                    other_protocol_schema = self.protocols[
                        value_reference.start_protocol
                    ]

                    other_protocol_object = registered_workflow_protocols[
                        other_protocol_schema.type
                    ](other_protocol_schema.id)
                    other_protocol_object.schema = other_protocol_schema

                    unnested_value_reference = self._get_unnested_protocol_path(
                        value_reference
                    )
                    unnested_source_path = self._get_unnested_protocol_path(source_path)

                    # Make sure the other protocol has the output referenced
                    # by this input.
                    other_protocol_object.get_value(unnested_value_reference)

                    # Do a very rudimentary type check between the input and
                    # output types. This is not currently possible for nested
                    # or indexed properties, or outputs of replicated protocols.
                    if (
                        value_reference.full_path != unnested_value_reference.full_path
                        or source_path.full_path != unnested_source_path.full_path
                    ):

                        continue

                    is_replicated_reference = False

                    for replicator in self.replicators:

                        if (
                            replicator.placeholder_id in protocol_id
                            and replicator.placeholder_id
                            in value_reference.protocol_path
                        ) or (
                            replicator.placeholder_id not in protocol_id
                            and replicator.placeholder_id
                            not in value_reference.protocol_path
                        ):

                            continue

                        is_replicated_reference = True
                        break

                    if is_replicated_reference:
                        continue

                    expected_input_type = protocol_object.get_class_attribute(
                        unnested_source_path
                    ).type_hint
                    expected_output_type = other_protocol_object.get_class_attribute(
                        unnested_value_reference
                    ).type_hint

                    if expected_input_type is None or expected_output_type is None:
                        continue

                    if not is_type_subclass_of_type(
                        expected_output_type, expected_input_type
                    ):

                        raise Exception(
                            f"The output type ({expected_output_type}) of {value_reference} does not "
                            f"match the requested input type ({expected_input_type}) of {source_path}"
                        )
