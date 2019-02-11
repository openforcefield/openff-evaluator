"""
A collection of schemas which represent elements of a property calculation workflow.
"""

from typing import Dict, List

from pydantic import BaseModel
from simtk import unit

from propertyestimator.utils.serialization import PolymorphicDataType, serialize_quantity, TypedBaseModel
from propertyestimator.workflow.plugins import available_protocols
from propertyestimator.workflow.utils import ProtocolPath


class ProtocolSchema(TypedBaseModel):
    """A json serializable representation which stores the
    user definable parameters of a protocol.
    """
    id: str = None
    type: str = None

    inputs: Dict[str, PolymorphicDataType] = {}

    class Config:

        arbitrary_types_allowed = True

        json_encoders = {
            unit.Quantity: lambda value: serialize_quantity(value),
            ProtocolPath: lambda value: value.full_path,
            PolymorphicDataType: lambda value: PolymorphicDataType.serialize(value)
        }


class ProtocolGroupSchema(ProtocolSchema):
    """A json serializable representation of a protocol
    definition.
    """
    grouped_protocol_schemas: List[ProtocolSchema] = []


class ProtocolReplicator(BaseModel):
    """A protocol replicator contains the information necessary to replicate
    parts of a property estimation workflow.

    The protocols referenced by `protocols_to_replicate` will be cloned for
    each value present in `template_values`. Properties of replicated protocols
    referenced by `template_targets` will have their value set to the value
    taken from `template_values`.

    Each of the protocols referenced in the `protocols_to_replicate` must have an id
    which contains the '$index' string (e.g component_$index_build_coordinates) -
    when the protocol is replicated, $index will be replaced by the protocols actual
    index, which corresponds to a value in the `template_values` array.

    Any protocols which take input from a replicated protocol will be updated to
    instead take a list of value, populated by the outputs of the replicated
    protocols.

    Notes
    -----
    * The protocols referenced to by `template_targets` **must not** be protocols
      which are being replicated.
    * The `template_values` property must be a list of either constant values,
      or :obj:`ProtocolPath`'s which take their value from the `global` scope.
    """
    protocols_to_replicate: List[ProtocolPath] = []

    template_values: PolymorphicDataType = None
    template_targets: List[ProtocolPath] = []

    class Config:

        arbitrary_types_allowed = True

        json_encoders = {
            unit.Quantity: lambda value: serialize_quantity(value),
            ProtocolPath: lambda value: value.full_path,
            PolymorphicDataType: lambda value: PolymorphicDataType.serialize(value)
        }

    def replicates_protocol_or_child(self, protocol_path):
        """Returns whether the protocol pointed to by `protocol_path` (or
        any of its children) will be replicated by this replicator."""

        for path_to_replace in self.protocols_to_replicate:

            if path_to_replace.full_path.find(protocol_path.start_protocol) < 0:
                continue

            return True

        return False


class WorkflowSchema(BaseModel):
    """Outlines the workflow which should be followed when calculating a certain property.
    """
    property_type: str = None
    id: str = None

    protocols: Dict[str, ProtocolSchema] = {}
    replicators: List[ProtocolReplicator] = []

    final_value_source: ProtocolPath = None

    outputs_to_store: Dict[str, Dict[str, ProtocolPath]] = {}

    class Config:
        arbitrary_types_allowed = True

        json_encoders = {
            unit.Quantity: lambda value: serialize_quantity(value),
            ProtocolPath: lambda value: value.full_path,
            PolymorphicDataType: lambda value: PolymorphicDataType.serialize(value)
        }

    def validate_interfaces(self):
        """Validates the flow of the data between protocols, ensuring
        that inputs and outputs correctly match up.
        """

        if self.final_value_source.start_protocol not in self.protocols:
            raise ValueError('The value source {} does not exist.'.format(self.final_value_source))

        for output_to_store in self.outputs_to_store:

            for output_type in self.outputs_to_store[output_to_store]:

                output_path = self.outputs_to_store[output_to_store][output_type]

                if output_path.start_protocol in self.protocols:
                    continue

                raise ValueError('The data source {} does not exist.'.format(self.final_value_source))

        for protocol_id in self.protocols:

            protocol_schema = self.protocols[protocol_id]

            protocol_object = available_protocols[protocol_schema.type](protocol_schema.id)
            protocol_object.schema = protocol_schema

            if protocol_id == self.final_value_source.start_protocol:
                protocol_object.get_value(self.final_value_source)

            for output_to_store in self.outputs_to_store:

                for output_type in self.outputs_to_store[output_to_store]:

                    output_path = self.outputs_to_store[output_to_store][output_type]

                    if output_path.start_protocol != protocol_id:
                        continue

                    protocol_object.get_value(output_path)

            for input_path in protocol_object.required_inputs:

                input_value = protocol_object.get_value(input_path)

                if input_value is None:

                    raise Exception('The {} required input of protocol {} in the {} schema was '
                                    'not set.'.format(input_path, protocol_id, self.id))

            for input_path in protocol_object.required_inputs:

                input_values = protocol_object.get_value_references(input_path)

                for input_value in input_values:

                    if input_value.is_global:
                        # We handle global input validation separately
                        continue

                    # Make sure the other protocol whose output we are interested
                    # in actually exists.
                    if input_value.start_protocol not in self.protocols:

                        raise Exception('The {} protocol of the {} schema tries to take input from a non-existent '
                                        'protocol: {}'.format(protocol_object.id, self.id, input_value.start_protocol))

                    other_protocol_schema = self.protocols[input_value.start_protocol]

                    other_protocol_object = available_protocols[other_protocol_schema.type](other_protocol_schema.id)
                    other_protocol_object.schema = other_protocol_schema

                    # Will throw the correct exception if missing.
                    other_protocol_object.get_value(input_value)

                    expected_input_type = protocol_object.get_attribute_type(input_path)
                    expected_output_type = other_protocol_object.get_attribute_type(input_value)

                    if protocol_schema.type == 'GeneratorGroup' and expected_input_type is list:
                        continue

                    if isinstance(protocol_object.get_value(input_path), list):
                        continue

                    if expected_input_type != expected_output_type and expected_input_type is not None:

                        raise Exception('The output type ({}) of {} does not match the requested '
                                        'input type ({}) of {}'.format(expected_output_type, input_value,
                                                                       expected_input_type, input_path))
