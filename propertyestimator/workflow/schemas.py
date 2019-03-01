"""
A collection of schemas which represent elements of a property calculation workflow.
"""
import re

from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.serialization import TypedBaseModel
from propertyestimator.workflow.plugins import available_protocols
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

        return {
            'id': self.id,
            'type': self.type,

            'inputs': self.inputs
        }

    def __setstate__(self, state):

        self.id = state['id']
        self.type = state['type']

        self.inputs = state['inputs']


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
        state.update({
            'grouped_protocol_schemas': self.grouped_protocol_schemas,
        })

        return state

    def __setstate__(self, state):

        super(ProtocolGroupSchema, self).__setstate__(state)
        self.grouped_protocol_schemas = state['grouped_protocol_schemas']


class ProtocolReplicator(TypedBaseModel):
    """A protocol replicator contains the information necessary to replicate
    parts of a property estimation workflow.

    The protocols referenced by `protocols_to_replicate` will be cloned for
    each value present in `template_values`. Protocols that are being replicated
    will also have any ReplicatorValue inputs replaced with the actual value.

    Each of the protocols referenced in the `protocols_to_replicate` must have an id
    which contains the '$(id_index)' string (e.g component_$(id_index)_build_coordinates)
    where here *`id` is the id of the replicator* - when the protocol is replicated, $(id_index)
    will be replaced by the protocols actual index, which corresponds to a value in the
    `template_values` array.

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

    def __init__(self, replicator_id=''):
        """Constructs a new ProtocolReplicator object.

        Parameters
        ----------
        replicator_id: str
            The id of this replicator.
        """
        self.id = replicator_id

        self.protocols_to_replicate = []
        self.template_values = None

    def __getstate__(self):

        return {
            'id': self.id,

            'protocols_to_replicate': self.protocols_to_replicate,
            'template_values': self.template_values
        }

    def __setstate__(self, state):

        self.id = state['id']

        self.protocols_to_replicate = state['protocols_to_replicate']
        self.template_values = state['template_values']

    def replicates_protocol_or_child(self, protocol_path):
        """Returns whether the protocol pointed to by `protocol_path` (or
        any of its children) will be replicated by this replicator."""

        for path_to_replace in self.protocols_to_replicate:

            if path_to_replace.full_path.find(protocol_path.start_protocol) < 0:
                continue

            return True

        return False


class WorkflowOutputToStore:

    def __init__(self):

        self.substance = None

        self.trajectory_file_path = None
        self.coordinate_file_path = None

        self.statistics_file_path = None

        self.statistical_inefficiency = None

    def __getstate__(self):

        return_value = {
            'substance': self.substance,
            'trajectory_file_path': self.trajectory_file_path,
            'coordinate_file_path': self.coordinate_file_path,
            'statistics_file_path': self.statistics_file_path,
            'statistical_inefficiency': self.statistical_inefficiency,
        }
        return return_value

    def __setstate__(self, state):

        self.substance = state['substance']
        self.trajectory_file_path = state['trajectory_file_path']
        self.coordinate_file_path = state['coordinate_file_path']
        self.statistics_file_path = state['statistics_file_path']
        self.statistical_inefficiency = state['statistical_inefficiency']


class WorkflowSchema(TypedBaseModel):
    """Outlines the workflow which should be followed when calculating
    a certain property.
    """

    def __init__(self,property_type=None):
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

        self.outputs_to_store = {}

    def __getstate__(self):

        return {
            'property_type': self.property_type,
            'id': self.id,

            'protocols': self.protocols,
            'replicators': self.replicators,

            'final_value_source': self.final_value_source,

            'outputs_to_store': self.outputs_to_store,
        }

    def __setstate__(self, state):

        self.property_type = state['property_type']
        self.id = state['id']

        self.protocols = state['protocols']
        self.replicators = state['replicators']

        self.final_value_source = state['final_value_source']

        self.outputs_to_store = state['outputs_to_store']

    def _validate_replicators(self):

        for replicator in self.replicators:

            assert replicator.id is not None and len(replicator.id) > 0

            if len(replicator.protocols_to_replicate) == 0:
                raise ValueError('A replicator does not have any protocols to replicate.')

            if (not isinstance(replicator.template_values, list) and
                not isinstance(replicator.template_values, ProtocolPath)):

                raise ValueError('The template values of a replicator must either be '
                                 'a list of values, or a reference to a list of values.')

            if isinstance(replicator.template_values, list):

                for template_value in replicator.template_values:

                    if not isinstance(template_value, ProtocolPath):
                        continue

                    if template_value.start_protocol not in self.protocols:
                        raise ValueError('The value source {} does not exist.'.format(template_value))

            elif isinstance(replicator.template_values, ProtocolPath):

                if not replicator.template_values.is_global:
                    raise ValueError('Template values must either be a constant, or come from the global '
                                     'scope.')

            for protocol_path in replicator.protocols_to_replicate:

                if protocol_path.start_protocol not in self.protocols:
                    raise ValueError('The value source {} does not exist.'.format(protocol_path))

                if protocol_path == self.final_value_source:

                    raise ValueError('The final value source cannot come from'
                                     'a protocol which is being replicated.')

                protocol_schema = self.protocols[protocol_path.start_protocol]

                if re.search(r'\$\(.*\)', protocol_schema.id) is None:

                    raise ValueError('Protocols which are being replicated must contain '
                                     'the replicator id $(id) their protocol id.')

    def _validate_final_value(self):

        if self.final_value_source is None:
            raise ValueError('The final value source must not be None.')

        if self.final_value_source.start_protocol not in self.protocols:
            raise ValueError('The value source {} does not exist.'.format(self.final_value_source))

        protocol_schema = self.protocols[self.final_value_source.start_protocol]

        protocol_object = available_protocols[protocol_schema.type](protocol_schema.id)
        protocol_object.schema = protocol_schema

        protocol_object.get_value(self.final_value_source)

        attribute_type = protocol_object.get_attribute_type(self.final_value_source)
        assert issubclass(attribute_type, EstimatedQuantity)

    def _validate_outputs_to_store(self):
        
        attributes_to_check = [
            'substance',
            'trajectory_file_path',
            'coordinate_file_path',
            'statistics_file_path',
            'statistical_inefficiency',
        ]

        for output_label in self.outputs_to_store:

            output_to_store = self.outputs_to_store[output_label]

            if not isinstance(output_to_store, WorkflowOutputToStore):

                raise ValueError('Only WorkflowOutputToStore objects are allowed '
                                 'in the outputs_to_store dictionary at this time.')

            for attribute_name in attributes_to_check:

                attribute_value = getattr(output_to_store, attribute_name)

                if isinstance(attribute_value, ReplicatorValue):

                    if len(self.replicators) == 0:

                        raise ValueError('An output to store is trying to take its value from a '
                                         'replicator, while this schema is no replicators.')

                    elif len([replicator for replicator in self.replicators if
                              attribute_value.replicator_id == replicator.id]) == 0:

                        raise ValueError('An output to store is trying to take its value from a '
                                         'replicator {} which does not exist.'.format(attribute_value.replicator_id))

                if not isinstance(attribute_value, ProtocolPath) or attribute_value.is_global:
                    continue

                if attribute_value.start_protocol not in self.protocols:
                    raise ValueError('The value source {} does not exist.'.format(attribute_value))

                protocol_schema = self.protocols[attribute_value.start_protocol]

                protocol_object = available_protocols[protocol_schema.type](protocol_schema.id)
                protocol_object.schema = protocol_schema

                protocol_object.get_value(attribute_value)

    def validate_interfaces(self):
        """Validates the flow of the data between protocols, ensuring
        that inputs and outputs correctly match up.
        """

        self._validate_final_value()
        self._validate_replicators()
        self._validate_outputs_to_store()

        for protocol_id in self.protocols:

            protocol_schema = self.protocols[protocol_id]

            protocol_object = available_protocols[protocol_schema.type](protocol_schema.id)
            protocol_object.schema = protocol_schema

            for input_path in protocol_object.required_inputs:

                input_value = protocol_object.get_value(input_path)

                if input_value is None:

                    raise Exception('The {} required input of protocol {} in the {} schema was '
                                    'not set.'.format(input_path, protocol_id, self.id))

            for input_path in protocol_object.required_inputs:

                value_references = protocol_object.get_value_references(input_path)

                for source_path, value_reference in value_references.items():

                    if value_reference.is_global:
                        # We handle global input validation separately
                        continue

                    # Make sure the other protocol whose output we are interested
                    # in actually exists.
                    if value_reference.start_protocol not in self.protocols:

                        raise Exception('The {} protocol of the {} schema tries to take input from a non-existent '
                                        'protocol: {}'.format(protocol_object.id, self.id,
                                                              value_reference.start_protocol))

                    other_protocol_schema = self.protocols[value_reference.start_protocol]

                    other_protocol_object = available_protocols[other_protocol_schema.type](other_protocol_schema.id)
                    other_protocol_object.schema = other_protocol_schema

                    # Will throw the correct exception if missing.
                    other_protocol_object.get_value(value_reference)

                    expected_input_type = protocol_object.get_attribute_type(source_path)
                    expected_output_type = other_protocol_object.get_attribute_type(value_reference)

                    if (expected_input_type is not None and expected_output_type is not None and
                        expected_input_type != expected_output_type):

                        raise Exception('The output type ({}) of {} does not match the requested '
                                        'input type ({}) of {}'.format(expected_output_type, value_reference,
                                                                       expected_input_type, source_path))
