"""
Defines the core workflow object and execution graph.
"""
import abc
import copy
import re
import traceback
import uuid
from os import path, makedirs

import mdtraj
from simtk import unit

from propertyestimator.storage import StoredSimulationData
from propertyestimator.utils import graph
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.statistics import StatisticsArray
from propertyestimator.utils.utils import SubhookedABCMeta
from propertyestimator.workflow.plugins import available_protocols
from propertyestimator.workflow.protocols import BaseProtocol
from propertyestimator.workflow.schemas import WorkflowSchema
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue


class IWorkflowProperty(SubhookedABCMeta):

    @staticmethod
    @abc.abstractmethod
    def get_default_workflow_schema(calculation_layer): pass


class Workflow:
    """Encapsulates and prepares a workflow which is able to estimate
    a physical property.
    """

    @property
    def schema(self):
        return self._get_schema()

    @schema.setter
    def schema(self, value):
        self._set_schema(value)

    def __init__(self, physical_property, global_metadata):
        """
        Constructs a new Workflow object.

        Parameters
        ----------
        physical_property: PhysicalProperty
            The property which this workflow aims to calculate.
        global_metadata: dict of str and Any
            A dictionary of the global metadata available to each
            of the workflow properties.
        """
        assert physical_property is not None and global_metadata is not None

        self.physical_property = physical_property
        self.global_metadata = global_metadata

        self.uuid = str(uuid.uuid4())

        self.protocols = {}

        self.starting_protocols = []
        self.dependants_graph = {}

        self.final_value_source = None
        self.outputs_to_store = {}

    def _get_schema(self):
        """Returns the schema that describes this workflow.

        Returns
        -------
        WorkflowSchema
            The schema that describes this workflow.
        """
        schema = WorkflowSchema()

        schema.id = self.uuid
        schema.property_type = type(self.physical_property).__name__

        schema.protocols = {}

        for protocol_id, protocol in self.protocols.items():
            schema.protocols[protocol_id] = protocol.schema

        schema.final_value_source = ProtocolPath.from_string(self.final_value_source.full_path)

        schema.outputs_to_store = {}

        for substance_identifier in self.outputs_to_store:

            schema.outputs_to_store[substance_identifier] = \
                copy.deepcopy(self.outputs_to_store[substance_identifier])

        return schema

    def _set_schema(self, value):
        """Sets this workflows properties from a `WorkflowSchema`.

        Parameters
        ----------
        value: WorkflowSchema
            The schema which outlines this steps in this workflow.
        """
        schema = WorkflowSchema.parse_json(value.json())

        self.final_value_source = ProtocolPath.from_string(schema.final_value_source.full_path)
        self.final_value_source.append_uuid(self.uuid)

        self.outputs_to_store = {}

        for output_label in schema.outputs_to_store:

            output_to_store = schema.outputs_to_store[output_label]

            for attribute_key in output_to_store.__getstate__():

                attribute_value = getattr(output_to_store, attribute_key)

                if not isinstance(attribute_value, ProtocolPath):
                    continue

                attribute_value.append_uuid(self.uuid)

            self.outputs_to_store[output_label] = output_to_store

        self._build_protocols(schema)
        self._build_dependants_graph()

    def _build_protocols(self, schema):
        """Creates a set of protocols based on a WorkflowSchema.

        Parameters
        ----------
        schema: WorkflowSchema
            The schema to use when creating the protocols
        """

        applied_replicators = []

        for replicator in schema.replicators:

            updated_protocols_to_replicate = []

            for protocol_to_replicate in replicator.protocols_to_replicate:

                for applied_replicator in applied_replicators:

                    for index in range(len(applied_replicator.template_values)):

                        replacement_string = '$({})'.format(applied_replicator.id)

                        updated_path = protocol_to_replicate.full_path.replace(replacement_string, str(index))
                        updated_protocols_to_replicate.append(ProtocolPath.from_string(updated_path))

            if len(updated_protocols_to_replicate) > 0:
                replicator.protocols_to_replicate = updated_protocols_to_replicate

            self._build_replicated_protocols(schema, replicator)
            applied_replicators.append(replicator)

        for protocol_name in schema.protocols:

            protocol_schema = schema.protocols[protocol_name]

            protocol = available_protocols[protocol_schema.type](protocol_schema.id)
            protocol.schema = protocol_schema

            # Try to set global properties on each of the protocols
            for input_path in protocol.required_inputs:

                value_references = protocol.get_value_references(input_path)

                for source_path, value_reference in value_references.items():

                    if not value_reference.is_global:
                        continue

                    value = self.global_metadata[value_reference.property_name]
                    protocol.set_value(source_path, value)

            protocol.set_uuid(self.uuid)
            self.protocols[protocol.id] = protocol

    def _build_replicated_protocols(self, schema, replicator):
        """A method to create a set of protocol schemas based on a ProtocolReplicator,
        and add them to the list of existing schemas.

        Parameters
        ----------
        schema: WorkflowSchema
            The schema which contains the protocol definitions
        replicator: :obj:`ProtocolReplicator`
            The replicator which describes which new protocols should
            be created.
        """

        template_values = replicator.template_values

        # Get the list of values which will be passed to the newly created protocols -
        # in particular those specified by `generator_protocol.template_targets`
        if isinstance(template_values, ProtocolPath):

            if not template_values.is_global:
                raise ValueError('Template values must either be a constant or come'
                                 'from the global scope (and not from {})'.format(template_values))

            template_values = self.global_metadata[template_values.property_name]

        replicated_protocols = []

        # Replicate the protocols.
        for protocol_path in replicator.protocols_to_replicate:

            if protocol_path.start_protocol in replicated_protocols:
                continue

            replicated_protocols.append(protocol_path.start_protocol)

            self._replicate_protocol(schema, protocol_path, replicator, template_values)

        outputs_to_replicate = []

        for output_label in self.outputs_to_store:

            replicator_ids = self._find_replicator_ids(output_label)

            if len(replicator_ids) <= 0 or replicator.id not in replicator_ids:
                continue

            outputs_to_replicate.append(output_label)

        # Check to see if there are any outputs to store pointing to
        # protocols which are being replicated.
        for output_label in outputs_to_replicate:

            output_to_replicate = self.outputs_to_store.pop(output_label)

            for index, template_value in enumerate(template_values):

                replacement_string = '$({})'.format(replicator.id)

                replicated_label = output_label.replace(replacement_string, str(index))
                replicated_output = copy.deepcopy(output_to_replicate)

                for attribute_key in replicated_output.__getstate__():

                    attribute_value = getattr(replicated_output, attribute_key)

                    if isinstance(attribute_value, ProtocolPath):

                        attribute_value = ProtocolPath.from_string(
                            attribute_value.full_path.replace(replacement_string, str(index)))

                    elif isinstance(attribute_value, ReplicatorValue):

                        if attribute_value.replicator_id != replicator.id:
                            continue

                        attribute_value = template_value

                    setattr(replicated_output, attribute_key, attribute_value)

                self.outputs_to_store[replicated_label] = replicated_output

        # Find any non-replicated protocols which take input from the replication templates,
        # and redirect the path to point to the actual replicated protocol.
        self._update_references_to_replicated(schema, replicator, template_values)

    @staticmethod
    def _find_replicator_ids(string):
        """Returns a list of any replicator ids (defined within a $(...))
        that are present in a given string.

        Parameters
        ----------
        string: str
            The string to search for replicator ids

        Returns
        -------
        list of str
            A list of any found replicator ids
        """
        return re.findall('[$][(](.*?)[)]', string, flags=0)

    @staticmethod
    def _replicate_protocol(schema, protocol_path, replicator, template_values):
        """Replicates a protocol in the workflow according to a
        :obj:`ProtocolReplicator`

        Parameters
        ----------
        schema: WorkflowSchema
            The schema which contains the protocol definitions
        protocol_path: :obj:`ProtocolPath`
            A reference path to the protocol to replicate.
        replicator: :obj:`ProtocolReplicator`
            The replicator object which describes how this
            protocol will be replicated.
        template_values: :obj:`list` of :obj:`Any`
            A list of the values which will be inserted
            into the newly replicated protocols.
        """
        schema_to_replicate = schema.protocols[protocol_path.start_protocol]
        replacement_string = '$({})'.format(replicator.id)

        if protocol_path.start_protocol == protocol_path.last_protocol:

            # If the protocol is not a group, replicate the protocol directly.
            for index in range(len(template_values)):

                replicated_schema_id = schema_to_replicate.id.replace(replacement_string, str(index))

                protocol = available_protocols[schema_to_replicate.type](replicated_schema_id)
                protocol.schema = schema_to_replicate

                # If this protocol references other protocols which are being
                # replicated, point it to the replicated version with the same index.
                for other_path in replicator.protocols_to_replicate:

                    _, other_path_components = ProtocolPath.to_components(other_path.full_path)

                    for protocol_id_to_rename in other_path_components:
                        protocol.replace_protocol(protocol_id_to_rename,
                                                  protocol_id_to_rename.replace(replacement_string, str(index)))

                schema.protocols[protocol.id] = protocol.schema

            schema.protocols.pop(protocol_path.start_protocol)

        else:

            # Otherwise, let the group replicate its own protocols.
            protocol = available_protocols[schema_to_replicate.type](schema_to_replicate.id)
            protocol.schema = schema_to_replicate

            protocol.apply_replicator(replicator, template_values)
            schema.protocols[protocol.id] = protocol.schema

        # Go through all of the newly created protocols, and update
        # their references and their values if targeted by the replicator.
        for index, template_value in enumerate(template_values):

            protocol_id = schema_to_replicate.id.replace(replacement_string, str(index))

            protocol_schema = schema.protocols[protocol_id]

            protocol = available_protocols[protocol_schema.type](protocol_schema.id)
            protocol.schema = protocol_schema

            template_value = template_values[index]

            # Pass the template values to the target protocols.
            for required_input in protocol.required_inputs:

                input_value = protocol.get_value(required_input)

                if not isinstance(input_value, ReplicatorValue):
                    continue

                if input_value.replicator_id != replicator.id:
                    continue

                protocol.set_value(required_input, template_value)

            schema.protocols[protocol_id] = protocol.schema

    @staticmethod
    def _update_references_to_replicated(schema, replicator, template_values):
        """Finds any non-replicated protocols which take input from a protocol
         which was replicated, and redirects the path to point to the actual
         replicated protocol.

        Parameters
        ----------
        schema: WorkflowSchema
            The schema which contains the protocol definitions
        replicator: :obj:`ProtocolReplicator`
            The replicator object which described how the protocols
            should have been replicated.
        template_values: :obj:`list` of :obj:`Any`
            The list of values that the protocols were replicated for.
        """
        replacement_string = '$({})'.format(replicator.id)

        for protocol_id in schema.protocols:

            protocol_schema = schema.protocols[protocol_id]

            protocol = available_protocols[protocol_schema.type](protocol_schema.id)
            protocol.schema = protocol_schema

            # Look at each of the protocols inputs and see if its value is either a ProtocolPath,
            # or a list of ProtocolPath's.
            for required_input in protocol.required_inputs:

                all_value_references = protocol.get_value_references(required_input)
                replicated_value_references = {}

                for source_path, value_reference in all_value_references.items():

                    if not replicator.replicates_protocol_or_child(value_reference):
                        continue

                    replicated_value_references[source_path] = value_reference

                if len(replicated_value_references) == 0:
                    continue

                generated_path_list = {}

                for source_path, value_reference in replicated_value_references.items():
                    # Replace the input value with a list of ProtocolPath's that point to
                    # the newly generated protocols.
                    path_list = [ProtocolPath.from_string(value_reference.full_path.replace(replacement_string,
                                                                                            str(index)))
                                 for index in range(len(template_values))]

                    generated_path_list[value_reference] = path_list

                input_value = protocol.get_value(required_input)

                if isinstance(input_value, ProtocolPath):
                    protocol.set_value(required_input, generated_path_list[input_value])
                    continue

                new_list_value = []

                for value in input_value:

                    if not isinstance(value, ProtocolPath) or value not in generated_path_list:
                        new_list_value.append(value)
                        continue

                    new_list_value.extend(generated_path_list[value])

                protocol.set_value(required_input, new_list_value)

            # Update the schema of the modified protocol.
            schema.protocols[protocol_id] = protocol.schema

    def _build_dependants_graph(self):
        """Builds a dictionary of key value pairs where each key represents the id of a
        protocol to be executed in this workflow, and each value a list ids of protocols
        which must be ran after the protocol identified by the key.
        """

        for protocol_name in self.protocols:
            self.dependants_graph[protocol_name] = []

        for dependant_protocol_name in self.protocols:

            dependant_protocol = self.protocols[dependant_protocol_name]

            for dependency in dependant_protocol.dependencies:

                if dependency.is_global:
                    # Global inputs are outside the scope of the
                    # schema dependency graph.
                    continue

                if dependency.start_protocol == dependant_protocol_name and dependency.start_protocol:
                    # Don't add self to the dependency list.
                    continue

                # Only add a dependency on the protocol at the head of the path,
                # dependencies on the rest of protocols in the path is then implied.
                if dependant_protocol.id in self.dependants_graph[dependency.start_protocol]:
                    continue

                self.dependants_graph[dependency.start_protocol].append(dependant_protocol.id)

        self.starting_protocols = graph.find_root_nodes(self.dependants_graph)

    def replace_protocol(self, old_protocol, new_protocol):
        """Replaces an existing protocol with a new one, while
        updating all input and local references to point to the
        new protocol.

        The main use of this method is when merging multiple protocols
        into one.

        Parameters
        ----------
        old_protocol : protocols.BaseProtocol or str
            The protocol (or its id) to replace.
        new_protocol : protocols.BaseProtocol or str
            The new protocol (or its id) to use.
        """

        old_protocol_id = old_protocol
        new_protocol_id = new_protocol

        if isinstance(old_protocol, BaseProtocol):
            old_protocol_id = old_protocol.id
        if isinstance(new_protocol, BaseProtocol):
            new_protocol_id = new_protocol.id

        if new_protocol_id in self.protocols:
            raise ValueError('A protocol with the same id already exists in this workflow.')

        for protocol_id in self.protocols:
            protocol = self.protocols[protocol_id]
            protocol.replace_protocol(old_protocol_id, new_protocol_id)

        if old_protocol_id in self.protocols and isinstance(new_protocol, BaseProtocol):
            self.protocols.pop(old_protocol_id)
            self.protocols[new_protocol_id] = new_protocol

        for index, starting_id in enumerate(self.starting_protocols):

            if starting_id == old_protocol_id:
                starting_id = new_protocol_id

            self.starting_protocols[index] = starting_id

        for protocol_id in self.dependants_graph:

            for index, dependant_id in enumerate(self.dependants_graph[protocol_id]):

                if dependant_id == old_protocol_id:
                    dependant_id = new_protocol_id

                self.dependants_graph[protocol_id][index] = dependant_id

        if old_protocol_id in self.dependants_graph:
            self.dependants_graph[new_protocol_id] = self.dependants_graph.pop(old_protocol_id)

        self.final_value_source.replace_protocol(old_protocol_id, new_protocol_id)

        for output_label in self.outputs_to_store:

            output_to_store = self.outputs_to_store[output_label]

            for attribute_key in output_to_store.__getstate__():

                attribute_value = getattr(output_to_store, attribute_key)

                if not isinstance(attribute_value, ProtocolPath):
                    continue

                attribute_value.replace_protocol(old_protocol_id,
                                                 new_protocol_id)

    @staticmethod
    def generate_default_metadata(physical_property, force_field_path, estimator_options):
        """Generates a default global metadata dictionary.
        
        Parameters
        ----------
        physical_property: PhysicalProperty
            The physical property whose arguments are available in the
            global scope.
        force_field_path: str
            The path to the force field parameters to use in the workflow.
        estimator_options: PropertyEstimatorOptions
            The options provided when an estimate request was submitted.

        Returns
        -------
        dict of str, Any

            The metadata dictionary, with the following
            keys / types:

            - thermodynamic_state: `ThermodynamicState`
            - substance: `Mixture`
            - components: list of `Mixture`
            - target_uncertainty: simtk.unit.Quantity
            - force_field_path: str
        """
        from propertyestimator.substances import Mixture

        components = []

        for component in physical_property.substance.components:

            mixture = Mixture()
            mixture.add_component(component.smiles, 1.0, False)

            components.append(mixture)

        # Define a dictionary of accessible 'global' properties.
        global_metadata = {
            "thermodynamic_state": physical_property.thermodynamic_state,
            "substance": physical_property.substance,
            "components": components,
            "target_uncertainty": physical_property.uncertainty * estimator_options.relative_uncertainty_tolerance,
            "force_field_path": force_field_path
        }

        if (isinstance(physical_property.uncertainty, unit.Quantity) and not
            isinstance(global_metadata['target_uncertainty'], unit.Quantity)):

            global_metadata['target_uncertainty'] = unit.Quantity(global_metadata['target_uncertainty'],
                                                                         physical_property.uncertainty.unit)

        return global_metadata


class WorkflowGraph:
    """A hierarchical structure for storing and submitting the workflows
    which will estimate a set of physical properties..
    """

    def __init__(self, root_directory=''):
        """Constructs a new WorkflowGraph

        Parameters
        ----------
        root_directory: str
            The root directory in which to store all outputs from
            this graph.
        """
        self._protocols_by_id = {}

        self._root_protocol_ids = []
        self._root_directory = root_directory

        self._dependants_graph = {}

        self._workflows_to_execute = {}

    def _insert_protocol(self, protocol_name, workflow, parent_protocol_ids):
        """Inserts a protocol into the workflow graph.

        Parameters
        ----------
        protocol_name : str
            The name of the protocol to insert.
        workflow : Workflow
            The workflow being inserted.
        parent_protocol_ids : `list` of str
            The ids of the new parents of the node to be inserted. If None,
            the protocol will be added as a new parent node.
        """

        if protocol_name in self._protocols_by_id:

            raise RuntimeError('A protocol with id {} has already been '
                               'inserted into the graph.'.format(protocol_name))

        protocols = self._root_protocol_ids if len(parent_protocol_ids) == 0 else []

        for parent_protocol_id in parent_protocol_ids:
            protocols.extend(x for x in self._dependants_graph[parent_protocol_id] if x not in protocols)

        protocol_to_insert = workflow.protocols[protocol_name]
        existing_protocol = None

        # Start by checking to see if the starting protocol of the workflow graph is
        # already present in the full graph.
        for protocol_id in protocols:

            if protocol_id in workflow.protocols:
                continue

            protocol = self._protocols_by_id[protocol_id]

            if not protocol.can_merge(protocol_to_insert):
                continue

            existing_protocol = protocol
            break

        if existing_protocol is not None:

            # Make a note that the existing protocol should be used in place
            # of this workflows version.
            merged_ids = existing_protocol.merge(protocol_to_insert)
            workflow.replace_protocol(protocol_to_insert, existing_protocol)

            for old_id, new_id in merged_ids.items():
                workflow.replace_protocol(old_id, new_id)

        else:

            root_directory = self._root_directory

            if len(parent_protocol_ids) == 1:

                parent_protocol = self._protocols_by_id[parent_protocol_ids[0]]
                root_directory = parent_protocol.directory

            protocol_to_insert.directory = path.join(root_directory, protocol_to_insert.id)

            # Add the protocol as a new protocol in the graph.
            self._protocols_by_id[protocol_name] = protocol_to_insert

            existing_protocol = self._protocols_by_id[protocol_name]
            self._dependants_graph[protocol_name] = []

            if len(parent_protocol_ids) == 0:
                self._root_protocol_ids.append(protocol_name)
            else:

                for protocol_id in workflow.dependants_graph:

                    if (protocol_name not in workflow.dependants_graph[protocol_id] or
                            protocol_id in self._dependants_graph[protocol_name]):
                        continue

                    self._dependants_graph[protocol_id].append(protocol_name)

        return existing_protocol.id

    def add_workflow(self, workflow):
        """Insert a workflow into the workflow graph.

        Parameters
        ----------
        workflow : Workflow
            The workflow to insert.
        """

        if workflow.uuid in self._workflows_to_execute:

            raise ValueError('A workflow with the uuid ({}) is '
                             'already in the graph.'.format(workflow.uuid))

        self._workflows_to_execute[workflow.uuid] = workflow

        protocol_execution_order = graph.topological_sort(workflow.dependants_graph)

        reduced_protocol_dependants = copy.deepcopy(workflow.dependants_graph)
        graph.apply_transitive_reduction(reduced_protocol_dependants)

        parent_protocol_ids = {}

        for protocol_id in protocol_execution_order:

            parent_ids = parent_protocol_ids.get(protocol_id) or []
            inserted_id = self._insert_protocol(protocol_id, workflow, parent_ids)

            for dependant in reduced_protocol_dependants[protocol_id]:

                if dependant not in parent_protocol_ids:
                    parent_protocol_ids[dependant] = []

                parent_protocol_ids[dependant].append(inserted_id)

    def submit(self, backend):
        """Submits the protocol graph to the backend of choice.

        Parameters
        ----------
        backend: PropertyEstimatorBackend
            The backend to execute the graph on.

        Returns
        -------
        list of Future:
            The futures of the submitted protocols.
        """
        submitted_futures = {}
        value_futures = []

        # Determine the ideal order in which to submit the
        # protocols.
        submission_order = graph.topological_sort(self._dependants_graph)

        # Build a dependency graph from the dependants graph so that
        # futures can be passed in the correct place.
        dependencies = graph.dependants_to_dependencies(self._dependants_graph)

        for node_id in submission_order:

            node = self._protocols_by_id[node_id]
            dependency_futures = []

            for dependency in dependencies[node_id]:
                dependency_futures.append(submitted_futures[dependency])

            submitted_futures[node_id] = backend.submit_task(WorkflowGraph._execute_protocol,
                                                             node.directory,
                                                             node.schema,
                                                             *dependency_futures)

        for workflow_id in self._workflows_to_execute:

            workflow = self._workflows_to_execute[workflow_id]

            # TODO: Fill in any extra required provenance.
            provenance = {}

            for protocol_id in workflow.protocols:
                
                protocol = workflow.protocols[protocol_id]
                provenance[protocol_id] = protocol.schema

            # from propertyestimator.properties import workflowSource
            #
            # source = CalculationSource(fidelity=SimulationLayer.__name__,
            #                            provenance=provenance)
            #
            # workflow.physical_property.source = source

            value_node_id = workflow.final_value_source.start_protocol

            final_futures = [
                submitted_futures[value_node_id],
            ]

            for output_label in workflow.outputs_to_store:

                output_to_store = workflow.outputs_to_store[output_label]

                for attribute_key in output_to_store.__getstate__():

                    attribute_value = getattr(output_to_store, attribute_key)

                    if not isinstance(attribute_value, ProtocolPath):
                        continue

                    final_futures.append(submitted_futures[attribute_value.start_protocol])

            # Gather the values and uncertainties of each property being calculated.
            value_futures.append(backend.submit_task(WorkflowGraph._gather_results,
                                                     workflow.physical_property,
                                                     workflow.final_value_source,
                                                     workflow.outputs_to_store,
                                                     *final_futures))

        return value_futures

    @staticmethod
    def _execute_protocol(directory, protocol_schema, *previous_outputs, available_resources, **kwargs):
        """Executes a protocol whose state is defined by the ``protocol_schema``.

        Parameters
        ----------
        protocol_schema: protocols.ProtocolSchema
            The schema defining the protocol to execute.
        previous_outputs: tuple of Any
            The results of previous protocol executions.

        Returns
        -------
        str
            The id of the executed protocol.
        dict of str and Any
            A dictionary which contains the outputs of the executed protocol.
        """

        # Store the results of the relevant previous protocols in a handy dictionary.
        # If one of the results is a failure, propagate it up the chain!
        previous_outputs_by_path = {}

        for parent_id, parent_output in previous_outputs:

            if isinstance(parent_output, PropertyEstimatorException):
                return protocol_schema.id, parent_output

            for output_path, output_value in parent_output.items():

                property_name, protocol_ids = ProtocolPath.to_components(output_path)

                if len(protocol_ids) == 0 or (len(protocol_ids) > 0 and protocol_ids[0] != parent_id):
                    protocol_ids.insert(0, parent_id)

                final_path = ProtocolPath(property_name, *protocol_ids)
                previous_outputs_by_path[final_path] = output_value

        # Recreate the protocol on the backend to bypass the need for static methods
        # and awkward args and kwargs syntax.
        protocol = available_protocols[protocol_schema.type](protocol_schema.id)
        protocol.schema = protocol_schema

        if not path.isdir(directory):
            makedirs(directory)

        for input_path in protocol.required_inputs:

            value_references = protocol.get_value_references(input_path)

            for source_path, target_path in value_references.items():

                if (target_path.start_protocol == input_path.start_protocol or
                        target_path.start_protocol == protocol.id):
                    continue

                protocol.set_value(source_path, previous_outputs_by_path[target_path])

        try:
            output_dictionary = protocol.execute(directory, available_resources)
        except Exception as e:
            # Except the unexcepted...
            formatted_exception = traceback.format_exception(None, e, e.__traceback__)

            return protocol.id, PropertyEstimatorException(directory=directory,
                                                           message='An unhandled exception occurred: '
                                                                   '{}'.format(formatted_exception))

        return protocol.id, output_dictionary

    @staticmethod
    def _gather_results(property_to_return, value_reference, outputs_to_store,
                        *protocol_results, **kwargs):
        """Gather the value and uncertainty calculated from the submission graph
        and store them in the property to return.

        Parameters
        ----------
        value_result: dict of string and Any
            The result dictionary of the protocol which calculated the value of the property.
        value_reference: ProtocolPath
            A reference to which property in the output dictionary is the actual value.
        outputs_to_store: dict of string and WorkflowOutputToStore
            A list of references to data which should be stored on the storage backend.
        property_to_return: PhysicalProperty
            The property to which the value and uncertainty belong.

        Returns
        -------
        CalculationLayerResult
            The result of attempting to estimate this property from a workflow graph.
        """
        from propertyestimator.layers.layers import CalculationLayerResult

        return_object = CalculationLayerResult()
        return_object.property_id = property_to_return.id

        results_by_id = {}

        for protocol_id, protocol_results in protocol_results:

            # Make sure none of the protocols failed and we actually have a value
            # and uncertainty.
            if isinstance(protocol_results, PropertyEstimatorException):

                return_object.workflow_error = protocol_results
                return return_object

            for output_path, output_value in protocol_results.items():

                property_name, protocol_ids = ProtocolPath.to_components(output_path)

                if len(protocol_ids) == 0 or (len(protocol_ids) > 0 and protocol_ids[0] != protocol_id):
                    protocol_ids.insert(0, protocol_id)

                final_path = ProtocolPath(property_name, *protocol_ids)
                results_by_id[final_path] = output_value

        property_to_return.value = results_by_id[value_reference].value
        property_to_return.uncertainty = results_by_id[value_reference].uncertainty

        return_object.calculated_property = property_to_return
        return_object.data_to_store = []

        # TODO: At the moment it is assumed that the output of a WorkflowGraph is
        #       a set of StoredSimulationData. This should be abstraced and made
        #       more general in future if possible.
        for output_label in outputs_to_store:

            output_to_store = outputs_to_store[output_label]

            data_to_store = StoredSimulationData()

            if output_to_store.substance is None:
                data_to_store.substance = property_to_return.substance
            elif isinstance(output_to_store.substance, ProtocolPath):
                data_to_store.substance = results_by_id[output_to_store.substance]
            else:
                data_to_store.substance = output_to_store.substance

            data_to_store.thermodynamic_state = property_to_return.thermodynamic_state

            data_to_store.provenance = property_to_return.source

            data_to_store.source_calculation_id = property_to_return.id

            coordinate_path = results_by_id[output_to_store.coordinate_file_path]
            trajectory_path = results_by_id[output_to_store.trajectory_file_path]

            data_to_store.trajectory_data = mdtraj.load_dcd(trajectory_path, top=coordinate_path)

            statistics_path = results_by_id[output_to_store.statistics_file_path]
            data_to_store.statistics_data = StatisticsArray.from_pandas_csv(statistics_path)

            data_to_store.statistical_inefficiency = results_by_id[output_to_store.statistical_inefficiency]

            return_object.data_to_store.append(data_to_store)

        return return_object
