"""
The direct simulation estimation layer.
"""

import copy
import logging
import pickle
import traceback
import uuid
from os import path, makedirs

import mdtraj

from propertyestimator.layers import register_calculation_layer, PropertyCalculationLayer
from propertyestimator.storage import StoredSimulationData
from propertyestimator.substances import Mixture
from propertyestimator.utils import graph
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.serialization import PolymorphicDataType, serialize_force_field
from propertyestimator.utils.statistics import StatisticsArray
from propertyestimator.workflow import WorkflowSchema
from propertyestimator.workflow import protocols, groups, plugins
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue
from .layers import CalculationLayerResult


class DirectCalculation:
    """Encapsulates and prepares the workflow needed to calculate a physical
    property by direct simulation methods.
    """

    def __init__(self, physical_property, force_field_path, schema, options):
        """
        Constructs a new DirectCalculation object.

        Parameters
        ----------
        physical_property: PhysicalProperty
            The protocol this node will execute.
        force_field_path: str
            The force field to use for this calculation.
        schema: WorkflowSchema
            The schema to use to calculate this property.
        options: PropertyEstimatorOptions
            The options to run the calculation with.
        """
        self.physical_property = physical_property

        self.uuid = str(uuid.uuid4())

        self.protocols = {}

        self.starting_protocols = []
        self.dependants_graph = {}

        self.final_value_source = None
        self.outputs_to_store = {}

        self.schema = WorkflowSchema.parse_raw(schema.json())

        components = []

        for component in physical_property.substance.components:

            mixture = Mixture()
            mixture.add_component(component.smiles, 1.0, False)

            components.append(mixture)

        # Define a dictionary of accessible 'global' properties.
        self.global_properties = {
            "thermodynamic_state": physical_property.thermodynamic_state,
            "substance": physical_property.substance,
            "components": components,
            "target_uncertainty": physical_property.uncertainty * options.relative_uncertainty_tolerance,
            "force_field_path": force_field_path
        }

        # TODO: Nasty hack to turn a unitless quantity back into a unitless quantity after
        #       scalar multiplication.
        from simtk import unit

        if (isinstance(physical_property.uncertainty, unit.Quantity) and not
            isinstance(self.global_properties['target_uncertainty'], unit.Quantity)):

            self.global_properties['target_uncertainty'] = unit.Quantity(self.global_properties['target_uncertainty'],
                                                                         physical_property.uncertainty.unit)

        self.final_value_source = copy.deepcopy(self.schema.final_value_source)
        self.final_value_source.append_uuid(self.uuid)

        self.outputs_to_store = copy.deepcopy(self.schema.outputs_to_store)

        for output_label in self.outputs_to_store:

            output_to_store = self.outputs_to_store[output_label].value

            for attribute_key in output_to_store.__getstate__():

                attribute_value = getattr(output_to_store, attribute_key)

                if not isinstance(attribute_value, ProtocolPath):
                    continue

                attribute_value.append_uuid(self.uuid)

            self.outputs_to_store[output_label] = output_to_store

        self._build_protocols()

        self._build_dependants_graph()
        self.update_schema()

    def _build_protocols(self):
        """Creates a set of protocols based on this calculations WorkflowSchema.
        """
        for replicator in self.schema.replicators:
            self._build_replicated_protocols(replicator)

        for protocol_name in self.schema.protocols:

            protocol_schema = self.schema.protocols[protocol_name]

            protocol = plugins.available_protocols[protocol_schema.type](protocol_schema.id)
            protocol.schema = protocol_schema

            # Try to set global properties on each of the protocols
            for input_path in protocol.required_inputs:

                input_values = protocol.get_value_references(input_path)
                values = {}

                for input_value in input_values:

                    if not input_value.is_global:
                        continue

                    values[input_value] = self.global_properties[input_value.property_name]

                input_value = protocol.get_value(input_path)

                if isinstance(input_value, ProtocolPath) and input_value in values:
                    protocol.set_value(input_path, values[input_value])

                elif isinstance(input_value, list):

                    value_list = []

                    for target_value in input_value:

                        if not isinstance(target_value, ProtocolPath):
                            value_list.append(target_value)
                        elif target_value in values:
                            value_list.append(values[target_value])
                        else:
                            value_list.append(target_value)

                    protocol.set_value(input_path, value_list)

            if isinstance(protocol, groups.ConditionalGroup):

                for condition in protocol.conditions:

                    left_value = condition.left_hand_value

                    if isinstance(left_value, protocols.ProtocolPath) and left_value.is_global:
                        condition.left_hand_value = self.global_properties[left_value.property_name]

                    right_value = condition.right_hand_value

                    if isinstance(right_value, protocols.ProtocolPath) and right_value.is_global:
                        condition.right_hand_value = self.global_properties[right_value.property_name]

            protocol.set_uuid(self.uuid)
            self.protocols[protocol.id] = protocol

    def _build_replicated_protocols(self, replicator):
        """A method to create a set of protocol schemas based on a ProtocolReplicator,
        and add them to the list of existing schemas.

        Parameters
        ----------
        replicator: :obj:`ProtocolReplicator`
            The replicator which describes which new protocols should
            be created.
        """

        template_values = replicator.template_values.value

        # Get the list of values which will be passed to the newly created protocols -
        # in particular those specified by `generator_protocol.template_targets`
        if isinstance(template_values, ProtocolPath):

            if not template_values.is_global:

                raise ValueError('Template values must either be a constant or come'
                                 'from the global scope (and not from {})'.format(template_values))

            template_values = self.global_properties[template_values.property_name]

        replicated_protocols = []

        # Replicate the protocols.
        for protocol_path in replicator.protocols_to_replicate:

            if protocol_path.start_protocol in replicated_protocols:
                continue

            replicated_protocols.append(protocol_path.start_protocol)

            self._replicate_protocol(protocol_path, replicator, template_values)

        outputs_to_replicate = [output_label for output_label in self.outputs_to_store if
                                output_label.find('$index') >= 0]

        # Check to see if there are any outputs to store pointing to
        # protocols which are being replicated.
        for output_label in outputs_to_replicate:

            output_to_replicate = self.outputs_to_store.pop(output_label)

            for index, template_value in enumerate(template_values):

                replicated_label = output_label.replace('$index', str(index))
                replicated_output = copy.deepcopy(output_to_replicate)

                for attribute_key in replicated_output.__getstate__():

                    attribute_value = getattr(replicated_output, attribute_key)

                    if isinstance(attribute_value, ProtocolPath):

                        attribute_value = ProtocolPath.from_string(
                            attribute_value.full_path.replace('$index', str(index)))

                    elif isinstance(attribute_value, ReplicatorValue):

                        attribute_value = template_value

                    setattr(replicated_output, attribute_key, attribute_value)

                self.outputs_to_store[replicated_label] = replicated_output

        # Find any non-replicated protocols which take input from the replication templates,
        # and redirect the path to point to the actual replicated protocol.
        self._update_references_to_replicated(replicator, template_values)

    def _replicate_protocol(self, protocol_path, replicator, template_values):
        """Replicates a protocol in the calculation according to a
        :obj:`ProtocolReplicator`

        Parameters
        ----------
        protocol_path: :obj:`ProtocolPath`
            A reference path to the protocol to replicate.
        replicator: :obj:`ProtocolReplicator`
            The replicator object which describes how this
            protocol will be replicated.
        template_values: :obj:`list` of :obj:`Any`
            A list of the values which will be inserted
            into the newly replicated protocols.
        """
        schema_to_replicate = self.schema.protocols[protocol_path.start_protocol]

        if protocol_path.start_protocol == protocol_path.last_protocol:

            # If the protocol is not a group, replicate the protocol directly.
            for index in range(len(template_values)):

                replicated_schema_id = schema_to_replicate.id.replace('$index', str(index))

                protocol = plugins.available_protocols[schema_to_replicate.type](replicated_schema_id)
                protocol.schema = schema_to_replicate

                # If this protocol references other protocols which are being
                # replicated, point it to the replicated version with the same index.
                for other_path in replicator.protocols_to_replicate:

                    _, other_path_components = ProtocolPath.to_components(other_path.full_path)

                    for protocol_id_to_rename in other_path_components:
                        protocol.replace_protocol(protocol_id_to_rename,
                                                  protocol_id_to_rename.replace('$index', str(index)))

                self.schema.protocols[protocol.id] = protocol.schema

            self.schema.protocols.pop(protocol_path.start_protocol)

        else:

            # Otherwise, let the group replicate its own protocols.
            protocol = plugins.available_protocols[schema_to_replicate.type](schema_to_replicate.id)
            protocol.schema = schema_to_replicate

            protocol.apply_replicator(replicator, template_values)
            self.schema.protocols[protocol.id] = protocol.schema

        # Go through all of the newly created protocols, and update
        # their references and their values if targeted by the replicator.
        for index, template_value in enumerate(template_values):

            protocol_id = schema_to_replicate.id.replace('$index', str(index))

            protocol_schema = self.schema.protocols[protocol_id]

            protocol = plugins.available_protocols[protocol_schema.type](protocol_schema.id)
            protocol.schema = protocol_schema

            template_value = template_values[index]

            # Pass the template values to the target protocols.
            for required_input in protocol.required_inputs:

                input_value = protocol.get_value(required_input)

                if not isinstance(input_value, ReplicatorValue):
                    continue

                protocol.set_value(required_input, template_value)

            self.schema.protocols[protocol_id] = protocol.schema

    def _update_references_to_replicated(self, replicator, template_values):
        """Finds any non-replicated protocols which take input from a protocol
         which was replicated, and redirects the path to point to the actual
         replicated protocol.

        Parameters
        ----------
        replicator: :obj:`ProtocolReplicator`
            The replicator object which described how the protocols
            should have been replicated.
        template_values: :obj:`list` of :obj:`Any`
            The list of values that the protocols were replicated for.
        """

        for protocol_id in self.schema.protocols:

            protocol_schema = self.schema.protocols[protocol_id]

            protocol = plugins.available_protocols[protocol_schema.type](protocol_schema.id)
            protocol.schema = protocol_schema

            # Look at each of the protocols inputs and see if its value is either a ProtocolPath,
            # or a list of ProtocolPath's.
            for required_input in protocol.required_inputs:

                input_values = [value for value in protocol.get_value_references(required_input) if
                                replicator.replicates_protocol_or_child(value)]

                if len(input_values) == 0:
                    continue

                generated_path_list = {}

                for input_value in input_values:
                    # Replace the input value with a list of ProtocolPath's that point to
                    # the newly generated protocols.
                    path_list = [ProtocolPath.from_string(input_value.full_path.replace('$index', str(index)))
                                 for index in range(len(template_values))]

                    generated_path_list[input_value] = path_list

                input_value = protocol.get_value(required_input)

                if isinstance(input_value, ProtocolPath):
                    protocol.set_value(required_input, generated_path_list[input_value])
                    continue

                new_list_value = []

                for value in input_value:

                    if not isinstance(value, ProtocolPath):
                        new_list_value.append(value)
                        continue

                    new_list_value.extend(generated_path_list[value])

                protocol.set_value(required_input, new_list_value)

            # Update the schema of the modified protocol.
            self.schema.protocols[protocol_id] = protocol.schema

    def _build_dependants_graph(self):
        """Builds a dictionary of key value pairs where each key represents the id of a
        protocol to be executed in this calculation, and each value a list ids of protocols
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
        
        if isinstance(old_protocol, protocols.BaseProtocol):
            old_protocol_id = old_protocol.id
        if isinstance(new_protocol, protocols.BaseProtocol):
            new_protocol_id = new_protocol.id

        if new_protocol_id in self.protocols:
            raise ValueError('A protocol with the same id already exists in this calculation.')

        for protocol_id in self.protocols:

            protocol = self.protocols[protocol_id]
            protocol.replace_protocol(old_protocol_id, new_protocol_id)

        if old_protocol_id in self.protocols and isinstance(new_protocol, protocols.BaseProtocol):

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

    def update_schema(self):

        self.schema.protocols = {}

        for protocol_id in self.protocols:

            protocol = self.protocols[protocol_id]
            self.schema.protocols[protocol_id] = protocol.schema


class DirectCalculationGraph:
    """A hierarchical structure for storing and submitting the workflows
    which will calculate a set of physical properties..
    """

    def __init__(self, root_directory=''):
        """Constructs a new DirectCalculationGraph

        Parameters
        ----------
        root_directory: str, default=''
            The root directory in which to store all outputs from
            this graph.
        """
        self._nodes_by_id = {}

        self._root_nodes = []
        self._root_directory = root_directory

        self._dependants_graph = {}

        self._calculations_to_run = {}

    def _insert_node(self, protocol_name, calculation, parent_node_ids):
        """Inserts a protocol into the calculation graph.

        Parameters
        ----------
        protocol_name : str
            The name of the protocol to insert.
        calculation : DirectCalculation
            The calculation being inserted.
        parent_node_ids : List[str]
            The ids of the new parents of the node to be inserted. If None,
            the protocol will be added as a new parent node.
        """

        if protocol_name in self._nodes_by_id:

            raise RuntimeError('A protocol with id ' + protocol_name + ' has already been inserted'
                                                                       ' into the graph.')

        nodes = self._root_nodes if len(parent_node_ids) == 0 else []

        for parent_node_id in parent_node_ids:
            nodes.extend(x for x in self._dependants_graph[parent_node_id] if x not in nodes)

        protocol_to_insert = calculation.protocols[protocol_name]
        existing_node = None

        # Start by checking to see if the starting node of the calculation graph is
        # already present in the full graph.
        for node_id in nodes:

            if node_id in calculation.protocols:
                continue

            node = self._nodes_by_id[node_id]

            if not node.can_merge(protocol_to_insert):
                continue

            existing_node = node
            break

        if existing_node is not None:
            # Make a note that the existing node should be used in place
            # of this calculations version.

            merged_ids = existing_node.merge(protocol_to_insert)
            calculation.replace_protocol(protocol_to_insert, existing_node)

            for old_id, new_id in merged_ids.items():
                calculation.replace_protocol(old_id, new_id)

        else:

            # parent_node = None if parent_node_name is None else self._nodes_by_id[parent_node_name]
            root_directory = self._root_directory

            if len(parent_node_ids) == 1:

                parent_node = self._nodes_by_id[parent_node_ids[0]]
                root_directory = parent_node.directory

            protocol_to_insert.directory = path.join(root_directory, protocol_to_insert.id)

            # Add the protocol as a new node in the graph.
            self._nodes_by_id[protocol_name] = protocol_to_insert

            existing_node = self._nodes_by_id[protocol_name]
            self._dependants_graph[protocol_name] = []

            if len(parent_node_ids) == 0:
                self._root_nodes.append(protocol_name)
            else:

                for node_id in calculation.dependants_graph:

                    if (protocol_name not in calculation.dependants_graph[node_id] or
                        node_id in self._dependants_graph[protocol_name]):

                        continue

                    self._dependants_graph[node_id].append(protocol_name)

        return existing_node.id

    def add_calculation(self, calculation):
        """Insert a calculation into the calculation graph.

        Parameters
        ----------
        calculation : DirectCalculation
            The calculation to insert.
        """

        if calculation.uuid in self._calculations_to_run:

            # Quick sanity check.
            raise ValueError('A calculation with the same uuid ({}) is '
                             'trying to run twice.'.format(calculation.uuid))

        self._calculations_to_run[calculation.uuid] = calculation

        protocol_execution_order = graph.topological_sort(calculation.dependants_graph)

        reduced_protocol_dependants = copy.deepcopy(calculation.dependants_graph)
        graph.apply_transitive_reduction(reduced_protocol_dependants)

        parent_protocol_ids = {}

        for protocol_id in protocol_execution_order:

            parent_ids = parent_protocol_ids.get(protocol_id) or []
            inserted_id = self._insert_node(protocol_id, calculation, parent_ids)

            for dependant in reduced_protocol_dependants[protocol_id]:

                if dependant not in parent_protocol_ids:
                    parent_protocol_ids[dependant] = []

                parent_protocol_ids[dependant].append(inserted_id)

        self._calculations_to_run[calculation.uuid].update_schema()

    def submit(self, backend):
        """Submits the protocol graph to the backend of choice.

        Parameters
        ----------
        backend: PropertyEstimatorBackend
            The backend to launch the graph on.

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

            node = self._nodes_by_id[node_id]
            dependency_futures = []

            for dependency in dependencies[node_id]:
                dependency_futures.append(submitted_futures[dependency])

            # Pull out any 'global' properties.
            global_properties = {}

            for dependency in node.dependencies:

                if not dependency.is_global:
                    continue

                global_properties[dependency.property_name] = node.get_value(dependency)

            # Do a quick sanity check to make sure they've all been set already.
            if len(global_properties) > 0:
                raise ValueError('The global_properties array should be empty by this point.')

            submitted_futures[node_id] = backend.submit_task(DirectCalculationGraph._execute_protocol,
                                                             node.directory,
                                                             node.schema,
                                                             *dependency_futures)

        for calculation_id in self._calculations_to_run:

            calculation = self._calculations_to_run[calculation_id]
            calculation.update_schema()

            # TODO: Fill in any extra required provenance.
            provenance = {}

            for protocol_id in calculation.protocols:

                protocol = calculation.protocols[protocol_id]
                provenance[protocol_id] = PolymorphicDataType(protocol.schema)

            from propertyestimator.properties import CalculationSource

            source = CalculationSource(fidelity=SimulationLayer.__name__,
                                       provenance=provenance)

            calculation.physical_property.source = source

            value_node_id = calculation.final_value_source.start_protocol

            final_futures = [
                submitted_futures[value_node_id],
            ]

            for output_label in calculation.outputs_to_store:

                output_to_store = calculation.outputs_to_store[output_label]

                for attribute_key in output_to_store.__getstate__():

                    attribute_value = getattr(output_to_store, attribute_key)

                    if not isinstance(attribute_value, ProtocolPath):
                        continue

                    final_futures.append(submitted_futures[attribute_value.start_protocol])

            # Gather the values and uncertainties of each property being calculated.
            value_futures.append(backend.submit_task(DirectCalculationGraph._gather_results,
                                                     calculation.physical_property,
                                                     calculation.final_value_source,
                                                     calculation.outputs_to_store,
                                                     *final_futures))

        return value_futures

    @staticmethod
    def _execute_protocol(directory, protocol_schema, *parent_outputs, available_resources, **kwargs):
        """Executes a protocol defined by ``protocol_schema``, and with
        inputs sets via the global scope and from previously executed protocols.

        Parameters
        ----------
        protocol_schema: protocols.ProtocolSchema
            The schema defining the protocol to execute.
        parent_outputs: tuple of Any
            The results of previous protocol executions.

        Returns
        -------
        str, dict of str and Any
            Returns a tuple of the id of the executed protocol, and a dictionary
            which contains the outputs of the executed protocol.
        """

        # Store the results of the relevant previous protocols in a handy dictionary.
        # If one of the results is a failure, propagate it up the chain!
        parent_outputs_by_path = {}

        for parent_id, parent_output in parent_outputs:

            if isinstance(parent_output, PropertyEstimatorException):
                return protocol_schema.id, parent_output

            for output_path, output_value in parent_output.items():

                property_name, protocol_ids = protocols.ProtocolPath.to_components(output_path)

                if len(protocol_ids) == 0 or (len(protocol_ids) > 0 and protocol_ids[0] != parent_id):
                    protocol_ids.insert(0, parent_id)

                final_path = protocols.ProtocolPath(property_name, *protocol_ids)
                parent_outputs_by_path[final_path] = output_value

        # Recreate the protocol on the backend to bypass the need for static methods
        # and awkward args and kwargs syntax.
        protocol = plugins.available_protocols[protocol_schema.type](protocol_schema.id)
        protocol.schema = protocol_schema

        if not path.isdir(directory):
            makedirs(directory)

        for input_path in protocol.required_inputs:

            target_paths = protocol.get_value_references(input_path)
            values = {}

            for target_path in target_paths:

                if (target_path.start_protocol == input_path.start_protocol or
                    target_path.start_protocol == protocol.id):

                    continue

                values[target_path] = parent_outputs_by_path[target_path]

            input_value = protocol.get_value(input_path)

            if isinstance(input_value, ProtocolPath) and input_value in values:
                protocol.set_value(input_path, values[input_value])

            elif isinstance(input_value, list):

                value_list = []

                for target_value in input_value:

                    if not isinstance(target_value, ProtocolPath):
                        value_list.append(target_value)
                    elif target_value in values:
                        value_list.append(values[target_value])
                    else:
                        value_list.append(target_value)

                protocol.set_value(input_path, value_list)

        if isinstance(protocol, groups.ConditionalGroup):

            for condition in protocol.conditions:

                left_value = condition.left_hand_value

                if (isinstance(left_value, protocols.ProtocolPath) and 
                    left_value.start_protocol is not None and
                    left_value.start_protocol != protocol.id):
                    
                    condition.left_hand_value = parent_outputs_by_path[left_value]

                right_value = condition.right_hand_value

                if (isinstance(right_value, protocols.ProtocolPath) and
                        right_value.start_protocol is not None and
                        right_value.start_protocol != protocol.id):

                    condition.right_hand_value = parent_outputs_by_path[right_value]

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
            The result of attempting to estimate this property by direct simulation.
        """

        return_object = CalculationLayerResult()
        return_object.property_id = property_to_return.id

        results_by_id = {}

        for protocol_id, protocol_results in protocol_results:

            # Make sure none of the protocols failed and we actually have a value
            # and uncertainty.
            if isinstance(protocol_results, PropertyEstimatorException):

                return_object.calculation_error = protocol_results
                return return_object

            for output_path, output_value in protocol_results.items():

                property_name, protocol_ids = protocols.ProtocolPath.to_components(output_path)

                if len(protocol_ids) == 0 or (len(protocol_ids) > 0 and protocol_ids[0] != protocol_id):
                    protocol_ids.insert(0, protocol_id)

                final_path = protocols.ProtocolPath(property_name, *protocol_ids)
                results_by_id[final_path] = output_value

        property_to_return.value = results_by_id[value_reference].value
        property_to_return.uncertainty = results_by_id[value_reference].uncertainty

        return_object.calculated_property = property_to_return
        return_object.data_to_store = []

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


@register_calculation_layer()
class SimulationLayer(PropertyCalculationLayer):
    """A calculation layer which aims to calculate physical properties
    directly from molecular simulation.

    .. warning :: This class is experimental and should not be used in a production environment.
    """

    @staticmethod
    def _build_calculation_graph(working_directory, properties, force_field_path, options):
        """ Construct a graph of the protocols needed to calculate a set of properties.

        Parameters
        ----------
        working_directory: str
            The local directory in which to store all local,
            temporary calculation data from this graph.
        properties : list of PhysicalProperty
            The properties to attempt to compute.
        force_field_path : str
            The path to the force field parameters to use in the calculation.
        options: PropertyEstimatorOptions
            The options to run the calculations with.
        """
        calculation_graph = DirectCalculationGraph(working_directory)

        for property_to_calculate in properties:

            property_type = type(property_to_calculate).__name__

            if property_type not in options.workflow_schemas:

                logging.warning('The property calculator does not support {} '
                                'calculations.'.format(property_type))

                continue

            schema = options.workflow_schemas[property_type]

            calculation = DirectCalculation(property_to_calculate,
                                            force_field_path,
                                            schema,
                                            options)

            calculation_graph.add_calculation(calculation)

        return calculation_graph

    @staticmethod
    def schedule_calculation(calculation_backend, storage_backend, layer_directory,
                             data_model, callback, synchronous=False):

        # Store a temporary copy of the force field for protocols to easily access.
        force_field = storage_backend.retrieve_force_field(data_model.force_field_id)
        force_field_path = path.join(layer_directory, 'force_field_{}'.format(data_model.force_field_id))

        with open(force_field_path, 'wb') as file_object:
            pickle.dump(serialize_force_field(force_field), file_object)

        calculation_graph = SimulationLayer._build_calculation_graph(layer_directory,
                                                                     data_model.queued_properties,
                                                                     force_field_path,
                                                                     data_model.options)

        simulation_futures = calculation_graph.submit(calculation_backend)

        PropertyCalculationLayer._await_results(calculation_backend, storage_backend, layer_directory,
                                                data_model, callback, simulation_futures, synchronous)
