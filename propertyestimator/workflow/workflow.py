"""
Defines the core workflow object and execution graph.
"""
import copy
import json
import logging
import math
import time
import uuid
from math import sqrt
from os import makedirs, path
from shutil import copy as file_copy

from propertyestimator import unit
from propertyestimator.attributes import UNDEFINED
from propertyestimator.forcefield import ForceFieldSource, SmirnoffForceFieldSource
from propertyestimator.storage.attributes import FilePath, StorageAttribute
from propertyestimator.substances import Substance
from propertyestimator.utils import graph
from propertyestimator.utils.exceptions import EvaluatorException
from propertyestimator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder
from propertyestimator.utils.string import extract_variable_index_and_name
from propertyestimator.utils.utils import get_nested_attribute
from propertyestimator.workflow.exceptions import WorkflowException
from propertyestimator.workflow.protocols import ProtocolGraph
from propertyestimator.workflow.schemas import (
    ProtocolReplicator,
    ProtocolSchema,
    WorkflowSchema,
)
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue


class Workflow:
    """Encapsulates and prepares a workflow which is able to estimate
    a physical property.
    """

    @property
    def protocols(self):
        """tuple of Protocol: The protocols in this workflow."""
        return {x.id: x for x in self._protocols}

    @property
    def schema(self):
        return self._get_schema()

    @schema.setter
    def schema(self, value):
        self._set_schema(value)

    def __init__(self, global_metadata, unique_id=None):
        """
        Constructs a new Workflow object.

        Parameters
        ----------
        global_metadata: dict of str and Any
            A dictionary of the metadata which will be made available to each
            of the workflow protocols through the pseudo "global" scope.
        unique_id: str, optional
            A unique identifier to assign to this workflow. This id will be appended
            to the ids of the protocols of this workflow. If none is provided,
            one will be chosen at random.
        """
        assert global_metadata is not None
        self._global_metadata = global_metadata

        if unique_id is None:
            unique_id = str(uuid.uuid4()).replace("-", "")

        self.uuid = unique_id

        self._protocols = []
        self._final_value_source = None
        self._gradients_sources = []
        self._outputs_to_store = {}

    def _get_schema(self):
        """Returns the schema that describes this workflow.

        Returns
        -------
        WorkflowSchema
            The schema that describes this workflow.
        """
        schema = WorkflowSchema()

        schema.id = self.uuid
        schema.protocol_schemas = {x.id: x.schema for x in self._protocols}

        if self._final_value_source != UNDEFINED:
            schema.final_value_source = self._final_value_source.copy()

        schema.gradients_sources = [source.copy() for source in self._gradients_sources]
        schema.outputs_to_store = copy.deepcopy(self._outputs_to_store)

        return schema

    def _set_schema(self, schema):
        """Sets this workflow's properties from a `WorkflowSchema`.

        Parameters
        ----------
        schema: WorkflowSchema
            The schema which outlines this steps in this workflow.
        """
        # Copy the schema.
        schema = WorkflowSchema.parse_json(schema.json())

        if schema.final_value_source != UNDEFINED:

            self._final_value_source = schema.final_value_source
            self._final_value_source.append_uuid(self.uuid)

        self._build_protocols(schema)

        self._gradients_sources = []

        if schema.gradients_sources != UNDEFINED:

            for gradient_source in schema.gradients_sources:
                gradient_source.append_uuid(self.uuid)
                self._gradients_sources.append(gradient_source)

        self._outputs_to_store = {}

        if schema.outputs_to_store != UNDEFINED:

            for label in schema.outputs_to_store:

                self._append_uuid_to_output_to_store(schema.outputs_to_store[label])
                self._outputs_to_store[label] = self._build_output_to_store(
                    schema.outputs_to_store[label]
                )

    def _append_uuid_to_output_to_store(self, output_to_store):
        """Appends this workflows uuid to all of the protocol paths
        within an output to store, and all of its child outputs.

        Parameters
        ----------
        output_to_store: BaseStoredData
            The output to store to append the uuid to.
        """

        for attribute_name in output_to_store.get_attributes(StorageAttribute):

            attribute_value = getattr(output_to_store, attribute_name)

            if not isinstance(attribute_value, ProtocolPath):
                continue

            attribute_value.append_uuid(self.uuid)

    def _build_output_to_store(self, output_to_store):
        """Sets the inputs of a `BaseStoredData` object which
        are taken from the global metadata.

        Parameters
        ----------
        output_to_store: BaseStoredData
            The output to set the inputs of.

        Returns
        -------
        BaseStoredData
            The built object with all of its inputs correctly set.
        """

        for attribute_name in output_to_store.get_attributes(StorageAttribute):

            attribute_value = getattr(output_to_store, attribute_name)

            if (
                not isinstance(attribute_value, ProtocolPath)
                or not attribute_value.is_global
            ):
                continue

            attribute_value = get_nested_attribute(
                self._global_metadata, attribute_value.property_name
            )
            setattr(output_to_store, attribute_name, attribute_value)

        return output_to_store

    def _build_protocols(self, schema):
        """Creates a set of protocols based on a WorkflowSchema.

        Parameters
        ----------
        schema: WorkflowSchema
            The schema to use when creating the protocols
        """
        self._apply_replicators(schema)

        for protocol_schema in schema.protocol_schemas:

            protocol = protocol_schema.to_protocol()

            # Try to set global properties on each of the protocols
            for input_path in protocol.required_inputs:

                value_references = protocol.get_value_references(input_path)

                for source_path, value_reference in value_references.items():

                    if not value_reference.is_global:
                        continue

                    value = get_nested_attribute(
                        self._global_metadata, value_reference.property_name
                    )
                    protocol.set_value(source_path, value)

            protocol.set_uuid(self.uuid)
            self._protocols.append(protocol)

    def _get_template_values(self, replicator):
        """Returns the values which which will be passed to the replicated
        protocols, evaluating any protocol paths to retrieve the referenced
        values.

        Parameters
        ----------
        replicator: ProtocolReplicator
            The replictor which is replicating the protocols.

        Returns
        -------
        Any
            The template values.
        """

        invalid_value_error = ValueError(
            f"Template values must either be a constant or come "
            f"from the global scope (and not from {replicator.template_values})"
        )

        # Get the list of values which will be passed to the newly created protocols.
        if isinstance(replicator.template_values, ProtocolPath):

            if not replicator.template_values.is_global:
                raise invalid_value_error

            return get_nested_attribute(
                self._global_metadata, replicator.template_values.property_name
            )

        elif not isinstance(replicator.template_values, list):
            raise NotImplementedError()

        evaluated_template_values = []

        for template_value in replicator.template_values:

            if not isinstance(template_value, ProtocolPath):

                evaluated_template_values.append(template_value)
                continue

            if not template_value.is_global:
                raise invalid_value_error

            evaluated_template_values.append(
                get_nested_attribute(
                    self._global_metadata, template_value.property_name
                )
            )

        return evaluated_template_values

    def _apply_replicators(self, schema):
        """Applies each of the protocol replicators in turn to the schema.

        Parameters
        ----------
        schema: WorkflowSchema
            The schema to apply the replicators to.
        """

        if schema.protocol_replicators == UNDEFINED:
            return

        while len(schema.protocol_replicators) > 0:

            replicator = schema.protocol_replicators.pop(0)

            # Apply this replicator
            self._apply_replicator(schema, replicator)

            if schema.json().find(replicator.placeholder_id) >= 0:

                raise RuntimeError(
                    f"The {replicator.id} replicator was not fully applied."
                )

    def _apply_replicator(self, schema, replicator):
        """A method to create a set of protocol schemas based on a ProtocolReplicator,
        and add them to the list of existing schemas.

        Parameters
        ----------
        schema: WorkflowSchema
            The schema which contains the protocol definitions
        replicator: `ProtocolReplicator`
            The replicator which describes which new protocols should
            be created.
        """

        # Get the list of values which will be passed to the newly created protocols.
        template_values = self._get_template_values(replicator)

        # Replicate the protocols.
        protocols = {}

        for protocol_schema in schema.protocol_schemas:

            protocol = protocol_schema.to_protocol()
            protocols[protocol.id] = protocol

        replicated_protocols, replication_map = replicator.apply(
            protocols, template_values
        )
        replicator.update_references(
            replicated_protocols, replication_map, template_values
        )

        # Update the schema with the replicated protocols.
        schema.protocol_schemas = [
            replicated_protocols[key].schema for key in replicated_protocols
        ]

        # Make sure to correctly replicate gradient sources.
        replicated_gradient_sources = []

        for gradient_source in schema.gradients_sources:

            if replicator.placeholder_id not in gradient_source.full_path:

                replicated_gradient_sources.append(gradient_source)
                continue

            for index, template_value in enumerate(template_values):

                replicated_source = ProtocolPath.from_string(
                    gradient_source.full_path.replace(
                        replicator.placeholder_id, str(index)
                    )
                )

                replicated_gradient_sources.append(replicated_source)

        schema.gradients_sources = replicated_gradient_sources

        # Replicate any outputs.
        self._apply_replicator_to_outputs(replicator, schema, template_values)
        # Replicate any replicators.
        self._apply_replicator_to_replicators(replicator, schema, template_values)

    @staticmethod
    def _apply_replicator_to_outputs(replicator, schema, template_values):
        """Applies a replicator to a schema outputs to store.

        Parameters
        ----------
        replicator: ProtocolReplicator
            The replicator to apply.
        schema: WorkflowSchema
            The schema which defines the outputs to store.
        template_values: List of Any
            The values being applied by the replicator.
        """

        outputs_to_replicate = []

        if schema.outputs_to_store != UNDEFINED:

            outputs_to_replicate = [
                label
                for label in schema.outputs_to_store
                if label.find(replicator.id) >= 0
            ]

        # Check to see if there are any outputs to store pointing to
        # protocols which are being replicated.
        for output_label in outputs_to_replicate:

            output_to_replicate = schema.outputs_to_store.pop(output_label)

            for index, template_value in enumerate(template_values):

                replicated_label = output_label.replace(
                    replicator.placeholder_id, str(index)
                )
                replicated_output = copy.deepcopy(output_to_replicate)

                for attribute_name in replicated_output.get_attributes(
                    StorageAttribute
                ):

                    attribute_value = getattr(replicated_output, attribute_name)

                    if isinstance(attribute_value, ProtocolPath):

                        attribute_value = ProtocolPath.from_string(
                            attribute_value.full_path.replace(
                                replicator.placeholder_id, str(index)
                            )
                        )

                    elif isinstance(attribute_value, ReplicatorValue):

                        if attribute_value.replicator_id != replicator.id:

                            # Make sure to handle nested dependent replicators.
                            attribute_value.replicator_id = attribute_value.replicator_id.replace(
                                replicator.placeholder_id, str(index)
                            )

                            continue

                        attribute_value = template_value

                    setattr(replicated_output, attribute_name, attribute_value)

                schema.outputs_to_store[replicated_label] = replicated_output

    @staticmethod
    def _apply_replicator_to_replicators(replicator, schema, template_values):
        """Applies a replicator to any replicators which depend upon
        it (e.g. replicators with ids similar to `other_id_$(replicator.id)`).

        Parameters
        ----------
        replicator: ProtocolReplicator
            The replicator being applied.
        schema: WorkflowSchema
            The workflow schema to which the replicator belongs.
        template_values: List of Any
            The values which the replicator is applying.
        """

        # Look over all of the replicators left to apply and update them
        # to point to the newly replicated protocols where appropriate.
        new_indices = [str(index) for index in range(len(template_values))]

        replicators = []

        for original_replicator in schema.protocol_replicators:

            # Check whether this replicator will be replicated.
            if replicator.placeholder_id not in original_replicator.id:

                replicators.append(original_replicator)
                continue

            # Create the replicated replicators
            for template_index in new_indices:

                replicator_id = original_replicator.id.replace(
                    replicator.placeholder_id, template_index
                )

                new_replicator = ProtocolReplicator(replicator_id)
                new_replicator.template_values = original_replicator.template_values

                # Make sure to replace any reference to the applied replicator
                # with the actual index.
                if isinstance(new_replicator.template_values, ProtocolPath):

                    updated_path = new_replicator.template_values.full_path.replace(
                        replicator.placeholder_id, template_index
                    )

                    new_replicator.template_values = ProtocolPath.from_string(
                        updated_path
                    )

                elif isinstance(new_replicator.template_values, list):

                    updated_values = []

                    for template_value in new_replicator.template_values:

                        if not isinstance(template_value, ProtocolPath):

                            updated_values.append(template_value)
                            continue

                        updated_path = template_value.full_path.replace(
                            replicator.placeholder_id, template_index
                        )
                        updated_values.append(ProtocolPath.from_string(updated_path))

                    new_replicator.template_values = updated_values

                replicators.append(new_replicator)

        schema.protocol_replicators = replicators

    def replace_protocol(self, old_protocol, new_protocol):
        """Replaces an existing protocol with a new one, while
        updating all input and local references to point to the
        new protocol.

        The main use of this method is when merging multiple protocols
        into one.

        Parameters
        ----------
        old_protocol : Protocol
            The protocol (or its id) to replace.
        new_protocol : Protocol
            The new protocol (or its id) to use.
        """

        if new_protocol.id in self._protocols:

            raise ValueError(
                "A protocol with the same id already exists in this workflow."
            )

        self._protocols.remove(old_protocol)
        self._protocols.append(new_protocol)

        for protocol in self._protocols:
            protocol.replace_protocol(old_protocol.id, new_protocol.id)

        if self._final_value_source is not None:
            self._final_value_source.replace_protocol(old_protocol.id, new_protocol.id)

        for gradient_source in self._gradients_sources:
            gradient_source.replace_protocol(old_protocol.id, new_protocol.id)

        for output_label in self._outputs_to_store:

            output_to_store = self._outputs_to_store[output_label]

            for attribute_name in output_to_store.get_attributes(StorageAttribute):

                attribute_value = getattr(output_to_store, attribute_name)

                if not isinstance(attribute_value, ProtocolPath):
                    continue

                attribute_value.replace_protocol(old_protocol.id, new_protocol.id)

    @staticmethod
    def _find_relevant_gradient_keys(
        substance, force_field_path, parameter_gradient_keys
    ):
        """Extract only those keys which may be applied to the
        given substance.

        Parameters
        ----------
        substance: Substance
            The substance to compare against.
        force_field_path: str
            The path to the force field which contains the parameters.
        parameter_gradient_keys: list of ParameterGradientKey
            The original list of parameter gradient keys.

        Returns
        -------
        list of ParameterGradientKey
            The filtered list of parameter gradient keys.
        """
        from openforcefield.topology import Molecule, Topology

        # noinspection PyTypeChecker
        if parameter_gradient_keys is None or len(parameter_gradient_keys) == 0:
            return []

        with open(force_field_path) as file:
            force_field_source = ForceFieldSource.parse_json(file.read())

        if not isinstance(force_field_source, SmirnoffForceFieldSource):
            return []

        force_field = force_field_source.to_force_field()

        all_molecules = []

        for component in substance.components:
            all_molecules.append(Molecule.from_smiles(component.smiles))

        topology = Topology.from_molecules(all_molecules)
        labelled_molecules = force_field.label_molecules(topology)

        reduced_parameter_keys = []

        for labelled_molecule in labelled_molecules:

            for parameter_key in parameter_gradient_keys:

                if (
                    parameter_key.tag not in labelled_molecule
                    or parameter_key in reduced_parameter_keys
                ):
                    continue

                contains_parameter = False

                for parameter in labelled_molecule[parameter_key.tag].store.values():

                    if parameter.smirks != parameter_key.smirks:
                        continue

                    contains_parameter = True
                    break

                if not contains_parameter:
                    continue

                reduced_parameter_keys.append(parameter_key)

        return reduced_parameter_keys

    @staticmethod
    def generate_default_metadata(
        physical_property,
        force_field_path,
        parameter_gradient_keys=None,
        target_uncertainty=None,
    ):
        """Generates the default global metadata dictionary.

        Parameters
        ----------
        physical_property: PhysicalProperty
            The physical property whose arguments are available in the
            global scope.
        force_field_path: str
            The path to the force field parameters to use in the workflow.
        parameter_gradient_keys: list of ParameterGradientKey
                A list of references to all of the parameters which all observables
                should be differentiated with respect to.
        target_uncertainty: unit.Quantity, optional
            The uncertainty which the property should be estimated to
            within.

        Returns
        -------
        dict of str, Any

            The metadata dictionary, with the following
            keys / types:

            - thermodynamic_state: `ThermodynamicState` - The state (T,p) at which the
                                                          property is being computed
            - substance: `Substance` - The composition of the system of interest.
            - components: list of `Substance` - The components present in the system for
                                              which the property is being estimated.
            - target_uncertainty: propertyestimator.unit.Quantity - The target uncertainty with which
                                                        properties should be estimated.
            - per_component_uncertainty: propertyestimator.unit.Quantity - The target uncertainty divided
                                                               by the sqrt of the number of
                                                               components in the system + 1
            - force_field_path: str - A path to the force field parameters with which the
                                      property should be evaluated with.
            - parameter_gradient_keys: list of ParameterGradientKey - A list of references to all of the
                                                                      parameters which all observables
                                                                      should be differentiated with respect to.
        """
        components = []

        for component in physical_property.substance.components:

            component_substance = Substance.from_components(component)
            components.append(component_substance)

        if target_uncertainty is None:
            target_uncertainty = math.inf * physical_property.value.units

        target_uncertainty = target_uncertainty.to(physical_property.value.units)

        # +1 comes from inclusion of the full mixture as a possible component.
        per_component_uncertainty = target_uncertainty / sqrt(
            physical_property.substance.number_of_components + 1
        )

        # Find only those gradient keys which will actually be relevant to the
        # property of interest
        relevant_gradient_keys = Workflow._find_relevant_gradient_keys(
            physical_property.substance, force_field_path, parameter_gradient_keys
        )

        # Define a dictionary of accessible 'global' properties.
        global_metadata = {
            "thermodynamic_state": physical_property.thermodynamic_state,
            "substance": physical_property.substance,
            "components": components,
            "target_uncertainty": target_uncertainty,
            "per_component_uncertainty": per_component_uncertainty,
            "force_field_path": force_field_path,
            "parameter_gradient_keys": relevant_gradient_keys,
        }

        # Include the properties metadata
        if physical_property.metadata != UNDEFINED:
            global_metadata.update(physical_property.metadata)

        return global_metadata

    def to_graph(self):
        """Converts this workflow to an executable `WorkflowGraph`.

        Returns
        -------
        WorkflowGraph
            The graph representation of this workflow.
        """
        graph = WorkflowGraph()
        graph.add_workflow(self)
        return graph


class WorkflowGraph:
    """A hierarchical structure for storing and submitting the workflows
    which will estimate a set of physical properties..
    """

    @property
    def protocols(self):
        """dict of str and Protocol: The protocols in this graph."""
        return self._protocol_graph.protocols

    @property
    def root_protocols(self):
        """list of str: The ids of the protocols in the group which do not
        take input from the other grouped protocols."""
        return self._protocol_graph.root_protocols

    @property
    def dependants_graph(self):
        """dict of str and str: A dictionary of which stores which grouped protocols
        are dependant on other grouped protocols. Each key in the dictionary is the
        id of a grouped protocol, and each value is the id of a protocol which depends
        on the protocol by the key."""
        return self._protocol_graph.dependants_graph

    def __init__(self):

        super(WorkflowGraph, self).__init__()

        self._workflows_to_execute = {}
        self._protocol_graph = ProtocolGraph()

    def add_workflow(self, workflow):
        """Insert a workflow into the workflow graph.

        Parameters
        ----------
        workflow : Workflow
            The workflow to insert.
        """

        if workflow.uuid in self._workflows_to_execute:

            raise ValueError(
                f"A workflow with the uuid {workflow.uuid} is already in the graph."
            )

        original_protocols = [*workflow.protocols.values()]
        self._workflows_to_execute[workflow.uuid] = workflow

        # Add the workflow protocols to the graph.
        merged_protocol_ids = self._protocol_graph.add_protocols(
            *original_protocols, allow_external_dependencies=False
        )

        # Update the workflow to use the possibly merged protocols
        for original_id, new_id in merged_protocol_ids.items():

            if original_id not in workflow.protocols:
                # Skip nested protocols (i.e. those in ProtocolGroup's).
                continue

            workflow.replace_protocol(
                workflow.protocols[original_id], self._protocol_graph.protocols[new_id]
            )

    def submit(self, root_directory, backend):
        """Submits the protocol graph to the backend of choice.

        Parameters
        ----------
        root_directory: str
            The directory to execute the graph in.
        backend: CalculationBackend, optional.
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
        submission_order = graph.topological_sort(self._inner_graph.dependants_graph)

        # Build a dependency graph from the dependants graph so that
        # futures can be passed in the correct place.
        dependencies = graph.dependants_to_dependencies(
            self._inner_graph.dependants_graph
        )

        # TODO: Use ProtocolGraph execution code?
        raise NotImplementedError()

        for node_id in submission_order:

            node = self._protocols_by_id[node_id]
            dependency_futures = []

            for dependency in dependencies[node_id]:
                dependency_futures.append(submitted_futures[dependency])

            submitted_futures[node_id] = backend.submit_task(
                WorkflowGraph._execute_protocol,
                node.directory,
                node.schema.json(),
                *dependency_futures,
                key=f"execute_{node_id}",
            )

        for workflow_id in self._workflows_to_execute:

            workflow = self._workflows_to_execute[workflow_id]

            # TODO: Fill in any extra required provenance.
            provenance = {}

            for protocol_id in workflow.protocols:

                protocol = workflow.protocols[protocol_id]
                provenance[protocol_id] = protocol.schema

            workflow.physical_property.source.provenance = provenance

            final_futures = []

            # Make sure we keep track of all of the futures which we
            # will use to populate things such as a final property value
            # or gradient keys.
            if workflow.final_value_source is not None:

                value_node_id = workflow.final_value_source.start_protocol
                final_futures.append(submitted_futures[value_node_id])

            for gradient_source in workflow.gradients_sources:

                protocol_id = gradient_source.start_protocol
                final_futures.append(submitted_futures[protocol_id])

            for output_label in workflow.outputs_to_store:

                output_to_store = workflow.outputs_to_store[output_label]

                for attribute_name in output_to_store.get_attributes(StorageAttribute):

                    attribute_value = getattr(output_to_store, attribute_name)

                    if not isinstance(attribute_value, ProtocolPath):
                        continue

                    final_futures.append(
                        submitted_futures[attribute_value.start_protocol]
                    )

            if len(final_futures) == 0:
                final_futures = [submitted_futures[key] for key in submitted_futures]

            # TODO
            value_futures.append(
                backend.submit_task(
                    WorkflowGraph._gather_results,
                    root_directory,
                    workflow.physical_property,
                    workflow.final_value_source,
                    workflow.gradients_sources,
                    workflow.outputs_to_store,
                    *final_futures,
                )
            )

        return value_futures

    @staticmethod
    def _save_protocol_output(file_path, output_dictionary):
        """Saves the results of executing a protocol (whether these be the true
        results or an exception) as a JSON file to disk.

        Parameters
        ----------
        file_path: str
            The path to save the output to.
        output_dictionary: dict of str and Any
            The results in the form of a dictionary which can be serialized
            by the `TypedJSONEncoder`
        """

        with open(file_path, "w") as file:
            json.dump(output_dictionary, file, cls=TypedJSONEncoder)

    @staticmethod
    def _execute_protocol(
        directory,
        protocol_schema_json,
        *previous_output_paths,
        available_resources,
        **_,
    ):
        """Executes a protocol whose state is defined by the ``protocol_schema``.

        Parameters
        ----------
        protocol_schema_json: str
            The JSON schema defining the protocol to execute.
        previous_output_paths: tuple of str
            Paths to the results of previous protocol executions.

        Returns
        -------
        str
            The id of the executed protocol.
        dict of str and Any
            A dictionary which contains the outputs of the executed protocol.
        """
        protocol_schema = ProtocolSchema.parse_json(protocol_schema_json)

        # The path where the output of this protocol will be stored.
        output_dictionary_path = path.join(
            directory, "{}_output.json".format(protocol_schema.id)
        )
        makedirs(directory, exist_ok=True)

        # We need to make sure ALL exceptions are handled within this method,
        # or any function which will be executed on a calculation backend to
        # avoid accidentally killing the backend.
        try:

            # If the output file already exists, we can assume this protocol has already
            # been executed and we can return immediately without re-executing.
            if path.isfile(output_dictionary_path):
                return protocol_schema.id, output_dictionary_path

            # Store the results of the relevant previous protocols in a handy dictionary.
            # If one of the results is a failure, propagate it up the chain.
            previous_outputs_by_path = {}

            for parent_id, previous_output_path in previous_output_paths:

                try:

                    with open(previous_output_path, "r") as file:
                        parent_output = json.load(file, cls=TypedJSONDecoder)

                except json.JSONDecodeError as e:

                    exception = EvaluatorException.from_exception(e)

                    WorkflowGraph._save_protocol_output(
                        output_dictionary_path, exception
                    )

                    return protocol_schema.id, output_dictionary_path

                if isinstance(parent_output, EvaluatorException):
                    return protocol_schema.id, previous_output_path

                for output_path, output_value in parent_output.items():

                    property_name, protocol_ids = ProtocolPath.to_components(
                        output_path
                    )

                    if len(protocol_ids) == 0 or (
                        len(protocol_ids) > 0 and protocol_ids[0] != parent_id
                    ):
                        protocol_ids.insert(0, parent_id)

                    final_path = ProtocolPath(property_name, *protocol_ids)
                    previous_outputs_by_path[final_path] = output_value

            # Recreate the protocol on the backend to bypass the need for static methods
            # and awkward args and kwargs syntax.
            protocol = protocol_schema.to_protocol()

            # Pass the outputs of previously executed protocols as input to the
            # protocol to execute.
            for input_path in protocol.required_inputs:

                value_references = protocol.get_value_references(input_path)

                for source_path, target_path in value_references.items():

                    if (
                        target_path.start_protocol == input_path.start_protocol
                        or target_path.start_protocol == protocol.id
                    ):

                        continue

                    property_name = target_path.property_name
                    property_index = None

                    nested_property_name = None

                    if property_name.find(".") > 0:

                        nested_property_name = ".".join(property_name.split(".")[1:])
                        property_name = property_name.split(".")[0]

                    if property_name.find("[") >= 0 or property_name.find("]") >= 0:
                        property_name, property_index = extract_variable_index_and_name(
                            property_name
                        )

                    _, target_protocol_ids = ProtocolPath.to_components(
                        target_path.full_path
                    )

                    target_value = previous_outputs_by_path[
                        ProtocolPath(property_name, *target_protocol_ids)
                    ]

                    if property_index is not None:
                        target_value = target_value[property_index]

                    if nested_property_name is not None:
                        target_value = get_nested_attribute(
                            target_value, nested_property_name
                        )

                    protocol.set_value(source_path, target_value)

            logging.info("Executing protocol: {}".format(protocol.id))

            start_time = time.perf_counter()
            output_dictionary = protocol.execute(directory, available_resources)
            end_time = time.perf_counter()

            logging.info(
                "Protocol finished executing ({} ms): {}".format(
                    (end_time - start_time) * 1000, protocol.id
                )
            )

            try:

                WorkflowGraph._save_protocol_output(
                    output_dictionary_path, output_dictionary
                )

            except TypeError as e:

                exception = EvaluatorException.from_exception(e)
                WorkflowGraph._save_protocol_output(output_dictionary_path, exception)

            return protocol.id, output_dictionary_path

        except Exception as e:

            logging.info(f"Protocol failed to execute: {protocol_schema.id}")

            exception = WorkflowException.from_exception(e)
            exception.protocol_id = protocol_schema.id

            WorkflowGraph._save_protocol_output(output_dictionary_path, exception)
            return protocol_schema.id, output_dictionary_path

    @staticmethod
    def _gather_results(
        directory,
        property_to_return,
        value_reference,
        gradient_sources,
        outputs_to_store,
        target_uncertainty,
        *protocol_result_paths,
        **_,
    ):
        """Gather the value and uncertainty calculated from the submission graph
        and store them in the property to return.

        Parameters
        ----------
        directory: str
            The directory to store any working files in.
        property_to_return: PhysicalProperty
            The property to which the value and uncertainty belong.
        value_reference: ProtocolPath, optional
            A reference to which property in the output dictionary is the actual value.
        gradient_sources: list of ProtocolPath
            A list of references to those entries in the output dictionaries which correspond
            to parameter gradients.
        outputs_to_store: dict of string and WorkflowOutputToStore
            A list of references to data which should be stored on the storage backend.
        target_uncertainty: unit.Quantity, optional
            The uncertainty within which this property should have been estimated. If this
            value is not `None` and the target has not been met, a `None` result will be returned
            indicating that this property could not be estimated by the workflow, but not because
            of an error.
        protocol_results: dict of string and str
            The result dictionary of the protocol which calculated the value of the property.

        Returns
        -------
        CalculationLayerResult, optional
            The result of attempting to estimate this property from a workflow graph. `None`
            will be returned if the target uncertainty is set but not met.
        """
        from propertyestimator.layers.layers import CalculationLayerResult

        if target_uncertainty is not None:
            target_uncertainty = unit.Quantity.from_tuple(target_uncertainty)

        return_object = CalculationLayerResult()
        return_object.property_id = property_to_return.id

        try:
            results_by_id = {}

            for protocol_id, protocol_result_path in protocol_result_paths:

                try:

                    with open(protocol_result_path, "r") as file:
                        protocol_results = json.load(file, cls=TypedJSONDecoder)

                except json.JSONDecodeError as e:

                    return_object.exception = EvaluatorException.from_exception(e)
                    return return_object

                # Make sure none of the protocols failed and we actually have a value
                # and uncertainty.
                if isinstance(protocol_results, EvaluatorException):

                    return_object.exception = protocol_results
                    return return_object

                for output_path, output_value in protocol_results.items():

                    property_name, protocol_ids = ProtocolPath.to_components(
                        output_path
                    )

                    if len(protocol_ids) == 0 or (
                        len(protocol_ids) > 0 and protocol_ids[0] != protocol_id
                    ):
                        protocol_ids.insert(0, protocol_id)

                    final_path = ProtocolPath(property_name, *protocol_ids)
                    results_by_id[final_path] = output_value

            if value_reference is not None:

                if (
                    target_uncertainty is not None
                    and results_by_id[value_reference].uncertainty > target_uncertainty
                ):

                    logging.info(
                        "The final uncertainty ({}) was not less than the target threshold ({}).".format(
                            results_by_id[value_reference].uncertainty,
                            target_uncertainty,
                        )
                    )

                    return None

                property_to_return.value = results_by_id[value_reference].value
                property_to_return.uncertainty = results_by_id[
                    value_reference
                ].uncertainty

            for gradient_source in gradient_sources:

                gradient = results_by_id[gradient_source]
                property_to_return.gradients.append(gradient)

            return_object.calculated_property = property_to_return
            return_object.data_to_store = []

            for output_to_store in outputs_to_store.values():

                unique_id = str(uuid.uuid4()).replace("-", "")

                data_object_path = path.join(
                    directory, f"results_{property_to_return.id}{unique_id}.json"
                )
                data_directory = path.join(
                    directory, f"results_{property_to_return.id}{unique_id}"
                )

                WorkflowGraph._store_output_data(
                    data_object_path, data_directory, output_to_store, results_by_id,
                )

                return_object.data_to_store.append((data_object_path, data_directory))

        except Exception as e:
            return_object.exception = EvaluatorException.from_exception(e)

        return return_object

    @staticmethod
    def _store_output_data(
        data_object_path, data_directory, output_to_store, results_by_id,
    ):

        """Collects all of the simulation to store, and saves it into a directory
        whose path will be passed to the storage backend to process.

        Parameters
        ----------
        data_object_path: str
            The file path to serialize the data object to.
        data_directory: str
            The path of the directory to store ancillary data in.
        output_to_store: BaseStoredData
            An object which contains `ProtocolPath`s pointing to the
            data to store.
        results_by_id: dict of ProtocolPath and any
            The results of the protocols which formed the property
            estimation workflow.
        """

        makedirs(data_directory, exist_ok=True)

        for attribute_name in output_to_store.get_attributes(StorageAttribute):

            attribute = getattr(output_to_store.__class__, attribute_name)
            attribute_value = getattr(output_to_store, attribute_name)

            if not isinstance(attribute_value, ProtocolPath):
                continue

            attribute_value = results_by_id[attribute_value]

            if issubclass(attribute.type_hint, FilePath):
                file_copy(attribute_value, data_directory)
                attribute_value = path.basename(attribute_value)

            setattr(output_to_store, attribute_name, attribute_value)

        with open(data_object_path, "w") as file:
            json.dump(output_to_store, file, cls=TypedJSONEncoder)
