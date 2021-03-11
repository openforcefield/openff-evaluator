"""
Defines the core workflow object and execution graph.
"""
import copy
import json
import math
import uuid
from math import sqrt
from os import makedirs, path
from shutil import copy as file_copy

from openff.evaluator import unit
from openff.evaluator.attributes import UNDEFINED, Attribute, AttributeClass
from openff.evaluator.backends import ComputeResources
from openff.evaluator.forcefield import (
    ForceFieldSource,
    ParameterGradient,
    SmirnoffForceFieldSource,
    TLeapForceFieldSource,
)
from openff.evaluator.protocols.paprika.forcefield import GAFFForceField
from openff.evaluator.storage.attributes import FilePath, StorageAttribute
from openff.evaluator.substances import Substance
from openff.evaluator.utils.exceptions import EvaluatorException
from openff.evaluator.utils.graph import retrieve_uuid
from openff.evaluator.utils.observables import (
    Observable,
    ObservableArray,
    ObservableFrame,
)
from openff.evaluator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder
from openff.evaluator.utils.utils import get_nested_attribute
from openff.evaluator.workflow import Protocol, ProtocolGraph
from openff.evaluator.workflow.schemas import ProtocolReplicator, WorkflowSchema
from openff.evaluator.workflow.utils import ProtocolPath, ReplicatorValue


class Workflow:
    """Encapsulates and prepares a workflow which is able to estimate
    a physical property.
    """

    @property
    def protocols(self):
        """tuple of Protocol: The protocols in this workflow."""
        return {x.id: x for x in self._protocols}

    @property
    def final_value_source(self):
        """ProtocolPath: The path to the protocol output which corresponds to the
        estimated value of the property being estimated.
        """
        return self._final_value_source

    @property
    def outputs_to_store(self):
        """dict of str and StorageBackend: A collection of data classes to populate
        ready to be stored by a `StorageBackend`.
        """
        return self._outputs_to_store

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
        self._final_value_source = UNDEFINED
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
        schema.protocol_schemas = [copy.deepcopy(x.schema) for x in self._protocols]

        if self._final_value_source != UNDEFINED:
            schema.final_value_source = self._final_value_source.copy()

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
                            attribute_value.replicator_id = (
                                attribute_value.replicator_id.replace(
                                    replicator.placeholder_id, str(index)
                                )
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

    def replace_protocol(self, old_protocol, new_protocol, update_paths_only=False):
        """Replaces an existing protocol with a new one, while
        updating all input and local references to point to the
        new protocol.

        The main use of this method is when merging multiple protocols
        into one.

        Parameters
        ----------
        old_protocol : Protocol or ProtocolPath
            The protocol (or its id) to replace.
        new_protocol : Protocol or ProtocolPath
            The new protocol (or its id) to use.
        update_paths_only: bool
            Whether only update the `final_value_source`, and `outputs_to_store`
            attributes, or to also update all of the protocols in `protocols`.
        """

        new_id = (
            new_protocol if not isinstance(new_protocol, Protocol) else new_protocol.id
        )

        if new_id in [x.id for x in self._protocols]:

            raise ValueError(
                "A protocol with the same id already exists in this workflow."
            )

        if isinstance(old_protocol, Protocol):
            self._protocols.remove(old_protocol)
            old_protocol = old_protocol.id
        if isinstance(new_protocol, Protocol):
            self._protocols.append(new_protocol)
            new_protocol = new_protocol.id

        if not update_paths_only:

            for protocol in self._protocols:
                protocol.replace_protocol(old_protocol, new_protocol)

        if self._final_value_source != UNDEFINED:
            self._final_value_source.replace_protocol(old_protocol, new_protocol)

        for output_label in self._outputs_to_store:

            output_to_store = self._outputs_to_store[output_label]

            for attribute_name in output_to_store.get_attributes(StorageAttribute):

                attribute_value = getattr(output_to_store, attribute_name)

                if not isinstance(attribute_value, ProtocolPath):
                    continue

                attribute_value.replace_protocol(old_protocol, new_protocol)

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
        from openff.toolkit.topology import Molecule, Topology

        # noinspection PyTypeChecker
        if parameter_gradient_keys == UNDEFINED or len(parameter_gradient_keys) == 0:
            return []

        with open(force_field_path) as file:
            force_field_source = ForceFieldSource.parse_json(file.read())

        if isinstance(force_field_source, TLeapForceFieldSource):
            if force_field_source.leap_source == "leaprc.gaff2":
                amber_type = "gaff2"
            elif force_field_source.leap_source == "leaprc.gaff":
                amber_type = "gaff"
            else:
                raise ValueError(
                    f"The {force_field_source.leap_source} source is currently "
                    f"unsupported. Only the 'leaprc.gaff2' and 'leaprc.gaff' "
                    f" sources are supported."
                )

            force_field = GAFFForceField(substance, amber_type)
            topology = force_field.topology.topology
            frcmod_parameters = force_field.frcmod_parameters

            reduced_parameter_keys = []

            for parameter_key in parameter_gradient_keys:
                contains_parameter = False

                if parameter_key.tag == "Bond":
                    bond_type = parameter_key.smirks.replace("-", " ").split()

                    for bond in frcmod_parameters["BOND"]:
                        bond = bond.replace("-", " ").split()
                        atom_type1 = bond.split("-")[0]
                        atom_type2 = bond.split("-")[1]
                        if [atom_type1, atom_type2] == bond_type or [
                            atom_type2,
                            atom_type1,
                        ] == bond_type:
                            contains_parameter = True
                            break

                elif parameter_key.tag == "Angle":
                    angle_type = parameter_key.smirks.replace("-", " ").split()

                    for angle in frcmod_parameters["ANGLE"]:
                        angle = angle.replace("-", " ").split()
                        atom_type1 = angle.split("-")[0]
                        atom_type2 = angle.split("-")[1]
                        atom_type3 = angle.split("-")[2]
                        if [atom_type1, atom_type2, atom_type3] == angle_type or [
                            atom_type3,
                            atom_type2,
                            atom_type1,
                        ] == angle_type:
                            contains_parameter = True
                            break

                elif parameter_key.tag == "Dihedral":
                    raise NotImplementedError()

                elif parameter_key.tag == "Improper":
                    raise NotImplementedError()

                elif parameter_key.tag == "vdW":
                    lj_atom_type = parameter_key.smirks

                    for atom_type in frcmod_parameters["NONBON"]:
                        if lj_atom_type == atom_type:
                            contains_parameter = True
                            break

                elif parameter_key.tag == "GBSA":
                    from simtk.openmm.app import element as E
                    from simtk.openmm.app.internal.customgbforces import (
                        _get_bonded_atom_list,
                    )

                    # Check if H is supported by the Implicit solvent model
                    igb_H = {1: ["H-C", "H-N", "H-O", "H-S"], 2: ["H-N"], 5: ["H-N"]}
                    if parameter_key.smirks in igb_H[force_field_source.igb]:
                        continue

                    mask_element = E.get_by_symbol(parameter_key.smirks[0])
                    connect_element = None
                    if "-" in parameter_key.smirks:
                        connect_element = E.get_by_symbol(
                            parameter_key.smirks.split("-")[-1]
                        )

                    all_bonds = _get_bonded_atom_list(topology)
                    for atom in topology.atoms():
                        current_atom = None
                        element = atom.element
                        if element is mask_element and connect_element is None:
                            current_atom = atom
                        elif element is mask_element and connect_element:
                            bondeds = all_bonds[atom]
                            if bondeds[0].element is connect_element:
                                current_atom = atom
                        if current_atom:
                            contains_parameter = True
                            break

                elif parameter_key.tag == "Electrostatic":
                    raise NotImplementedError()

                else:
                    raise KeyError(
                        f"Parameter tag {parameter_key.tag} is not supported in GAFF."
                    )

                if not contains_parameter:
                    continue

                reduced_parameter_keys.append(parameter_key)

            return reduced_parameter_keys

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
        target_uncertainty: openff.evaluator.unit.Quantity, optional
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
            - target_uncertainty: openff.evaluator.unit.Quantity - The target uncertainty with which
                                                                   properties should be estimated.
            - per_component_uncertainty: openff.evaluator.unit.Quantity - The target uncertainty divided
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
        graph.add_workflows(self)
        return graph

    @classmethod
    def from_schema(cls, schema, metadata, unique_id=None):
        """Creates a workflow from its schema blueprint, and the associated metadata.

        Parameters
        ----------
        schema: WorkflowSchema
            The schema blueprint for this workflow.
        metadata: dict of str and Any
            The metadata to make available to the workflow.
        unique_id: str, optional
            A unique identifier to assign to this workflow. This id will be appended
            to the ids of the protocols of this workflow. If none is provided one will
            be chosen at random.

        Returns
        -------
        cls
            The created workflow.
        """
        workflow = cls(metadata, unique_id)
        workflow.schema = schema

        return workflow

    def execute(
        self, root_directory="", calculation_backend=None, compute_resources=None
    ):
        """Executes the workflow.

        Parameters
        ----------
        root_directory: str
            The directory to execute the graph in.
        calculation_backend: CalculationBackend, optional.
            The backend to execute the graph on. This parameter
            is mutually exclusive with `compute_resources`.
        compute_resources: CalculationBackend, optional.
            The compute resources to run using. If None and no
            `calculation_backend` is specified, the workflow will
            be executed on a single CPU thread. This parameter
            is mutually exclusive with `calculation_backend`.

        Returns
        -------
        WorkflowResult or Future of WorkflowResult:
          The result of executing this workflow. If executed on a
          `calculation_backend`, the result will be wrapped in a
          `Future` object.
        """
        if calculation_backend is None and compute_resources is None:
            compute_resources = ComputeResources(number_of_threads=1)

        workflow_graph = self.to_graph()
        return workflow_graph.execute(
            root_directory, calculation_backend, compute_resources
        )[0]


class WorkflowResult(AttributeClass):
    """The result of executing a `Workflow` as part of a
    `WorkflowGraph`.
    """

    workflow_id = Attribute(
        docstring="The id of the workflow associated with this result.",
        type_hint=str,
    )

    value = Attribute(
        docstring="The estimated value of the property and the uncertainty "
        "in that value.",
        type_hint=unit.Measurement,
        optional=True,
    )
    gradients = Attribute(
        docstring="The gradients of the estimated value with respect to the "
        "specified force field parameters.",
        type_hint=list,
        default_value=[],
    )

    exceptions = Attribute(
        docstring="Any exceptions raised by the layer while estimating the "
        "property.",
        type_hint=list,
        default_value=[],
    )

    data_to_store = Attribute(
        docstring="Paths to the data objects to store.",
        type_hint=list,
        default_value=[],
    )

    def validate(self, attribute_type=None):
        super(WorkflowResult, self).validate(attribute_type)

        assert all(isinstance(x, ParameterGradient) for x in self.gradients)

        assert all(isinstance(x, tuple) for x in self.data_to_store)
        assert all(len(x) == 2 for x in self.data_to_store)
        assert all(all(isinstance(y, str) for y in x) for x in self.data_to_store)

        assert all(isinstance(x, EvaluatorException) for x in self.exceptions)


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

    def __init__(self):

        super(WorkflowGraph, self).__init__()

        self._workflows_to_execute = {}
        self._protocol_graph = ProtocolGraph()

    def add_workflows(self, *workflows):
        """Insert a set of workflows into the workflow graph.

        Parameters
        ----------
        workflow: Workflow
            The workflow to insert.
        """

        workflow_uuids = [x.uuid for x in workflows]

        if len(set(workflow_uuids)) != len(workflow_uuids):
            raise ValueError("A number of workflows have the same uuid.")

        existing_uuids = [x for x in workflow_uuids if x in self._workflows_to_execute]

        if len(existing_uuids) > 0:

            raise ValueError(
                f"Workflows with the uuids {existing_uuids} are already in the graph."
            )

        original_protocols = []

        for workflow in workflows:

            original_protocols.extend(workflow.protocols.values())
            self._workflows_to_execute[workflow.uuid] = workflow

        # Add the workflow protocols to the graph.
        merged_protocol_ids = self._protocol_graph.add_protocols(
            *original_protocols, allow_external_dependencies=False
        )

        # Update the workflow to use the possibly merged protocols
        for original_id, new_id in merged_protocol_ids.items():

            original_protocol = original_id
            new_protocol = new_id

            for workflow in workflows:

                if (
                    retrieve_uuid(
                        original_protocol
                        if isinstance(original_protocol, str)
                        else original_protocol.id
                    )
                    != workflow.uuid
                ):
                    continue

                if original_protocol in workflow.protocols:
                    # Only retrieve the actual protocol if it isn't nested in
                    # a group.
                    original_protocol = workflow.protocols[original_id]
                    new_protocol = self._protocol_graph.protocols[new_id]

                workflow.replace_protocol(original_protocol, new_protocol, True)

    def execute(
        self, root_directory="", calculation_backend=None, compute_resources=None
    ):
        """Executes the workflow graph.

        Parameters
        ----------
        root_directory: str
            The directory to execute the graph in.
        calculation_backend: CalculationBackend, optional.
            The backend to execute the graph on. This parameter
            is mutually exclusive with `compute_resources`.
        compute_resources: CalculationBackend, optional.
            The compute resources to run using. If None and no
            `calculation_backend` is specified, the workflow will
            be executed on a single CPU thread. This parameter
            is mutually exclusive with `calculation_backend`.

        Returns
        -------
        list of WorkflowResult or list of Future of WorkflowResult:
            The results of executing the graph. If a `calculation_backend`
            is specified, these results will be wrapped in a `Future`.
        """
        if calculation_backend is None and compute_resources is None:
            compute_resources = ComputeResources(number_of_threads=1)

        protocol_outputs = self._protocol_graph.execute(
            root_directory, calculation_backend, compute_resources
        )

        value_futures = []

        for workflow_id in self._workflows_to_execute:

            workflow = self._workflows_to_execute[workflow_id]
            data_futures = []

            # Make sure we keep track of all of the futures which we
            # will use to populate things such as a final property value
            # or gradient keys.
            if workflow.final_value_source != UNDEFINED:

                protocol_id = workflow.final_value_source.start_protocol
                data_futures.append(protocol_outputs[protocol_id])

            if workflow.outputs_to_store != UNDEFINED:

                for output_label, output_to_store in workflow.outputs_to_store.items():

                    for attribute_name in output_to_store.get_attributes(
                        StorageAttribute
                    ):

                        attribute_value = getattr(output_to_store, attribute_name)

                        if not isinstance(attribute_value, ProtocolPath):
                            continue

                        data_futures.append(
                            protocol_outputs[attribute_value.start_protocol]
                        )

            if len(data_futures) == 0:
                data_futures = [*protocol_outputs.values()]

            if calculation_backend is None:

                value_futures.append(
                    WorkflowGraph._gather_results(
                        root_directory,
                        workflow.uuid,
                        workflow.final_value_source,
                        workflow.outputs_to_store,
                        *data_futures,
                    )
                )

            else:

                value_futures.append(
                    calculation_backend.submit_task(
                        WorkflowGraph._gather_results,
                        root_directory,
                        workflow.uuid,
                        workflow.final_value_source,
                        workflow.outputs_to_store,
                        *data_futures,
                    )
                )

        return value_futures

    @staticmethod
    def _gather_results(
        directory,
        workflow_id,
        value_reference,
        outputs_to_store,
        *protocol_result_paths,
        **_,
    ):
        """Gather the data associated with the workflows in this graph.

        Parameters
        ----------
        directory: str
            The directory to store any working files in.
        workflow_id: str
            The id of the workflow associated with this result.
        value_reference: ProtocolPath, optional
            A reference to which property in the output dictionary is the actual value.
        outputs_to_store: dict of str and WorkflowOutputToStore
            A list of references to data which should be stored on the storage backend.
        protocol_results: dict of str and str
            The result dictionary of the protocol which calculated the value of the property.

        Returns
        -------
        CalculationLayerResult, optional
            The result of attempting to estimate this property from a workflow graph. `None`
            will be returned if the target uncertainty is set but not met.
        """

        return_object = WorkflowResult()
        return_object.workflow_id = workflow_id

        try:

            results_by_id = {}

            for protocol_id, protocol_result_path in protocol_result_paths:

                with open(protocol_result_path, "r") as file:
                    protocol_results = json.load(file, cls=TypedJSONDecoder)

                # Make sure none of the protocols failed and we actually have a value
                # and uncertainty.
                if isinstance(protocol_results, EvaluatorException):

                    return_object.exceptions.append(protocol_results)
                    return return_object

                # Store the protocol results in a dictionary, with keys of the
                # path to the original protocol output.
                for protocol_path, output_value in protocol_results.items():

                    protocol_path = ProtocolPath.from_string(protocol_path)

                    if (
                        protocol_path.start_protocol is None
                        or protocol_path.start_protocol != protocol_id
                    ):
                        protocol_path.prepend_protocol_id(protocol_id)

                    results_by_id[protocol_path] = output_value

            if value_reference is not None:

                return_object.value = results_by_id[value_reference].value.plus_minus(
                    results_by_id[value_reference].error
                )
                return_object.gradients = results_by_id[value_reference].gradients

            return_object.data_to_store = []

            for output_to_store in outputs_to_store.values():

                unique_id = str(uuid.uuid4()).replace("-", "")

                data_object_path = path.join(directory, f"data_{unique_id}.json")
                data_directory = path.join(directory, f"data_{unique_id}")

                WorkflowGraph._store_output_data(
                    data_object_path,
                    data_directory,
                    output_to_store,
                    results_by_id,
                )

                return_object.data_to_store.append((data_object_path, data_directory))

        except Exception as e:
            return_object.exceptions.append(EvaluatorException.from_exception(e))

        return return_object

    @staticmethod
    def _store_output_data(
        data_object_path,
        data_directory,
        output_to_store,
        results_by_id,
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

            if isinstance(attribute_value, ProtocolPath):

                # Strip any nested attribute accessors before retrieving the result
                property_name = attribute_value.property_name.split(".")[0].split("[")[
                    0
                ]

                result_path = ProtocolPath(property_name, *attribute_value.protocol_ids)
                result = results_by_id[result_path]

                if result_path != attribute_value:

                    result = get_nested_attribute(
                        {property_name: result}, attribute_value.property_name
                    )

                attribute_value = result

                # Do not store gradient information for observables as this information
                # is very workflow / context specific.
                if isinstance(
                    attribute_value, (Observable, ObservableArray, ObservableFrame)
                ):
                    attribute_value.clear_gradients()

            if issubclass(attribute.type_hint, FilePath):
                file_copy(attribute_value, data_directory)
                attribute_value = path.basename(attribute_value)

            setattr(output_to_store, attribute_name, attribute_value)

        with open(data_object_path, "w") as file:
            json.dump(output_to_store, file, cls=TypedJSONEncoder)
