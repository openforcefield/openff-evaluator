"""
A collection of specialized workflow building blocks, which when chained together,
form a larger property estimation workflow.
"""

import copy
import logging
import pickle
import sys
from os import path

import numpy as np
import pymbar
from simtk import openmm, unit
from simtk.openmm import app

from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState, Ensemble
from propertyestimator.utils import packmol, graph, utils, statistics, timeseries, create_molecule_from_smiles
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import setup_platform_with_resources
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.serialization import deserialize_quantity, deserialize_force_field
from propertyestimator.utils.statistics import StatisticsArray, bootstrap
from propertyestimator.utils.utils import get_nested_attribute, set_nested_attribute
from propertyestimator.workflow.decorators import protocol_input, protocol_output, MergeBehaviour
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.schemas import ProtocolSchema
from propertyestimator.workflow.utils import ProtocolPath


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

    @protocol_input(value_type=bool)
    def allow_merging(self):
        """bool: If true, this protocol is allowed to merge with other identical protocols."""
        pass

    def __init__(self, protocol_id):

        # A unique identifier for this node.
        self._id = protocol_id

        # Defines whether a protocol is allowed to try and merge with other identical ones.
        self._allow_merging = True

        # Find the required inputs and outputs.
        self.provided_outputs = []
        self.required_inputs = []

        output_attributes = utils.find_types_with_decorator(type(self), 'ProtocolOutputObject')
        input_attributes = utils.find_types_with_decorator(type(self), 'ProtocolInputObject')

        for output_attribute in output_attributes:
            self.provided_outputs.append(ProtocolPath(output_attribute))

        for input_attribute in input_attributes:
            self.required_inputs.append(ProtocolPath(input_attribute))

        # The directory in which to execute the protocol.
        self.directory = None

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

            schema.inputs[input_path.full_path] = self.get_value(input_path)

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

            if isinstance(value, dict) and 'unit' in value and 'unitless_value' in value:
                value = deserialize_quantity(value)

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

    def can_merge(self, other):
        """Determines whether this protocol can be merged with another.

        Parameters
        ----------
        other : :obj:`BaseProtocol`
            The protocol to compare against.

        Returns
        ----------
        bool
            True if the two protocols are safe to merge.
        """
        if not self.allow_merging:
            return False

        if not isinstance(self, type(other)):
            return False

        for input_path in self.required_inputs:

            if input_path.start_protocol is not None and input_path.start_protocol != self.id:
                continue

            # Do not consider paths that point to child (e.g grouped) protocols.
            # These should be handled by the container classes themselves.
            if not (input_path.start_protocol is None or (
                    input_path.start_protocol == input_path.last_protocol and
                    input_path.start_protocol == self.id)):

                continue

            # If no merge behaviour flag is present (for example in the case of
            # ConditionalGroup conditions), simply assume this is handled explicitly
            # elsewhere.
            if not hasattr(type(self), input_path.property_name):
                continue

            if not hasattr(getattr(type(self), input_path.property_name), 'merge_behavior'):
                continue

            merge_behavior = getattr(type(self), input_path.property_name).merge_behavior

            if merge_behavior != MergeBehaviour.ExactlyEqual:
                continue

            if input_path not in other.required_inputs:
                return False

            self_value = self.get_value(input_path)
            other_value = other.get_value(input_path)

            if self_value != other_value:
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

        for input_path in self.required_inputs:

            # Do not consider paths that point to child (e.g grouped) protocols.
            # These should be handled by the container classes themselves.
            if not (input_path.start_protocol is None or (
                    input_path.start_protocol == input_path.last_protocol and
                    input_path.start_protocol == self.id)):

                continue

            # If no merge behaviour flag is present (for example in the case of
            # ConditionalGroup conditions), simply assume this is handled explicitly
            # elsewhere.
            if not hasattr(type(self), input_path.property_name):
                continue

            if not hasattr(getattr(type(self), input_path.property_name), 'merge_behavior'):
                continue

            merge_behavior = getattr(type(self), input_path.property_name).merge_behavior

            if merge_behavior == MergeBehaviour.ExactlyEqual:
                continue

            value = None

            if merge_behavior == MergeBehaviour.SmallestValue:
                value = min(self.get_value(input_path), other.get_value(input_path))
            elif merge_behavior == MergeBehaviour.GreatestValue:
                value = max(self.get_value(input_path), other.get_value(input_path))

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

        if not isinstance(input_value, list) and not isinstance(input_value, dict):
            return {}

        property_name, protocols_ids = ProtocolPath.to_components(input_path.full_path)

        return_paths = {}

        if isinstance(input_value, list):

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

        if reference_path.start_protocol is not None and reference_path.start_protocol != self.id:
            raise ValueError('The reference path {} does not point to this protocol'.format(reference_path))

        if (reference_path.property_name.count(ProtocolPath.property_separator) >= 1 or
            reference_path.property_name.find('[') > 0):

            return None
            # raise ValueError('The expected type cannot be found for '
            #                  'nested property names: {}'.format(reference_path.property_name))

        return getattr(type(self), reference_path.property_name).value_type

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

    def apply_replicator(self, replicator, template_values):
        """Applies a `ProtocolReplicator` to this protocol.

        Parameters
        ----------
        replicator: :obj:`ProtocolReplicator`
            The replicator to apply.
        template_values
            The values to pass to each of the replicated protocols.
        """
        raise ValueError('The {} protocol does not contain any protocols to replicate.'.format(self.id))


@register_calculation_protocol()
class BuildCoordinatesPackmol(BaseProtocol):
    """Creates a set of 3D coordinates with a specified composition.

    Notes
    -----
    The coordinates are created using packmol.
    """

    @protocol_input(int)
    def max_molecules(self):
        """The maximum number of molecules to be added to the system."""
        pass

    @protocol_input(unit.Quantity)
    def mass_density(self):
        """The target density of the created system."""
        pass

    @protocol_input(Substance)
    def substance(self):
        """The composition of the system to build."""
        pass

    @protocol_output(str)
    def coordinate_file_path(self):
        """The file path to the created PDB coordinate file."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        # inputs
        self._substance = None

        # outputs
        self._coordinate_file_path = None
        self._positions = None

        self._max_molecules = 1000
        self._mass_density = 0.95 * unit.grams / unit.milliliters

    def execute(self, directory, available_resources):

        logging.info('Generating coordinates: ' + self.id)

        if self._substance is None:

            return PropertyEstimatorException(directory=directory,
                                              message='The substance input is non-optional')

        molecules = []

        for component in self._substance.components:

            molecule = create_molecule_from_smiles(component.smiles)

            if molecule is None:

                return PropertyEstimatorException(directory=directory,
                                                  message='{} could not be converted to a Molecule'.format(component))

            molecules.append(molecule)

        # Determine how many molecules of each type will be present in the system.
        mole_fractions = np.array([component.mole_fraction for component in self._substance.components])

        n_copies = np.random.multinomial(self._max_molecules - self._substance.number_of_impurities,
                                         pvals=mole_fractions)

        # Each impurity must have exactly one molecule
        for (index, component) in enumerate(self._substance.components):

            if component.impurity:
                n_copies[index] = 1

        # Create packed box
        topology, positions = packmol.pack_box(molecules, n_copies, mass_density=self._mass_density)

        if topology is None or positions is None:

            return PropertyEstimatorException(directory=directory,
                                              message='Packmol failed to complete.')

        self._coordinate_file_path = path.join(directory, 'output.pdb')

        with open(self._coordinate_file_path, 'w+') as minimised_file:
            app.PDBFile.writeFile(topology, positions, minimised_file)

        logging.info('Coordinates generated: ' + self._substance.identifier)

        return self._get_output_dictionary()


@register_calculation_protocol()
class BuildSmirnoffSystem(BaseProtocol):
    """Parametrise a set of molecules with a given smirnoff force field.
    """

    @protocol_input(str)
    def force_field_path(self, value):
        """The file path to the force field parameters to assign to the system."""
        pass

    @protocol_input(str)
    def coordinate_file_path(self, value):
        """The file path to the coordinate file which defines the system to which the
        force field parameters will be assigned."""
        pass

    @protocol_input(Substance)
    def substance(self):
        """The composition of the system."""
        pass

    @protocol_input(unit.Quantity)
    def nonbonded_cutoff(self):
        """The cutoff after which non-bonded interactions are truncated."""
        pass

    @protocol_output(str)
    def system_path(self):
        """The assigned system."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        # inputs
        self._force_field_path = None
        self._coordinate_file_path = None
        self._substance = None

        self._nonbonded_cutoff = 1.0 * unit.nanometer

        # outputs
        self._system_path = None

    def execute(self, directory, available_resources):

        logging.info('Generating topology: ' + self.id)

        pdb_file = app.PDBFile(self._coordinate_file_path)

        force_field = None

        try:

            with open(self._force_field_path, 'rb') as file:
                force_field = deserialize_force_field(pickle.load(file))

        except pickle.UnpicklingError:

            try:

                from openforcefield.typing.engines.smirnoff import ForceField
                force_field = ForceField(self._force_field_path)

            except Exception as e:

                return PropertyEstimatorException(directory=directory,
                                                  message='{} could not load the ForceField: {}'.format(self.id, e))

        molecules = []

        for component in self._substance.components:

            molecule = create_molecule_from_smiles(component.smiles)

            if molecule is None:
                return PropertyEstimatorException(directory=directory,
                                                  message='{} could not be converted to a Molecule'.format(component))

            molecules.append(molecule)

        from openforcefield.typing.engines import smirnoff

        system = force_field.createSystem(pdb_file.topology,
                                          molecules,
                                          nonbondedMethod=smirnoff.PME,
                                          nonbondedCutoff=self._nonbonded_cutoff,
                                          chargeMethod='OECharges_AM1BCCSym')

        if system is None:

            return PropertyEstimatorException(directory=directory,
                                              message='Failed to create a system from the'
                                                       'provided topology and molecules')

        from simtk.openmm import XmlSerializer
        system_xml = XmlSerializer.serialize(system)

        self._system_path = path.join(directory, 'system.xml')

        with open(self._system_path, 'wb') as file:
            file.write(system_xml.encode('utf-8'))

        logging.info('Topology generated: ' + self.id)

        return self._get_output_dictionary()


@register_calculation_protocol()
class RunEnergyMinimisation(BaseProtocol):
    """A protocol to minimise the potential energy of a system.

    .. todo:: Add arguments for max iterations + tolerance
    """

    @protocol_input(str)
    def input_coordinate_file(self, value):
        """The coordinates to minimise."""
        pass

    @protocol_input(str)
    def system_path(self, value):
        """The path to the XML system object which defines the forces present in the system."""
        pass

    @protocol_output(str)
    def output_coordinate_file(self):
        """The file path to the minimised coordinates."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        # inputs
        self._input_coordinate_file = None

        self._system_path = None
        self._system = None

        # outputs
        self._output_coordinate_file = None

    def execute(self, directory, available_resources):

        logging.info('Minimising energy: ' + self.id)

        platform = setup_platform_with_resources(available_resources)

        input_pdb_file = app.PDBFile(self._input_coordinate_file)

        from simtk.openmm import XmlSerializer

        with open(self._system_path, 'rb') as file:
            self._system = XmlSerializer.deserialize(file.read().decode())

        integrator = openmm.VerletIntegrator(0.002 * unit.picoseconds)
        simulation = app.Simulation(input_pdb_file.topology, self._system, integrator, platform)

        box_vectors = input_pdb_file.topology.getPeriodicBoxVectors()

        if box_vectors is None:
            box_vectors = simulation.system.getDefaultPeriodicBoxVectors()

        simulation.context.setPeriodicBoxVectors(*box_vectors)
        simulation.context.setPositions(input_pdb_file.positions)

        simulation.minimizeEnergy()

        positions = simulation.context.getState(getPositions=True).getPositions()

        self._output_coordinate_file = path.join(directory, 'minimised.pdb')

        with open(self._output_coordinate_file, 'w+') as minimised_file:
            app.PDBFile.writeFile(simulation.topology, positions, minimised_file)

        logging.info('Energy minimised: ' + self.id)

        return self._get_output_dictionary()


@register_calculation_protocol()
class RunOpenMMSimulation(BaseProtocol):
    """Performs a molecular dynamics simulation in a given ensemble using
    an OpenMM backend.
    """

    @protocol_input(int, merge_behavior=MergeBehaviour.GreatestValue)
    def steps(self):
        """The number of timesteps to evolve the system by."""
        pass

    @protocol_input(unit.Quantity, merge_behavior=MergeBehaviour.SmallestValue)
    def thermostat_friction(self):
        """The thermostat friction coefficient."""
        pass

    @protocol_input(unit.Quantity, merge_behavior=MergeBehaviour.SmallestValue)
    def timestep(self):
        """The timestep to evolve the system by at each step."""
        pass

    @protocol_input(int, merge_behavior=MergeBehaviour.SmallestValue)
    def output_frequency(self):
        """The frequency with which to write to the output statistics and trajectory files."""
        pass

    @protocol_input(Ensemble)
    def ensemble(self):
        """The thermodynamic ensemble to simulate in."""
        pass

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self):
        """The thermodynamic conditions to simulate under"""
        pass

    @protocol_input(str)
    def input_coordinate_file(self):
        """The file path to the starting coordinates."""
        pass

    @protocol_input(str)
    def system_path(self):
        """A path to the XML system object which defines the forces present in the system."""
        pass

    @protocol_output(str)
    def output_coordinate_file(self):
        """The file path to the coordinates of the final system configuration."""
        pass

    @protocol_output(str)
    def trajectory_file_path(self):
        """The file path to the trajectory sampled during the simulation."""
        pass

    @protocol_output(str)
    def statistics_file_path(self):
        """The file path to the statistics sampled during the simulation."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._steps = 1000

        self._thermostat_friction = 1.0 / unit.picoseconds
        self._timestep = 0.001 * unit.picoseconds

        self._output_frequency = 1000

        self._ensemble = Ensemble.NPT

        # keep a track of the simulation object in case we need to restart.
        self._simulation_object = None

        # inputs
        self._input_coordinate_file = None
        self._thermodynamic_state = None

        self._system = None
        self._system_path = None

        # outputs
        self._output_coordinate_file = None

        self._trajectory_file_path = None
        self._statistics_file_path = None

        self._temporary_statistics_path = None

    def execute(self, directory, available_resources):

        temperature = self._thermodynamic_state.temperature
        pressure = self._thermodynamic_state.pressure

        if temperature is None:

            return PropertyEstimatorException(directory=directory,
                                              message='A temperature must be set to perform '
                                                       'a simulation in any ensemble')

        if Ensemble(self._ensemble) == Ensemble.NPT and pressure is None:

            return PropertyEstimatorException(directory=directory,
                                              message='A pressure must be set to perform an NPT simulation')

        logging.info('Performing a simulation in the ' + str(self._ensemble) + ' ensemble: ' + self.id)

        if self._simulation_object is None:
            self._simulation_object = self._setup_new_simulation(directory, temperature, pressure, available_resources)

        try:
            self._simulation_object.step(self._steps)
        except Exception as e:

            return PropertyEstimatorException(directory=directory,
                                              message='Simulation failed: {}'.format(e))

        # Save the newly generated statistics data as a pandas csv file.
        pressure = None if self._ensemble == Ensemble.NVT else self._thermodynamic_state.pressure

        working_statistics = statistics.StatisticsArray.from_openmm_csv(self._temporary_statistics_path, pressure)
        working_statistics.save_as_pandas_csv(self._statistics_file_path)

        positions = self._simulation_object.context.getState(getPositions=True).getPositions()

        topology = app.PDBFile(self._input_coordinate_file).topology
        topology.setPeriodicBoxVectors(self._simulation_object.context.getState().getPeriodicBoxVectors())

        self._output_coordinate_file = path.join(directory, 'output.pdb')

        logging.info('Simulation performed in the ' + str(self._ensemble) + ' ensemble: ' + self.id)

        with open(self._output_coordinate_file, 'w+') as configuration_file:

            app.PDBFile.writeFile(topology,
                                  positions, configuration_file)

        return self._get_output_dictionary()

    def _setup_new_simulation(self, directory, temperature, pressure, available_resources):
        """Creates a new OpenMM simulation object.

        Parameters
        ----------
        directory: str
            The directory in which the object will produce output files.
        temperature: unit.Quantiy
            The temperature at which to run the simulation
        pressure: unit.Quantiy
            The pressure at which to run the simulation
        available_resources: ComputeResources
            The resources available to run on.
        """
        import openmmtools

        platform = setup_platform_with_resources(available_resources)

        input_pdb_file = app.PDBFile(self._input_coordinate_file)

        from simtk.openmm import XmlSerializer

        with open(self._system_path, 'rb') as file:
            self._system = XmlSerializer.deserialize(file.read().decode())

        openmm_state = openmmtools.states.ThermodynamicState(system=self._system,
                                                             temperature=temperature,
                                                             pressure=pressure)

        integrator = openmm.LangevinIntegrator(temperature,
                                               self._thermostat_friction,
                                               self._timestep)

        simulation = app.Simulation(input_pdb_file.topology,
                                    openmm_state.get_system(True),
                                    integrator,
                                    platform)

        box_vectors = input_pdb_file.topology.getPeriodicBoxVectors()

        if box_vectors is None:
            box_vectors = simulation.system.getDefaultPeriodicBoxVectors()

        simulation.context.setPeriodicBoxVectors(*box_vectors)
        simulation.context.setPositions(input_pdb_file.positions)
        simulation.context.setVelocitiesToTemperature(temperature)

        trajectory_path = path.join(directory, 'trajectory.dcd')
        statistics_path = path.join(directory, 'statistics.csv')

        self._temporary_statistics_path = path.join(directory, 'temp_statistics.csv')

        self._trajectory_file_path = trajectory_path
        self._statistics_file_path = statistics_path

        configuration_path = path.join(directory, 'input.pdb')

        with open(configuration_path, 'w+') as configuration_file:

            app.PDBFile.writeFile(input_pdb_file.topology,
                                  input_pdb_file.positions, configuration_file)

        simulation.reporters.append(app.DCDReporter(trajectory_path, self._output_frequency))

        simulation.reporters.append(app.StateDataReporter(self._temporary_statistics_path, self._output_frequency,
                                                          step=True, potentialEnergy=True, kineticEnergy=True,
                                                          totalEnergy=True, temperature=True, volume=True,
                                                          density=True))

        return simulation


@register_calculation_protocol()
class AveragePropertyProtocol(BaseProtocol):
    """An abstract base class for protocols which will calculate the
    average of a property and its uncertainty via bootstrapping.
    """

    @protocol_input(int, merge_behavior=MergeBehaviour.GreatestValue)
    def bootstrap_iterations(self):
        """The number of bootstrap iterations to perform."""
        pass

    @protocol_input(float, merge_behavior=MergeBehaviour.GreatestValue)
    def bootstrap_sample_size(self):
        """The relative sample size to use for bootstrapping."""
        pass

    @protocol_output(EstimatedQuantity)
    def value(self):
        """The averaged value."""
        pass

    @protocol_output(int)
    def equilibration_index(self):
        """The index in the data set after which the data is stationary."""
        pass

    @protocol_output(float)
    def statistical_inefficiency(self):
        """The statistical inefficiency in the data set."""
        pass

    @protocol_output(unit.Quantity)
    def uncorrelated_values(self):
        """The uncorrelated values which the average was calculated from."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._bootstrap_iterations = 250
        self._bootstrap_sample_size = 1.0

        self._value = None

        self._equilibration_index = None
        self._statistical_inefficiency = None

        self._uncorrelated_values = None

    def _bootstrap_function(self, **sample_kwargs):
        """The function to perform on the data set being sampled by
        bootstrapping.

        Parameters
        ----------
        sample_kwargs: dict of str and np.ndarray
            A key words dictionary of the bootstrap sample data, where the
            sample data is a numpy array of shape=(num_frames, num_dimensions)
            with dtype=float.

        Returns
        -------
        float
            The result of evaluating the data.
        """

        assert len(sample_kwargs) == 1
        sample_data = next(iter(sample_kwargs.values()))

        return sample_data.mean()

    def execute(self, directory, available_resources):
        return self._get_output_dictionary()


@register_calculation_protocol()
class AverageTrajectoryProperty(AveragePropertyProtocol):
    """An abstract base class for protocols which will calculate the
    average of a property from a simulation trajectory.
    """

    @protocol_input(str)
    def input_coordinate_file(self):
        """The file path to the starting coordinates of a trajectory."""
        pass

    @protocol_input(str)
    def trajectory_path(self):
        """The file path to the trajectory to average over."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._input_coordinate_file = None
        self._trajectory_path = None

        self.trajectory = None

    def execute(self, directory, available_resources):

        import mdtraj

        if self._trajectory_path is None:

            return PropertyEstimatorException(directory=directory,
                                              message='The AverageTrajectoryProperty protocol '
                                                       'requires a previously calculated trajectory')

        self.trajectory = mdtraj.load_dcd(filename=self._trajectory_path, top=self._input_coordinate_file)

        return self._get_output_dictionary()


@register_calculation_protocol()
class ExtractAverageStatistic(AveragePropertyProtocol):
    """Extracts the average value from a statistics file which was generated
    during a simulation.
    """

    @protocol_input(str)
    def statistics_path(self):
        """The file path to the trajectory to average over."""
        pass

    @protocol_input(statistics.ObservableType)
    def statistics_type(self):
        """The file path to the trajectory to average over."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._statistics_path = None
        self._statistics_type = statistics.ObservableType.PotentialEnergy

        self._statistics = None

    def execute(self, directory, available_resources):

        logging.info('Extracting {}: {}'.format(self._statistics_type, self.id))

        if self._statistics_path is None:

            return PropertyEstimatorException(directory=directory,
                                              message='The ExtractAverageStatistic protocol '
                                                       'requires a previously calculated statistics file')

        self._statistics = statistics.StatisticsArray.from_pandas_csv(self.statistics_path)

        values = self._statistics.get_observable(self._statistics_type)

        if values is None or len(values) == 0:

            return PropertyEstimatorException(directory=directory,
                                              message='The {} statistics file contains no '
                                                      'data.'.format(self._statistics_path))

        statistics_unit = values[0].unit
        values.value_in_unit(statistics_unit)

        values = np.array(values)

        values, self._equilibration_index, self._statistical_inefficiency = \
            timeseries.decorrelate_time_series(values)

        final_value, final_uncertainty = bootstrap(self._bootstrap_function,
                                                   self._bootstrap_iterations,
                                                   self._bootstrap_sample_size,
                                                   values=values)

        self._uncorrelated_values = values * statistics_unit

        self._value = EstimatedQuantity(unit.Quantity(final_value, statistics_unit),
                                        unit.Quantity(final_uncertainty, statistics_unit), self.id)

        logging.info('Extracted {}: {}'.format(self._statistics_type, self.id))

        return self._get_output_dictionary()


@register_calculation_protocol()
class ExtractUncorrelatedData(BaseProtocol):
    """An abstract base class for protocols which will subsample
    a data set, yielding only equilibrated, uncorrelated data.
    """

    @protocol_input(int)
    def equilibration_index(self):
        """The index in the data set after which the data is stationary."""
        pass

    @protocol_input(float)
    def statistical_inefficiency(self):
        """The statistical inefficiency in the data set."""
        pass

    @protocol_output(int)
    def number_of_uncorrelated_samples(self):
        """The number of uncorrelated samples."""
        pass

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._equilibration_index = None
        self._statistical_inefficiency = None

        self._number_of_uncorrelated_samples = None

    def execute(self, directory, available_resources):
        raise NotImplementedError


@register_calculation_protocol()
class ExtractUncorrelatedTrajectoryData(ExtractUncorrelatedData):
    """A protocol which will subsample frames from a trajectory, yielding only uncorrelated 
    frames as determined from a provided statistical inefficiency and equilibration time.
    """

    @protocol_input(str)
    def input_coordinate_file(self):
        """The file path to the starting coordinates of a trajectory."""
        pass

    @protocol_input(str)
    def input_trajectory_path(self):
        """The file path to the trajectory to subsample."""
        pass

    @protocol_output(str)
    def output_trajectory_path(self):
        """The file path to the subsampled trajectory."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._input_coordinate_file = None
        self._input_trajectory_path = None

        self._output_trajectory_path = None

    def execute(self, directory, available_resources):

        import mdtraj

        logging.info('Subsampling trajectory: {}'.format(self.id))

        if self._input_trajectory_path is None:

            return PropertyEstimatorException(directory=directory,
                                              message='The ExtractUncorrelatedTrajectoryData protocol '
                                                       'requires a previously calculated trajectory')

        trajectory = mdtraj.load_dcd(filename=self._input_trajectory_path, top=self._input_coordinate_file)
        trajectory = trajectory[self._equilibration_index:]

        uncorrelated_indices = timeseries.get_uncorrelated_indices(trajectory.n_frames, self._statistical_inefficiency)
        uncorrelated_trajectory = trajectory[uncorrelated_indices]

        self._output_trajectory_path = path.join(directory, 'uncorrelated_trajectory.dcd')
        uncorrelated_trajectory.save_dcd(self._output_trajectory_path)

        self._number_of_uncorrelated_samples = len(trajectory)

        logging.info('Trajectory subsampled: {}'.format(self.id))

        return self._get_output_dictionary()


@register_calculation_protocol()
class ExtractUncorrelatedStatisticsData(ExtractUncorrelatedData):
    """A protocol which will subsample entries from a statistics array, yielding only uncorrelated
    entries as determined from a provided statistical inefficiency and equilibration time.
    """

    @protocol_input(str)
    def input_statistics_path(self):
        """The file path to the statistics to subsample."""
        pass

    @protocol_output(str)
    def output_statistics_path(self):
        """The file path to the subsampled statistics."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._input_statistics_path = None
        self._output_statistics_path = None

    def execute(self, directory, available_resources):

        logging.info('Subsampling statistics: {}'.format(self.id))

        if self._input_statistics_path is None:

            return PropertyEstimatorException(directory=directory,
                                              message='The ExtractUncorrelatedStatisticsData protocol '
                                                       'requires a previously calculated statisitics file')

        statistics = StatisticsArray.from_pandas_csv(self._input_statistics_path)

        uncorrelated_indices = timeseries.get_uncorrelated_indices(len(statistics) - self._equilibration_index,
                                                                   self._statistical_inefficiency)

        uncorrelated_indices = [index + self._equilibration_index for index in uncorrelated_indices]
        uncorrelated_statistics = statistics.from_statistics_array(statistics, uncorrelated_indices)

        self._output_statistics_path = path.join(directory, 'uncorrelated_statistics.csv')
        uncorrelated_statistics.save_as_pandas_csv(self._output_statistics_path)

        logging.info('Statistics subsampled: {}'.format(self.id))

        self._number_of_uncorrelated_samples = len(uncorrelated_statistics)

        return self._get_output_dictionary()


@register_calculation_protocol()
class AddQuantities(BaseProtocol):
    """A protocol to add together a list of values.

    Notes
    -----
    The `values` input must either be a list of unit.Quantity, a ProtocolPath to a list
    of unit.Quantity, or a list of ProtocolPath which each point to a unit.Quantity.
    """

    @protocol_input(list)
    def values(self):
        """The values to add together."""
        pass

    @protocol_output(EstimatedQuantity)
    def result(self):
        """The sum of the values."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new AddQuantities object."""
        super().__init__(protocol_id)

        self._values = None
        self._result = None

    def execute(self, directory, available_resources):

        self._result = None

        for value in self._values:

            if self._result is None:

                self._result = value
                continue

            self._result += value

        return self._get_output_dictionary()


@register_calculation_protocol()
class SubtractQuantities(BaseProtocol):
    """A protocol to subtract one value from another such that:

    `result = value_b - value_a`
    """

    @protocol_input(EstimatedQuantity)
    def value_a(self):
        """`value_a` in the formula `result = value_b - value_a`"""
        pass

    @protocol_input(EstimatedQuantity)
    def value_b(self):
        """`value_b` in the formula  `result = value_b - value_a`"""
        pass

    @protocol_output(EstimatedQuantity)
    def result(self):
        """The sum of the values."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new AddQuantities object."""
        super().__init__(protocol_id)

        self._value_a = None
        self._value_b = None

        self._result = None

    def execute(self, directory, available_resources):

        self._result = self._value_b - self._value_a
        return self._get_output_dictionary()


@register_calculation_protocol()
class UnpackStoredSimulationData(BaseProtocol):
    """Loads a pickled `StoredSimulationData` object from disk,
    and makes its attributes easily accessible to other protocols.
    """

    @protocol_input(tuple)
    def simulation_data_path(self):
        """A tuple which contains both the path to the pickled simulation data object,
        and the force field which was used to generate the stored data."""
        pass

    @protocol_output(Substance)
    def substance(self):
        """The substance which was stored."""
        pass

    @protocol_output(ThermodynamicState)
    def thermodynamic_state(self):
        """The thermodynamic state which was stored."""
        pass

    @protocol_output(float)
    def statistical_inefficiency(self):
        """The statistical inefficiency of the stored data."""
        pass

    @protocol_output(str)
    def coordinate_file_path(self):
        """A path to the stored simulation trajectory."""
        pass

    @protocol_output(str)
    def trajectory_file_path(self):
        """A path to the stored simulation trajectory."""
        pass

    @protocol_output(str)
    def statistics_file_path(self):
        """A path to the stored simulation statistics array."""
        pass

    @protocol_output(str)
    def force_field_path(self):
        """A path to the force field parameters used to generate
        the stored data."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new UnpackStoredSimulationData object."""
        super().__init__(protocol_id)

        self._simulation_data_path = None

        self._substance = None
        self._thermodynamic_state = None

        self._statistical_inefficiency = None

        self._coordinate_file_path = None
        self._trajectory_file_path = None

        self._statistics_file_path = None

        self._force_field_path = None

    def execute(self, directory, available_resources):

        if len(self._simulation_data_path) != 2:

            return PropertyEstimatorException(directory=directory,
                                              message='The simulation data path should be a tuple'
                                                      'of a path to the pickled data object, and'
                                                      'a path to the force field used to generate it.')

        data_object = None

        pickled_object_path = self._simulation_data_path[0]
        force_field_path = self._simulation_data_path[1]

        if not path.isfile(force_field_path):

            return PropertyEstimatorException(directory=directory,
                                              message='The path to the force field'
                                                      'is invalid: {}'.format(force_field_path))

        try:

            with open(pickled_object_path, 'rb') as file:
                data_object = pickle.load(file)

        except (IOError, pickle.UnpicklingError) as e:

            return PropertyEstimatorException(directory=directory,
                                              message='Failed to load the data object: {}'.format(e))

        self._substance = data_object.substance
        self._thermodynamic_state = data_object.thermodynamic_state

        self._statistical_inefficiency = data_object.statistical_inefficiency

        self._coordinate_file_path = path.join(directory, 'coordinates.pdb')
        self._trajectory_file_path = path.join(directory, 'trajectory.dcd')

        data_object.trajectory_data[0].save_pdb(self._coordinate_file_path)
        data_object.trajectory_data.save_dcd(self._trajectory_file_path)

        self._statistics_file_path = path.join(directory, 'statistics.csv')
        data_object.statistics_data.save_as_pandas_csv(self._statistics_file_path)

        self._force_field_path = force_field_path

        return self._get_output_dictionary()


@register_calculation_protocol()
class ConcatenateTrajectories(BaseProtocol):
    """A protocol which concatenates multiple trajectories into
    a single one.
    """

    @protocol_input(list)
    def input_coordinate_paths(self):
        """A list of paths to the starting coordinates for each of the trajectories."""
        pass

    @protocol_input(list)
    def input_trajectory_paths(self):
        """A list of paths to the trajectories to concatenate."""
        pass

    @protocol_output(str)
    def output_coordinate_path(self):
        """The path the coordinate file which contains the topology of
        the concatenated trajectory."""
        pass

    @protocol_output(str)
    def output_trajectory_path(self):
        """The path to the concatenated trajectory."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new AddQuantities object."""
        super().__init__(protocol_id)

        self._input_coordinate_paths = None
        self._input_trajectory_paths = None

        self._output_coordinate_path = None
        self._output_trajectory_path = None

    def execute(self, directory, available_resources):

        import mdtraj

        if len(self._input_coordinate_paths) != len(self._input_trajectory_paths):

            return PropertyEstimatorException(directory=directory, message='There should be the same number of '
                                                                           'coordinate and trajectory paths.')

        if len(self._input_trajectory_paths) == 0:

            return PropertyEstimatorException(directory=directory, message='No trajectories were '
                                                                           'given to concatenate.')

        trajectories = []

        for coordinate_path, trajectory_path in zip(self._input_coordinate_paths,
                                                    self._input_trajectory_paths):

            self._output_coordinate_path = self._output_coordinate_path or coordinate_path
            trajectories.append(mdtraj.load_dcd(trajectory_path, coordinate_path))

        output_trajectory = trajectories[0] if len(trajectories) == 1 else mdtraj.join(trajectories, True, False)

        self._output_trajectory_path = path.join(directory, 'output_trajectory.dcd')
        output_trajectory.save_dcd(self._output_trajectory_path)

        return self._get_output_dictionary()


@register_calculation_protocol()
class CalculateReducedPotentialOpenMM(BaseProtocol):
    """Calculates the reduced potential for a given
    set of configurations.
    """

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self):
        pass

    @protocol_input(str)
    def system_path(self):
        pass

    @protocol_input(str)
    def coordinate_file_path(self):
        pass

    @protocol_input(str)
    def trajectory_file_path(self):
        pass

    @protocol_output(np.ndarray)
    def reduced_potentials(self):
        pass

    def __init__(self, protocol_id):
        """Constructs a new UnpackStoredSimulationData object."""
        super().__init__(protocol_id)

        self._thermodynamic_state = None

        self._system_path = None
        self._system = None

        self._coordinate_file_path = None
        self._trajectory_file_path = None

        self._reduced_potentials = None

    def execute(self, directory, available_resources):

        import openmmtools
        import mdtraj

        from simtk.openmm import XmlSerializer

        with open(self._system_path, 'rb') as file:
            self._system = XmlSerializer.deserialize(file.read().decode())

        trajectory = mdtraj.load_dcd(self._trajectory_file_path, self._coordinate_file_path)
        self._system.setDefaultPeriodicBoxVectors(*trajectory.openmm_boxes(0))

        openmm_state = openmmtools.states.ThermodynamicState(system=self._system,
                                                             temperature=self._thermodynamic_state.temperature,
                                                             pressure=self._thermodynamic_state.pressure)

        integrator = openmmtools.integrators.VelocityVerletIntegrator(0.01*unit.femtoseconds)

        # Setup the requested platform:
        platform = setup_platform_with_resources(available_resources)

        context_cache = openmmtools.cache.ContextCache(platform)
        openmm_context, openmm_context_integrator = context_cache.get_context(openmm_state,
                                                                              integrator)

        reduced_potentials = np.zeros(trajectory.n_frames)

        for frame_index in range(trajectory.n_frames):

            positions = trajectory.openmm_positions(frame_index)
            box_vectors = trajectory.openmm_boxes(frame_index)

            openmm_context.setPeriodicBoxVectors(*box_vectors)
            openmm_context.setPositions(positions)

            # set box vectors
            reduced_potentials[frame_index] = openmm_state.reduced_potential(openmm_context)

        self._reduced_potentials = reduced_potentials

        return self._get_output_dictionary()


@register_calculation_protocol()
class ReweightWithMBARProtocol(BaseProtocol):
    """Reweights a set of observables using MBAR to calculate
    the average value of the observables at a different state
    than they were originally measured.
    """

    @protocol_input(list)
    def reference_reduced_potentials(self):
        """A list of the reduced potentials of each reference state."""
        pass

    @protocol_input(list)
    def reference_observables(self):
        """A list of the observables to be reweighted from each reference state."""
        pass

    @protocol_input(list)
    def target_reduced_potentials(self):
        """The reduced potentials of the target state."""
        pass

    @protocol_input(bool)
    def bootstrap_uncertainties(self):
        """If true, bootstrapping will be used to estimated the total uncertainty"""
        pass

    @protocol_input(int)
    def bootstrap_iterations(self):
        """The number of bootstrap iterations to perform if bootstraped
        uncertainties have been requested"""
        pass

    @protocol_input(float)
    def bootstrap_sample_size(self):
        """The relative bootstrap sample size to use if bootstraped
        uncertainties have been requested"""
        pass

    @protocol_input(int)
    def required_effective_samples(self):
        """The minimum number of MBAR effective samples for the reweighted
        value to be trusted. If this minimum is not met then the uncertainty
        will be set to sys.float_info.max"""
        pass

    @protocol_output(EstimatedQuantity)
    def value(self):
        pass

    def __init__(self, protocol_id):
        """Constructs a new ReweightWithMBARProtocol object."""
        super().__init__(protocol_id)

        self._reference_reduced_potentials = None
        self._reference_observables = None

        self._target_reduced_potentials = None

        self._bootstrap_uncertainties = False
        self._bootstrap_iterations = 1
        self._bootstrap_sample_size = 1.0

        self._required_effective_samples = 50

        self._value = None

    def execute(self, directory, available_resources):

        if len(self._reference_observables) == 0:

            return PropertyEstimatorException(directory=directory,
                                              message='There were no observables to reweight.')

        if not isinstance(self._reference_observables[0], unit.Quantity):

            return PropertyEstimatorException(directory=directory,
                                              message='The reference_observables input should be'
                                                      'a list of unit.Quantity wrapped ndarray\'s.')

        observables = self._prepare_observables_array(self._reference_observables)
        observable_unit = self._reference_observables[0].unit

        if self._bootstrap_uncertainties:

            reference_potentials = np.transpose(np.array(self._reference_reduced_potentials))
            target_potentials = np.transpose(np.array(self._target_reduced_potentials))

            frame_counts = np.array([len(observable) for observable in self._reference_observables])

            # Construct an mbar object to get out the number of effective samples.
            mbar = pymbar.MBAR(self._reference_reduced_potentials,
                               frame_counts, verbose=False, relative_tolerance=1e-12)

            effective_samples = mbar.computeEffectiveSampleNumber().max()

            value, uncertainty = bootstrap(self._bootstrap_function,
                                           self._bootstrap_iterations,
                                           self._bootstrap_sample_size,
                                           frame_counts,
                                           reference_reduced_potentials=reference_potentials,
                                           target_reduced_potentials=target_potentials,
                                           observables=np.transpose(observables))

            if effective_samples < self._required_effective_samples:
                uncertainty = sys.float_info.max

            self._value = EstimatedQuantity(value * observable_unit,
                                            uncertainty * observable_unit,
                                            self.id)

        else:

            values, uncertainties, effective_samples = self._reweight_observables(self._reference_reduced_potentials,
                                                                                  self._target_reduced_potentials,
                                                                                  observables=observables)

            uncertainty = uncertainties['observables']

            if effective_samples < self._required_effective_samples:
                uncertainty = sys.float_info.max

            self._value = EstimatedQuantity(values['observables'] * observable_unit,
                                            uncertainty * observable_unit,
                                            self.id)

        return self._get_output_dictionary()

    @staticmethod
    def _prepare_observables_array(reference_observables):
        """Takes a list of reference observables, and concatenates them
        into a single Quantity wrapped numpy array.

        Parameters
        ----------
        reference_observables: List of unit.Quantity
            A list of observables for each reference state,
            which each observable is a Quantity wrapped numpy
            array.

        Returns
        -------
        np.ndarray
            A unitless numpy array of all of the observables.
        """
        frame_counts = np.array([len(observable) for observable in reference_observables])
        number_of_configurations = frame_counts.sum()

        observable_dimensions = 1 if len(reference_observables[0].shape) == 1 else reference_observables[0].shape[1]
        observable_unit = reference_observables[0].unit

        observables = np.zeros((observable_dimensions, number_of_configurations))

        # Build up an array which contains the observables from all
        # of the reference states.
        for index_k, observables_k in enumerate(reference_observables):

            start_index = np.array(frame_counts[0:index_k]).sum()

            for index in range(0, frame_counts[index_k]):

                value = observables_k[index].value_in_unit(observable_unit)

                if not isinstance(value, np.ndarray):
                    observables[0][start_index + index] = value
                    continue

                for dimension in range(observable_dimensions):
                    observables[dimension][start_index + index] = value[dimension]

        return observables

    def _bootstrap_function(self, reference_reduced_potentials, target_reduced_potentials, **reference_observables):
        """The function which will be called after each bootstrap
        iteration, if bootstrapping is being employed to estimated
        the reweighting uncertainty.

        Parameters
        ----------
        reference_reduced_potentials
        target_reduced_potentials
        reference_observables

        Returns
        -------
        float
            The bootstrapped value,
        """
        assert len(reference_observables) == 1

        transposed_observables = {}

        for key in reference_observables:
            transposed_observables[key] = np.transpose(reference_observables[key])

        values, _, _ = self._reweight_observables(np.transpose(reference_reduced_potentials),
                                                               np.transpose(target_reduced_potentials),
                                                               **transposed_observables)

        return next(iter(values.values()))

    def _reweight_observables(self, reference_reduced_potentials, target_reduced_potentials, **reference_observables):
        """Reweights a set of reference observables to
        the target state.

        Returns
        -------
        dict of str and float or list of float
            The reweighted values.
        dict of str and float or list of float
            The MBAR calculated uncertainties in the reweighted values.
        int
            The number of effective samples.
        """

        frame_counts = np.array([len(observable) for observable in self._reference_observables])

        # Construct the mbar object.
        mbar = pymbar.MBAR(reference_reduced_potentials,
                           frame_counts, verbose=False, relative_tolerance=1e-12)

        max_effective_samples = mbar.computeEffectiveSampleNumber().max()

        values = {}
        uncertainties = {}

        for observable_key in reference_observables:

            observable = reference_observables[observable_key]
            observable_dimensions = observable.shape[0]

            if observable_dimensions == 1:

                results = mbar.computeExpectations(observable,
                                                   target_reduced_potentials,
                                                   state_dependent=True)

                values[observable_key] = results[0][0]
                uncertainties[observable_key] = results[1][0]

            else:

                value = []
                uncertainty = []

                for dimension in range(observable_dimensions):

                    results = mbar.computeExpectations(observable[dimension],
                                                       target_reduced_potentials,
                                                       state_dependent=True)

                    value.append(results[0][0])
                    uncertainty.append(results[1][0])

                values[observable_key] = np.array(value)
                uncertainties[observable_key] = np.array(uncertainty)

        return values, uncertainties, max_effective_samples
