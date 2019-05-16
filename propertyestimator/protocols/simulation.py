"""
A collection of protocols for running molecular simulations.
"""
import contextlib
import logging
import os
import shutil
from os import path

import numpy as np
import yaml
from simtk import unit, openmm
from simtk.openmm import app

from propertyestimator.thermodynamics import ThermodynamicState, Ensemble
from propertyestimator.utils import statistics
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import setup_platform_with_resources
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow.decorators import protocol_input, protocol_output, MergeBehaviour
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


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

        with open(self._system_path, 'rb') as file:
            self._system = openmm.XmlSerializer.deserialize(file.read().decode())

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

        self._output_frequency = 500000

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
        pressure = None if self._ensemble == Ensemble.NVT else self._thermodynamic_state.pressure

        if temperature is None:

            return PropertyEstimatorException(directory=directory,
                                              message='A temperature must be set to perform '
                                                      'a simulation in any ensemble')

        if Ensemble(self._ensemble) == Ensemble.NPT and pressure is None:

            return PropertyEstimatorException(directory=directory,
                                              message='A pressure must be set to perform an NPT simulation')

        logging.info('Performing a simulation in the ' + str(self._ensemble) + ' ensemble: ' + self.id)

        if self._simulation_object is None:

            # Set up the simulation object if one does not already exist
            # (if this protocol is part of a conditional group for e.g.
            # the simulation object will most likely persist without the
            # need to recreate it at each iteration.)
            self._simulation_object = self._setup_simulation_object(directory,
                                                                    temperature,
                                                                    pressure,
                                                                    available_resources)

        try:
            self._simulation_object.step(self._steps)
        except Exception as e:

            return PropertyEstimatorException(directory=directory,
                                              message='Simulation failed: {}'.format(e))

        # Save the newly generated statistics data as a pandas csv file.
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

    def _setup_simulation_object(self, directory, temperature, pressure, available_resources):
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
        from simtk.openmm import XmlSerializer

        platform = setup_platform_with_resources(available_resources)

        input_pdb_file = app.PDBFile(self._input_coordinate_file)

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

        checkpoint_path = path.join(directory, 'checkpoint.chk')

        if path.isfile(checkpoint_path):

            # Load the simulation state from a checkpoint file.
            with open(checkpoint_path, 'rb') as f:
                simulation.context.loadCheckpoint(f.read())

        else:

            # Populate the simulation object from the starting input files.
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

        simulation.reporters.append(app.CheckpointReporter(checkpoint_path, self._output_frequency))

        return simulation


@register_calculation_protocol()
class BaseYankProtocol(BaseProtocol):
    """An abstract base class for protocols which will performs a set of alchemical
    free energy simulations using the YANK framework.

    Protocols which inherit from this base must implement the abstract `_get_yank_options`
    methods.
    """

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self):
        """The state at which to run the calculations."""
        pass

    @protocol_input(int, merge_behavior=MergeBehaviour.GreatestValue)
    def number_of_iterations(self):
        """The number of YANK iterations to perform."""
        pass

    @protocol_input(int, merge_behavior=MergeBehaviour.GreatestValue)
    def steps_per_iteration(self):
        """The number of steps per YANK iteration to perform."""
        pass

    @protocol_input(int, merge_behavior=MergeBehaviour.SmallestValue)
    def checkpoint_interval(self):
        """The number of iterations between saving YANK checkpoint files."""
        pass

    @protocol_input(unit.Quantity, merge_behavior=MergeBehaviour.SmallestValue)
    def timestep(self):
        """The length of the timestep to take."""
        pass

    @protocol_input(str)
    def force_field_path(self):
        """The path to the force field to use for the calculations"""
        pass

    @protocol_input(bool)
    def verbose(self):
        """Controls whether or not to run YANK at high verbosity."""
        pass

    @protocol_output(EstimatedQuantity)
    def estimated_free_energy(self):
        """The estimated free energy value and its uncertainty returned
        by YANK."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new BaseYankProtocol object."""

        super().__init__(protocol_id)

        self._thermodynamic_state = None
        self._timestep = 1 * unit.femtosecond

        self._number_of_iterations = 1

        self._steps_per_iteration = 500
        self._checkpoint_interval = 50

        self._force_field_path = None

        self._verbose = False

        self._estimated_free_energy = None

    def _get_options_dictionary(self):
        """Returns a dictionary of options which will be serialized
        to a yaml file and passed to YANK.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of YANK options.
        """

        # TODO: Should we be specifying `constraints` here?
        #       I imagine we shouldn't when passing OMM
        #       system.xml files.
        #
        # TODO: Same with `anisotropic_dispersion_cutoff`

        from openforcefield.utils import quantity_to_string

        return {
            'verbose': self._verbose,
            'output_dir': '.',

            'temperature': quantity_to_string(self._thermodynamic_state.temperature),
            'pressure': quantity_to_string(self._thermodynamic_state.pressure),

            'minimize': True,

            'default_number_of_iterations': self._number_of_iterations,
            'default_nsteps_per_iteration': self._steps_per_iteration,
            'checkpoint_interval': self._checkpoint_interval,

            'default_timestep': quantity_to_string(self._timestep),

            # TODO: Are these always sensible choices? I guess
            #       implementations of this class can override
            #       this where needed.
            'annihilate_electrostatics': True,
            'annihilate_sterics': False
        }

    def _get_solvent_dictionary(self):
        """Returns a dictionary of the solvent which will be serialized
        to a yaml file and passed to YANK. In most cases, this should
        just be passing force field settings over, such as PME settings.

        # TODO: Again, is this neccessary if we are using system.xml
        #       files...

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of YANK solvents.
        """
        from openforcefield.typing.engines.smirnoff.forcefield import ForceField

        force_field = ForceField(self._force_field_path)

        if not np.isclose(force_field.get_parameter_handler('Electrostatics').cutoff.value_in_unit(unit.angstrom),
                          force_field.get_parameter_handler('vdW').cutoff.value_in_unit(unit.angstrom)):

            raise ValueError('The electrostatic and vdW cutoffs must be identical.')

        cutoff = force_field.get_parameter_handler('vdW').cutoff
        switch_distance = force_field.get_parameter_handler('vdW').switch_width

        charge_method = force_field.get_parameter_handler('Electrostatics').method

        if charge_method.lower() != 'pme':
            raise ValueError('Currently only PME electrostatics are supported.')

        # Do we need to include `ewald_error_tolerance`?
        return {'default': {
            'nonbonded_method': charge_method,
            'nonbonded_cutoff': cutoff.value_in_unit(unit.nanometers),

            'switch_distance': switch_distance.value_in_unit(unit.nanometers),
        }}

    def _get_system_dictionary(self):
        """Returns a dictionary of the system which will be serialized
        to a yaml file and passed to YANK. Only a single system may be
        specified.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of YANK systems.
        """
        raise NotImplementedError()

    def _get_protocol_dictionary(self):
        """Returns a dictionary of the protocol which will be serialized
        to a yaml file and passed to YANK. Only a single protocol may be
        specified.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of a YANK protocol.
        """
        raise NotImplementedError()

    def _get_full_input_dictionary(self):
        """Returns a dictionary of the full YANK inputs which will be serialized
        to a yaml file and passed to YANK

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of a YANK input file.
        """

        system_dictionary = self._get_system_dictionary()
        system_key = next(iter(system_dictionary))

        protocol_dictionary = self._get_protocol_dictionary()
        protocol_key = next(iter(protocol_dictionary))

        return {
            'options': self._get_options_dictionary(),

            'solvents': self._get_solvent_dictionary(),

            'systems': system_dictionary,
            'protocols': protocol_dictionary,

            'experiments': {
                'system': system_key,
                'protocol': protocol_key
            }
        }

    @staticmethod
    def _extract_trajectory(checkpoint_path, output_trajectory_path):
        """Extracts the stored trajectory of the 'initial' state from a
        yank `.nc` checkpoint file and stores it to disk as a `.dcd` file.

        Parameters
        ----------
        checkpoint_path: str
            The path to the yank `.nc` file
        output_trajectory_path: str
            The path to store the extracted trajectory at.
        """

        from yank.analyze import extract_trajectory

        mdtraj_trajectory = extract_trajectory(checkpoint_path, state_index=0)
        mdtraj_trajectory.save_dcd(output_trajectory_path)

    @staticmethod
    @contextlib.contextmanager
    def _temporarily_change_directory(file_path):
        """A context to temporarily change the working directory.

        Parameters
        ----------
        file_path: str
            The file path to temporarily change into.
        """
        prev_dir = os.getcwd()
        os.chdir(path.abspath(file_path))

        try:
            yield
        finally:
            os.chdir(prev_dir)

    def execute(self, directory, available_resources):

        from yank.experiment import ExperimentBuilder
        from yank.analyze import ExperimentAnalyzer

        yaml_filename = path.join(directory, 'yank.yaml')

        # Create the yank yaml input file from a dictionary of options.
        with open(yaml_filename, 'w') as file:
            yaml.dump(self._get_full_input_dictionary(), file)

        with self._temporarily_change_directory(directory):

            exp_builder = ExperimentBuilder('yank.yaml')
            exp_builder.run_experiments()

            analyzer = ExperimentAnalyzer('experiments')
            output = analyzer.auto_analyze()

            free_energy = output['free_energy']['free_energy_diff_unit']
            free_energy_uncertainty = output['free_energy']['free_energy_diff_error_unit']

            self._estimated_free_energy = EstimatedQuantity(free_energy,
                                                            free_energy_uncertainty,
                                                            self._id)

        return self._get_output_dictionary()


@register_calculation_protocol()
class LigandReceptorYankProtocol(BaseYankProtocol):
    """An abstract base class for protocols which will performs a set of
    alchemical free energy simulations using the YANK framework.

    Protocols which inherit from this base must implement the abstract
    `_get_*_dictionary` methods.
    """

    @protocol_input(str)
    def ligand_residue_name(self):
        """The residue name of the ligand."""
        pass

    @protocol_input(str)
    def solvated_ligand_coordinates(self):
        """The file path to the solvated ligand coordinates."""
        pass

    @protocol_input(str)
    def solvated_ligand_system(self):
        """The file path to the solvated ligand system object."""
        pass

    @protocol_input(str)
    def solvated_complex_coordinates(self):
        """The file path to the solvated complex coordinates."""
        pass

    @protocol_input(str)
    def solvated_complex_system(self):
        """The file path to the solvated complex system object."""
        pass

    @protocol_output(str)
    def solvated_ligand_trajectory_path(self):
        """The file path to the generated ligand trajectory."""
        pass

    @protocol_output(str)
    def solvated_complex_trajectory_path(self):
        """The file path to the generated ligand trajectory."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new LigandReceptorYankProtocol object."""

        super().__init__(protocol_id)

        self._ligand_residue_name = None

        self._solvated_ligand_coordinates = None
        self._solvated_ligand_system = None

        self._solvated_complex_coordinates = None
        self._solvated_complex_system = None

        self._local_ligand_coordinates = 'ligand.pdb'
        self._local_ligand_system = 'ligand.xml'

        self._local_complex_coordinates = 'complex.pdb'
        self._local_complex_system = 'complex.xml'

        self._solvated_ligand_trajectory_path = None
        self._solvated_complex_trajectory_path = None

    def _get_system_dictionary(self):
        """Returns a dictionary of the system which will be serialized
        to a yaml file and passed to YANK. Only a single system may be
        specified.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of YANK systems.
        """

        solvent_dictionary = self._get_solvent_dictionary()
        solvent_key = next(iter(solvent_dictionary))

        host_guest_dictionary = {
            'phase1_path': [self._local_complex_system, self._local_complex_coordinates],
            'phase2_path': [self._local_ligand_system, self._local_ligand_coordinates],

            'ligand_dsl': f'resname {self._ligand_residue_name}',
            'solvent': solvent_key
        }

        return {'host-guest': host_guest_dictionary}

    def _get_protocol_dictionary(self):
        """Returns a dictionary of the protocol which will be serialized
        to a yaml file and passed to YANK. Only a single protocol may be
        specified.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of a YANK protocol.
        """

        absolute_binding_dictionary = {
            'complex': {'alchemical_path': 'auto'},
            'solvent': {'alchemical_path': 'auto'}
        }

        return {'absolute_binding_dictionary': absolute_binding_dictionary}

    def execute(self, directory, available_resources):

        # Because of quirks in where Yank looks files while doing temporary
        # directory changes, we need to copy the coordinate files locally so
        # they are correctly found.
        shutil.copyfile(self._solvated_ligand_coordinates, path.join(directory, self._local_ligand_coordinates))
        shutil.copyfile(self._solvated_ligand_system, path.join(directory, self._local_ligand_system))

        shutil.copyfile(self._solvated_complex_coordinates, path.join(directory, self._local_complex_coordinates))
        shutil.copyfile(self._solvated_complex_system, path.join(directory, self._local_complex_system))

        super(LigandReceptorYankProtocol, self).execute(directory, available_resources)

        ligand_yank_path = path.join(directory, 'experiments', 'solvent.nc')
        complex_yank_path = path.join(directory, 'experiments', 'complex.nc')

        self._solvated_ligand_trajectory_path = path.join(directory, 'ligand.dcd')
        self._solvated_complex_trajectory_path = path.join(directory, 'complex.dcd')

        self._extract_trajectory(ligand_yank_path, self._solvated_ligand_trajectory_path)
        self._extract_trajectory(complex_yank_path, self._solvated_complex_trajectory_path)

        return self._get_output_dictionary()
