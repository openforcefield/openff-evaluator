"""
A collection of protocols for running molecular simulations.
"""
import logging
import os
import shutil
import threading
import traceback
from enum import Enum

import numpy as np
import yaml
from simtk import unit, openmm
from simtk.openmm import app

from propertyestimator.thermodynamics import ThermodynamicState, Ensemble
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import setup_platform_with_resources
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import StatisticsArray, ObservableType
from propertyestimator.utils.utils import temporarily_change_directory, safe_unlink
from propertyestimator.workflow.decorators import protocol_input, protocol_output, MergeBehaviour
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class RunEnergyMinimisation(BaseProtocol):
    """A protocol to minimise the potential energy of a system.
    """

    @protocol_input(str)
    def input_coordinate_file(self):
        """The coordinates to minimise."""
        pass

    @protocol_input(unit.Quantity)
    def tolerance(self):
        """The energy tolerance to which the system should be minimized."""
        pass

    @protocol_input(int)
    def max_iterations(self):
        """The maximum number of iterations to perform.  If this is 0,
        minimization is continued until the results converge without regard
        to how many iterations it takes."""
        pass

    @protocol_input(str)
    def system_path(self):
        """The path to the XML system object which defines the forces present in the system."""
        pass

    @protocol_input(bool)
    def enable_pbc(self):
        """If true, periodic boundary conditions will be enabled."""
        pass

    @protocol_output(str)
    def output_coordinate_file(self):
        """The file path to the minimised coordinates."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._input_coordinate_file = None

        self._system_path = None
        self._system = None

        self._enable_pbc = True

        self._tolerance = 10*unit.kilojoules_per_mole
        self._max_iterations = 0

        self._output_coordinate_file = None

    def execute(self, directory, available_resources):

        logging.info('Minimising energy: ' + self.id)

        platform = setup_platform_with_resources(available_resources)

        input_pdb_file = app.PDBFile(self._input_coordinate_file)

        with open(self._system_path, 'rb') as file:
            self._system = openmm.XmlSerializer.deserialize(file.read().decode())

        if not self._enable_pbc:

            for force_index in range(self._system.getNumForces()):

                force = self._system.getForce(force_index)

                if not isinstance(force, openmm.NonbondedForce):
                    continue

                force.setNonbondedMethod(0)  # NoCutoff = 0, NonbondedMethod.CutoffNonPeriodic = 1

        # TODO: Expose the constraint tolerance
        integrator = openmm.VerletIntegrator(0.002 * unit.picoseconds)
        simulation = app.Simulation(input_pdb_file.topology, self._system, integrator, platform)

        box_vectors = input_pdb_file.topology.getPeriodicBoxVectors()

        if box_vectors is None:
            box_vectors = simulation.system.getDefaultPeriodicBoxVectors()

        simulation.context.setPeriodicBoxVectors(*box_vectors)
        simulation.context.setPositions(input_pdb_file.positions)

        simulation.minimizeEnergy(self._tolerance, self._max_iterations)

        positions = simulation.context.getState(getPositions=True).getPositions()

        self._output_coordinate_file = os.path.join(directory, 'minimised.pdb')

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

    @protocol_input(bool)
    def enable_pbc(self):
        """If true, periodic boundary conditions will be enabled."""
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
        self._timestep = 0.002 * unit.picoseconds

        self._output_frequency = 500000

        self._ensemble = Ensemble.NPT

        self._input_coordinate_file = None
        self._thermodynamic_state = None

        self._system_path = None

        self._enable_pbc = True

        self._output_coordinate_file = None

        self._trajectory_file_path = None
        self._statistics_file_path = None

        # Keep a track of the file names used for temporary working files.
        self._temporary_statistics_path = None
        self._temporary_trajectory_path = None

        self._checkpoint_path = None

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

        if Ensemble(self._ensemble) == Ensemble.NPT and self._enable_pbc is False:

            return PropertyEstimatorException(directory=directory,
                                              message='PBC must be enabled when running in the NPT ensemble.')

        logging.info('Performing a simulation in the ' + str(self._ensemble) + ' ensemble: ' + self.id)

        # Clean up any temporary files from previous (possibly failed)
        # simulations.
        self._temporary_statistics_path = os.path.join(directory, 'temp_statistics.csv')
        self._temporary_trajectory_path = os.path.join(directory, 'temp_trajectory.dcd')

        safe_unlink(self._temporary_statistics_path)
        safe_unlink(self._temporary_trajectory_path)

        self._checkpoint_path = os.path.join(directory, 'checkpoint.xml')

        # Set up the output file paths
        self._trajectory_file_path = os.path.join(directory, 'trajectory.dcd')
        self._statistics_file_path = os.path.join(directory, 'statistics.csv')

        # Set up the simulation objects.
        context, integrator = self._setup_simulation_objects(temperature,
                                                             pressure,
                                                             available_resources)

        result = self._simulate(directory, temperature, pressure, context, integrator)
        return result

    def _setup_simulation_objects(self, temperature, pressure, available_resources):
        """Initializes the objects needed to perform the simulation.
        This comprises of a context, and an integrator.

        Parameters
        ----------
        temperature: simtk.unit.Quantity
            The temperature to run the simulation at.
        pressure: simtk.unit.Quantity
            The pressure to run the simulation at.
        available_resources: ComputeResources
            The resources available to run on.

        Returns
        -------
        simtk.openmm.Context
            The created openmm context which takes advantage
            of the available compute resources.
        openmmtools.integrators.LangevinIntegrator
            The Langevin integrator which will propogate
            the simulation.
        """

        import openmmtools
        from simtk.openmm import XmlSerializer

        # Create a platform with the correct resources.
        platform = setup_platform_with_resources(available_resources)

        # Load in the system object from the provided xml file.
        with open(self._system_path, 'r') as file:
            system = XmlSerializer.deserialize(file.read())

        # Disable the periodic boundary conditions if requested.
        if not self._enable_pbc:

            for force_index in range(system.getNumForces()):
                force = system.getForce(force_index)

                if not isinstance(force, openmm.NonbondedForce):
                    continue

                force.setNonbondedMethod(0)  # NoCutoff = 0, NonbondedMethod.CutoffNonPeriodic = 1

            pressure = None

        # Use the openmmtools ThermodynamicState object to help
        # set up a system which contains the correct barostat if
        # one should be present.
        openmm_state = openmmtools.states.ThermodynamicState(system=system,
                                                             temperature=temperature,
                                                             pressure=pressure)

        system = openmm_state.get_system(remove_thermostat=True)

        # Set up the integrator.
        integrator = openmmtools.integrators.LangevinIntegrator(temperature=temperature,
                                                                collision_rate=self._thermostat_friction,
                                                                timestep=self._timestep)

        # Create the simulation context.
        context = openmm.Context(system, integrator, platform)

        # Initialize the context with the correct positions etc.
        if os.path.isfile(self._checkpoint_path):

            # Load the simulation state from a checkpoint file.
            logging.info(f'Loading the checkpoint from {self._checkpoint_path}.')

            with open(self._checkpoint_path, 'r') as file:
                checkpoint_state = XmlSerializer.deserialize(file.read())

            context.setState(checkpoint_state)

        else:

            logging.info(f'No checkpoint file was found at {self._checkpoint_path}.')

            # Populate the simulation object from the starting input files.
            input_pdb_file = app.PDBFile(self._input_coordinate_file)

            if self._enable_pbc:

                # Optionally set up the box vectors.
                box_vectors = input_pdb_file.topology.getPeriodicBoxVectors()

                if box_vectors is None:

                    raise ValueError('The input file must contain box vectors '
                                     'when running with PBC.')

                context.setPeriodicBoxVectors(*box_vectors)

            context.setPositions(input_pdb_file.positions)
            context.setVelocitiesToTemperature(temperature)

        return context, integrator

    def _write_statistics_array(self, raw_statistics, temperature, pressure,
                                degrees_of_freedom, total_mass):
        """Appends a set of statistics to an existing `statistics_array`.
        Those statistics are potential energy, kinetic energy, total energy,
        volume, density and reduced potential.

        Parameters
        ----------
        raw_statistics: dict of ObservableType and float
            A dictionary of potential energies, kinetic energies and volumes.
        temperature: simtk.unit.Quantity
            The temperature the system is being simulated at.
        pressure: simtk.unit.Quantity
            The pressure the system is being simulated at.
        degrees_of_freedom: int
            The number of degrees of freedom the system has.
        total_mass: simtk.unit.Quantity
            The total mass of the system.

        Returns
        -------
        StatisticsArray
            The statistics array with statistics appended.
        """
        beta = 1.0 / (unit.BOLTZMANN_CONSTANT_kB * temperature)

        potential_energies = np.array(raw_statistics[ObservableType.PotentialEnergy]) * unit.kilojoule_per_mole
        kinetic_energies = np.array(raw_statistics[ObservableType.KineticEnergy]) * unit.kilojoule_per_mole
        volumes = np.array(raw_statistics[ObservableType.Volume]) * unit.angstrom ** 3

        # Calculate the instantaneous temperature, taking account the
        # systems degrees of freedom.
        temperatures = 2.0 * kinetic_energies / (degrees_of_freedom * unit.MOLAR_GAS_CONSTANT_R)

        # Calculate the systems enthalpy and reduced potential.
        total_energies = potential_energies + kinetic_energies
        enthalpies = None

        reduced_potentials = potential_energies / unit.AVOGADRO_CONSTANT_NA

        if pressure is not None:

            pv_terms = pressure * volumes

            reduced_potentials += pv_terms
            enthalpies = total_energies + pv_terms * unit.AVOGADRO_CONSTANT_NA

        reduced_potentials = (beta * reduced_potentials) * unit.dimensionless

        # Calculate the systems density.
        densities = total_mass / (volumes * unit.AVOGADRO_CONSTANT_NA)

        statistics_array = StatisticsArray()

        statistics_array[ObservableType.PotentialEnergy] = potential_energies
        statistics_array[ObservableType.KineticEnergy] = kinetic_energies
        statistics_array[ObservableType.TotalEnergy] = total_energies
        statistics_array[ObservableType.Temperature] = temperatures
        statistics_array[ObservableType.Volume] = volumes
        statistics_array[ObservableType.Density] = densities
        statistics_array[ObservableType.ReducedPotential] = reduced_potentials

        if enthalpies is not None:
            statistics_array[ObservableType.Enthalpy] = enthalpies

        statistics_array.to_pandas_csv(self._temporary_statistics_path)

    def _simulate(self, directory, temperature, pressure, context, integrator):
        """Performs the simulation using a given context
        and integrator.

        Parameters
        ----------
        directory: str
            The directory the trajectory is being run in.
        temperature: simtk.unit.Quantity
            The temperature to run the simulation at.
        pressure: simtk.unit.Quantity
            The pressure to run the simulation at.
        context: simtk.openmm.Context
            The OpenMM context to run with.
        integrator: simtk.openmm.Integrator
            The integrator to evolve the simulation with.
        """
        import mdtraj

        # Build the reporters which we will use to report the state
        # of the simulation.
        input_pdb_file = app.PDBFile(self._input_coordinate_file)
        topology = input_pdb_file.topology

        with open(os.path.join(directory, 'input.pdb'), 'w+') as configuration_file:
            app.PDBFile.writeFile(input_pdb_file.topology, input_pdb_file.positions, configuration_file)

        trajectory_file_object = open(self._temporary_trajectory_path, 'wb')
        trajectory_dcd_object = app.DCDFile(trajectory_file_object,
                                            topology,
                                            integrator.getStepSize(),
                                            0,
                                            self._output_frequency,
                                            False)

        raw_statistics = {
            ObservableType.PotentialEnergy: [],
            ObservableType.KineticEnergy: [],
            ObservableType.Volume: []
        }

        # Define any constants needed for extracting system statistics
        # Compute the instantaneous temperature of degrees of freedom.
        # This snipped is taken from the build in OpenMM `StateDataReporter`
        system = context.getSystem()

        degrees_of_freedom = sum([3 for i in range(system.getNumParticles()) if
                                  system.getParticleMass(i) > 0 * unit.dalton])

        degrees_of_freedom -= system.getNumConstraints()

        if any(type(system.getForce(i)) == openmm.CMMotionRemover for i in range(system.getNumForces())):
            degrees_of_freedom -= 3

        total_mass = 0.0 * unit.dalton

        for i in range(system.getNumParticles()):
            total_mass += system.getParticleMass(i)

        # Perform the simulation.
        current_step_count = 0
        result = None

        try:

            while current_step_count < self.steps:

                steps_to_take = min(self._output_frequency, self.steps - current_step_count)
                integrator.step(steps_to_take)

                state = context.getState(getPositions=True,
                                         getVelocities=True,
                                         getForces=True,
                                         getEnergy=True,
                                         getParameters=True,
                                         enforcePeriodicBox=self.enable_pbc)

                # Write out the current frame of the trajectory.
                trajectory_dcd_object.writeModel(positions=state.getPositions(),
                                                 periodicBoxVectors=state.getPeriodicBoxVectors())

                # Write out the energies and system statistics.
                raw_statistics[ObservableType.PotentialEnergy].append(state.getPotentialEnergy().
                                                                      value_in_unit(unit.kilojoule_per_mole))
                raw_statistics[ObservableType.KineticEnergy].append(state.getKineticEnergy().
                                                                    value_in_unit(unit.kilojoule_per_mole))
                raw_statistics[ObservableType.Volume].append(state.getPeriodicBoxVolume().
                                                             value_in_unit(unit.angstrom ** 3))

                self._write_statistics_array(raw_statistics, temperature, pressure,
                                             degrees_of_freedom, total_mass)

                # Create a checkpoint file.
                state_xml = openmm.XmlSerializer.serialize(state)

                with open(self._checkpoint_path, 'w') as file:
                    file.write(state_xml)

                current_step_count += steps_to_take

        except Exception as e:

            result = PropertyEstimatorException(directory=directory,
                                                message='Simulation failed: {}'.format(e))

        # Make sure to close the open trajectory stream.
        trajectory_file_object.close()

        if isinstance(result, PropertyEstimatorException):
            return result

        # Move the trajectory and statistics files to their
        # final location.
        if not os.path.isfile(self._trajectory_file_path):
            os.replace(self._temporary_trajectory_path, self._trajectory_file_path)
        else:
            mdtraj_topology = mdtraj.Topology.from_openmm(topology)

            existing_trajectory = mdtraj.load_dcd(self._trajectory_file_path, mdtraj_topology)
            current_trajectory = mdtraj.load_dcd(self._temporary_trajectory_path, mdtraj_topology)

            concatenated_trajectory = mdtraj.join([existing_trajectory,
                                                   current_trajectory],
                                                   check_topology=False,
                                                   discard_overlapping_frames=False)

            concatenated_trajectory.save_dcd(self._trajectory_file_path)

        if not os.path.isfile(self._statistics_file_path):
            os.replace(self._temporary_statistics_path, self._statistics_file_path)
        else:

            existing_statistics = StatisticsArray.from_pandas_csv(self._statistics_file_path)
            current_statistics = StatisticsArray.from_pandas_csv(self._temporary_statistics_path)

            concatenated_statistics = StatisticsArray.join(existing_statistics,
                                                           current_statistics)

            concatenated_statistics.to_pandas_csv(self._statistics_file_path)

        # Save out the final positions.
        final_state = context.getState(getPositions=True)
        positions = final_state.getPositions()
        topology.setPeriodicBoxVectors(final_state.getPeriodicBoxVectors())

        self._output_coordinate_file = os.path.join(directory, 'output.pdb')

        with open(self._output_coordinate_file, 'w+') as configuration_file:
            app.PDBFile.writeFile(topology,positions, configuration_file)

        logging.info(f'Simulation performed in the {str(self._ensemble)} ensemble: {self._id}')
        return self._get_output_dictionary()


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
        self._timestep = 2 * unit.femtosecond

        self._number_of_iterations = 1

        self._steps_per_iteration = 500
        self._checkpoint_interval = 50

        self._force_field_path = None

        self._verbose = False

        self._estimated_free_energy = None

    def _get_options_dictionary(self, available_resources):
        """Returns a dictionary of options which will be serialized
        to a yaml file and passed to YANK.

        Parameters
        ----------
        available_resources: ComputeResources
            The resources available to execute on.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of YANK options.
        """

        from openforcefield.utils import quantity_to_string

        platform_name = 'CPU'

        if available_resources.number_of_gpus > 0:

            # A platform which runs on GPUs has been requested.
            from propertyestimator.backends import ComputeResources
            toolkit_enum = ComputeResources.GPUToolkit(available_resources.preferred_gpu_toolkit)

            # A platform which runs on GPUs has been requested.
            platform_name = 'CUDA' if toolkit_enum == ComputeResources.GPUToolkit.CUDA else \
                                                      ComputeResources.GPUToolkit.OpenCL

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

            'annihilate_electrostatics': True,
            'annihilate_sterics': False,

            'platform': platform_name
        }

    def _get_solvent_dictionary(self):
        """Returns a dictionary of the solvent which will be serialized
        to a yaml file and passed to YANK. In most cases, this should
        just be passing force field settings over, such as PME settings.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of YANK solvents.
        """
        from openforcefield.typing.engines.smirnoff.forcefield import ForceField
        force_field = ForceField(self._force_field_path)

        charge_method = force_field.get_parameter_handler('Electrostatics').method

        if charge_method.lower() != 'pme':
            raise ValueError('Currently only PME electrostatics are supported.')

        return {'default': {
            'nonbonded_method': charge_method,
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

    def _get_experiments_dictionary(self):
        """Returns a dictionary of the experiments which will be serialized
        to a yaml file and passed to YANK. Only a single experiment may be
        specified.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of a YANK experiment.
        """

        system_dictionary = self._get_system_dictionary()
        system_key = next(iter(system_dictionary))

        protocol_dictionary = self._get_protocol_dictionary()
        protocol_key = next(iter(protocol_dictionary))

        return {
            'system': system_key,
            'protocol': protocol_key
        }

    def _get_full_input_dictionary(self, available_resources):
        """Returns a dictionary of the full YANK inputs which will be serialized
        to a yaml file and passed to YANK

        Parameters
        ----------
        available_resources: ComputeResources
            The resources available to execute on.

        Returns
        -------
        dict of str and Any
            A yaml compatible dictionary of a YANK input file.
        """

        return {
            'options': self._get_options_dictionary(available_resources),

            'solvents': self._get_solvent_dictionary(),

            'systems': self._get_system_dictionary(),
            'protocols': self._get_protocol_dictionary(),

            'experiments': self._get_experiments_dictionary()
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

        mdtraj_trajectory = extract_trajectory(checkpoint_path, state_index=0, image_molecules=True)
        mdtraj_trajectory.save_dcd(output_trajectory_path)

    @staticmethod
    def _run_yank(directory, available_resources):
        """Runs YANK within the specified directory which contains a `yank.yaml`
        input file.

        Parameters
        ----------
        directory: str
            The directory within which to run yank.

        Returns
        -------
        simtk.unit.Quantity
            The free energy returned by yank.
        simtk.unit.Quantity
            The uncertainty in the free energy returned by yank.
        """

        from yank.experiment import ExperimentBuilder
        from yank.analyze import ExperimentAnalyzer

        with temporarily_change_directory(directory):

            # Set the default properties on the desired platform
            # before calling into yank.
            setup_platform_with_resources(available_resources)

            exp_builder = ExperimentBuilder('yank.yaml')
            exp_builder.run_experiments()

            analyzer = ExperimentAnalyzer('experiments')
            output = analyzer.auto_analyze()

            free_energy = output['free_energy']['free_energy_diff_unit']
            free_energy_uncertainty = output['free_energy']['free_energy_diff_error_unit']

        return free_energy, free_energy_uncertainty

    @staticmethod
    def _run_yank_as_process(queue, directory, available_resources):
        """A wrapper around the `_run_yank` method which takes
        a `multiprocessing.Queue` as input, thereby allowing it
        to be launched from a separate process and still return
        it's output back to the main process.

        Parameters
        ----------
        queue: multiprocessing.Queue
            The queue object which will communicate with the
            launched process.
        directory: str
            The directory within which to run yank.

        Returns
        -------
        simtk.unit.Quantity
            The free energy returned by yank.
        simtk.unit.Quantity
            The uncertainty in the free energy returned by yank.
        str, optional
            The stringified errors which occurred on the other process,
            or `None` if no exceptions were raised.
        """

        free_energy = None
        free_energy_uncertainty = None

        error = None

        try:
            free_energy, free_energy_uncertainty = BaseYankProtocol._run_yank(directory, available_resources)
        except Exception as e:
            error = traceback.format_exception(None, e, e.__traceback__)

        queue.put((free_energy, free_energy_uncertainty, error))

    def execute(self, directory, available_resources):

        yaml_filename = os.path.join(directory, 'yank.yaml')

        # Create the yank yaml input file from a dictionary of options.
        with open(yaml_filename, 'w') as file:
            yaml.dump(self._get_full_input_dictionary(available_resources), file)

        # Yank is not safe to be called from anything other than the main thread.
        # If the current thread is not detected as the main one, then yank should
        # be spun up in a new process which should itself be safe to run yank in.
        if threading.current_thread() is threading.main_thread():
            logging.info('Launching YANK in the main thread.')
            free_energy, free_energy_uncertainty = self._run_yank(directory, available_resources)
        else:

            from multiprocessing import Process, Queue

            logging.info('Launching YANK in a new process.')

            # Create a queue to pass the results back to the main process.
            queue = Queue()
            # Create the process within which yank will run.
            process = Process(target=BaseYankProtocol._run_yank_as_process, args=[queue, directory,
                                                                                  available_resources])

            # Start the process and gather back the output.
            process.start()
            free_energy, free_energy_uncertainty, error = queue.get()
            process.join()

            if error is not None:
                return PropertyEstimatorException(directory, error)

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

    class RestraintType(Enum):
        """The types of ligand restraints available within yank.
        """
        Harmonic = 'Harmonic'
        FlatBottom = 'FlatBottom'

    @protocol_input(str)
    def ligand_residue_name(self):
        """The residue name of the ligand."""
        pass

    @protocol_input(str)
    def receptor_residue_name(self):
        """The residue name of the receptor."""
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

    @protocol_input(bool)
    def apply_restraints(self):
        """Determines whether the ligand should be explicitly restrained to
        the receptor in order to stop the ligand from temporarily unbinding.
        """
        pass

    @protocol_input(RestraintType)
    def restraint_type(self):
        """The type of ligand restraint applied, provided that `apply_restraints`
        is `True`"""
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
        self._receptor_residue_name = None

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

        self._apply_restraints = True
        self._restraint_type = LigandReceptorYankProtocol.RestraintType.Harmonic

    def _get_system_dictionary(self):

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

        absolute_binding_dictionary = {
            'complex': {'alchemical_path': 'auto'},
            'solvent': {'alchemical_path': 'auto'}
        }

        return {'absolute_binding_dictionary': absolute_binding_dictionary}

    def _get_experiments_dictionary(self):

        experiments_dictionary = super(LigandReceptorYankProtocol, self)._get_experiments_dictionary()

        if self._apply_restraints:

            experiments_dictionary['restraint'] = {
                'restrained_ligand_atoms': f'(resname {self._ligand_residue_name}) and (mass > 1.5)',
                'restrained_receptor_atoms': f'(resname {self._receptor_residue_name}) and (mass > 1.5)',

                'type': self._restraint_type.value
            }

        return experiments_dictionary

    def execute(self, directory, available_resources):

        # Because of quirks in where Yank looks files while doing temporary
        # directory changes, we need to copy the coordinate files locally so
        # they are correctly found.
        shutil.copyfile(self._solvated_ligand_coordinates, os.path.join(directory, self._local_ligand_coordinates))
        shutil.copyfile(self._solvated_ligand_system, os.path.join(directory, self._local_ligand_system))

        shutil.copyfile(self._solvated_complex_coordinates, os.path.join(directory, self._local_complex_coordinates))
        shutil.copyfile(self._solvated_complex_system, os.path.join(directory, self._local_complex_system))

        result = super(LigandReceptorYankProtocol, self).execute(directory, available_resources)

        if isinstance(result, PropertyEstimatorException):
            return result

        ligand_yank_path = os.path.join(directory, 'experiments', 'solvent.nc')
        complex_yank_path = os.path.join(directory, 'experiments', 'complex.nc')

        self._solvated_ligand_trajectory_path = os.path.join(directory, 'ligand.dcd')
        self._solvated_complex_trajectory_path = os.path.join(directory, 'complex.dcd')

        self._extract_trajectory(ligand_yank_path, self._solvated_ligand_trajectory_path)
        self._extract_trajectory(complex_yank_path, self._solvated_complex_trajectory_path)

        return self._get_output_dictionary()
