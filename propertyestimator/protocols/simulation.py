"""
A collection of protocols for running molecular simulations.
"""
import io
import json
import logging
import os
import re
import traceback

import numpy as np

import pandas as pd
from simtk import openmm, unit as simtk_unit
from simtk.openmm import app

from propertyestimator import unit
from propertyestimator.thermodynamics import ThermodynamicState, Ensemble
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import setup_platform_with_resources, pint_quantity_to_openmm, disable_pbc
from propertyestimator.utils.serialization import TypedJSONEncoder, TypedJSONDecoder
from propertyestimator.utils.statistics import StatisticsArray, ObservableType
from propertyestimator.workflow.decorators import protocol_input, protocol_output, InequalityMergeBehaviour, UNDEFINED
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class RunEnergyMinimisation(BaseProtocol):
    """A protocol to minimise the potential energy of a system.
    """

    input_coordinate_file = protocol_input(
        docstring='The coordinates to minimise.',
        type_hint=str,
        default_value=UNDEFINED
    )
    system_path = protocol_input(
        docstring='The path to the XML system object which defines the forces present '
                  'in the system.',
        type_hint=str,
        default_value=UNDEFINED
    )

    tolerance = protocol_input(
        docstring='The energy tolerance to which the system should be minimized.',
        type_hint=unit.Quantity,
        default_value=10 * unit.kilojoules / unit.mole
    )
    max_iterations = protocol_input(
        docstring='The maximum number of iterations to perform. If this is 0, '
                  'minimization is continued until the results converge without regard to '
                  'how many iterations it takes.',
        type_hint=int,
        default_value=10
    )

    enable_pbc = protocol_input(
        docstring='If true, periodic boundary conditions will be enabled.',
        type_hint=bool,
        default_value=True
    )

    output_coordinate_file = protocol_output(
        docstring='The file path to the minimised coordinates.',
        type_hint=str
    )

    def execute(self, directory, available_resources):

        logging.info('Minimising energy: ' + self.id)

        platform = setup_platform_with_resources(available_resources)

        input_pdb_file = app.PDBFile(self.input_coordinate_file)

        with open(self.system_path, 'rb') as file:
            system = openmm.XmlSerializer.deserialize(file.read().decode())

        if not self.enable_pbc:

            for force_index in range(system.getNumForces()):

                force = system.getForce(force_index)

                if not isinstance(force, openmm.NonbondedForce):
                    continue

                force.setNonbondedMethod(0)  # NoCutoff = 0, NonbondedMethod.CutoffNonPeriodic = 1

        # TODO: Expose the constraint tolerance
        integrator = openmm.VerletIntegrator(0.002 * simtk_unit.picoseconds)
        simulation = app.Simulation(input_pdb_file.topology, system, integrator, platform)

        box_vectors = input_pdb_file.topology.getPeriodicBoxVectors()

        if box_vectors is None:
            box_vectors = simulation.system.getDefaultPeriodicBoxVectors()

        simulation.context.setPeriodicBoxVectors(*box_vectors)
        simulation.context.setPositions(input_pdb_file.positions)

        simulation.minimizeEnergy(pint_quantity_to_openmm(self.tolerance), self.max_iterations)

        positions = simulation.context.getState(getPositions=True).getPositions()

        self.output_coordinate_file = os.path.join(directory, 'minimised.pdb')

        with open(self.output_coordinate_file, 'w+') as minimised_file:
            app.PDBFile.writeFile(simulation.topology, positions, minimised_file)

        logging.info('Energy minimised: ' + self.id)

        return self._get_output_dictionary()


@register_calculation_protocol()
class BaseOpenMMSimulation(BaseProtocol):

    steps_per_iteration = protocol_input(
        docstring='The number of steps to propogate the system by at '
                  'each iteration. The total number of steps performed '
                  'by this protocol will be `total_number_of_iterations * '
                  'steps_per_iteration`.',
        type_hint=int, merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=1000000
    )
    total_number_of_iterations = protocol_input(
        docstring='The number of times to propogate the system forward by the '
                  '`steps_per_iteration` number of steps. The total number of '
                  'steps performed by this protocol will be `total_number_of_iterations * '
                  'steps_per_iteration`.',
        type_hint=int, merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=1
    )

    timestep = protocol_input(
        docstring='The timestep to evolve the system by at each step.',
        type_hint=unit.Quantity, merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=2.0 * unit.femtosecond
    )

    thermodynamic_state = protocol_input(
        docstring='The thermodynamic conditions to simulate under',
        type_hint=ThermodynamicState,
        default_value=UNDEFINED
    )
    ensemble = protocol_input(
        docstring='The thermodynamic ensemble to simulate in.',
        type_hint=Ensemble,
        default_value=Ensemble.NPT
    )

    thermostat_friction = protocol_input(
        docstring='The thermostat friction coefficient.',
        type_hint=unit.Quantity, merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=1.0 / unit.picoseconds
    )

    input_coordinate_file = protocol_input(
        docstring='The file path to the starting coordinates.',
        type_hint=str,
        default_value=UNDEFINED
    )
    system_path = protocol_input(
        docstring='A path to the XML system object which defines the forces present '
                  'in the system.',
        type_hint=str,
        default_value=UNDEFINED
    )

    enable_pbc = protocol_input(
        docstring='If true, periodic boundary conditions will be enabled.',
        type_hint=bool,
        default_value=True
    )

    allow_gpu_platforms = protocol_input(
        docstring='If true, OpenMM will be allowed to run using a GPU if available, '
                  'otherwise it will be constrained to only using CPUs.',
        type_hint=bool,
        default_value=True
    )
    high_precision = protocol_input(
        docstring='If true, OpenMM will be run using a platform with high precision '
                  'settings. This will be the Reference platform when only a CPU is '
                  'available, or double precision mode when a GPU is available.',
        type_hint=bool,
        default_value=False
    )

    output_coordinate_file = protocol_output(
        docstring='The file path to the coordinates of the final system configuration.',
        type_hint=str
    )
    trajectory_file_path = protocol_output(
        docstring='The file path to the trajectory sampled during the simulation.',
        type_hint=str
    )
    statistics_file_path = protocol_output(
        docstring='The file path to the statistics sampled during the simulation.',
        type_hint=str
    )

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._local_trajectory_path = None
        self._local_statistics_path = None

        self._context = None
        self._integrator = None


@register_calculation_protocol()
class RunOpenMMSimulation(BaseOpenMMSimulation):
    """Performs a molecular dynamics simulation in a given ensemble using
    an OpenMM backend.
    """
    class _Checkpoint:
        """A temporary checkpoint file which keeps track
        of the parts of the simulation state not stored in
        the checkpoint state xml file.
        """

        def __init__(self, output_frequency=-1, checkpoint_frequency=-1,
                     steps_per_iteration=-1, current_step_number=0):

            self.output_frequency = output_frequency
            self.checkpoint_frequency = checkpoint_frequency
            self.steps_per_iteration = steps_per_iteration
            self.current_step_number = current_step_number

        def __getstate__(self):
            return {
                'output_frequency': self.output_frequency,
                'checkpoint_frequency': self.checkpoint_frequency,
                'steps_per_iteration': self.steps_per_iteration,
                'current_step_number': self.current_step_number
            }

        def __setstate__(self, state):
            self.output_frequency = state['output_frequency']
            self.checkpoint_frequency = state['checkpoint_frequency']
            self.steps_per_iteration = state['steps_per_iteration']
            self.current_step_number = state['current_step_number']

    class _Simulation:
        """A fake simulation class to use with the
        openmm file reporters.
        """

        def __init__(self, integrator, topology, system, current_step):
            self.integrator = integrator
            self.topology = topology
            self.system = system
            self.currentStep = current_step

    output_frequency = protocol_input(
        docstring='The frequency (in number of steps) with which to write to the '
                  'output statistics and trajectory files.',
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=3000
    )
    checkpoint_frequency = protocol_input(
        docstring='The frequency (in multiples of `output_frequency`) with which to '
                  'write to a checkpoint file, e.g. if `output_frequency=100` and '
                  '`checkpoint_frequency==2`, a checkpoint file would be saved every '
                  '200 steps.',
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        optional=True,
        default_value=10
    )

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._checkpoint_path = None
        self._state_path = None

    def execute(self, directory, available_resources):

        # We handle most things in OMM units here.
        temperature = self.thermodynamic_state.temperature
        openmm_temperature = pint_quantity_to_openmm(temperature)

        pressure = None if self.ensemble == Ensemble.NVT else self.thermodynamic_state.pressure
        openmm_pressure = pint_quantity_to_openmm(pressure)

        if openmm_temperature is None:

            return PropertyEstimatorException(directory=directory,
                                              message='A temperature must be set to perform '
                                                      'a simulation in any ensemble')

        if Ensemble(self.ensemble) == Ensemble.NPT and openmm_pressure is None:

            return PropertyEstimatorException(directory=directory,
                                              message='A pressure must be set to perform an NPT simulation')

        if Ensemble(self.ensemble) == Ensemble.NPT and self.enable_pbc is False:

            return PropertyEstimatorException(directory=directory,
                                              message='PBC must be enabled when running in the NPT ensemble.')

        logging.info('Performing a simulation in the ' + str(self.ensemble) + ' ensemble: ' + self.id)

        # Set up the internal file paths
        self._checkpoint_path = os.path.join(directory, 'checkpoint.json')
        self._state_path = os.path.join(directory, 'checkpoint_state.xml')

        self._local_trajectory_path = os.path.join(directory, 'trajectory.dcd')
        self._local_statistics_path = os.path.join(directory, 'openmm_statistics.csv')

        # Set up the simulation objects.
        if self._context is None or self._integrator is None:

            self._context, self._integrator = self._setup_simulation_objects(openmm_temperature,
                                                                             openmm_pressure,
                                                                             available_resources)

        # Save a copy of the starting configuration if it doesn't already exist
        local_input_coordinate_path = os.path.join(directory, 'input.pdb')

        if not os.path.isfile(local_input_coordinate_path):

            input_pdb_file = app.PDBFile(self.input_coordinate_file)

            with open(local_input_coordinate_path, 'w+') as configuration_file:
                app.PDBFile.writeFile(input_pdb_file.topology, input_pdb_file.positions, configuration_file)

        # Run the simulation.
        result = self._simulate(directory, self._context, self._integrator)

        if isinstance(result, PropertyEstimatorException):
            return result

        # Set the output paths.
        self.trajectory_file_path = self._local_trajectory_path
        self.statistics_file_path = os.path.join(directory, 'statistics.csv')

        # Save out the final statistics in the property estimator format
        self._save_final_statistics(self.statistics_file_path, temperature, pressure)

        return self._get_output_dictionary()

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
        if not self.allow_gpu_platforms:

            from propertyestimator.backends import ComputeResources
            available_resources = ComputeResources(available_resources.number_of_threads)

        platform = setup_platform_with_resources(available_resources, self.high_precision)

        # Load in the system object from the provided xml file.
        with open(self.system_path, 'r') as file:
            system = XmlSerializer.deserialize(file.read())

        # Disable the periodic boundary conditions if requested.
        if not self.enable_pbc:

            disable_pbc(system)
            pressure = None

        # Use the openmmtools ThermodynamicState object to help
        # set up a system which contains the correct barostat if
        # one should be present.
        openmm_state = openmmtools.states.ThermodynamicState(system=system,
                                                             temperature=temperature,
                                                             pressure=pressure)

        system = openmm_state.get_system(remove_thermostat=True)

        # Set up the integrator.
        thermostat_friction = pint_quantity_to_openmm(self.thermostat_friction)
        timestep = pint_quantity_to_openmm(self.timestep)

        integrator = openmmtools.integrators.LangevinIntegrator(temperature=temperature,
                                                                collision_rate=thermostat_friction,
                                                                timestep=timestep)

        # Create the simulation context.
        context = openmm.Context(system, integrator, platform)

        # Initialize the context with the correct positions etc.
        input_pdb_file = app.PDBFile(self.input_coordinate_file)

        if self.enable_pbc:

            # Optionally set up the box vectors.
            box_vectors = input_pdb_file.topology.getPeriodicBoxVectors()

            if box_vectors is None:

                raise ValueError('The input file must contain box vectors '
                                 'when running with PBC.')

            context.setPeriodicBoxVectors(*box_vectors)

        context.setPositions(input_pdb_file.positions)
        context.setVelocitiesToTemperature(temperature)

        return context, integrator

    def _write_checkpoint_file(self, current_step_number, context):
        """Writes a simulation checkpoint file to disk.

        Parameters
        ----------
        current_step_number: int
            The total number of steps which have been taken so
            far.
        context: simtk.openmm.Context
            The current OpenMM context.
        """

        # Write the current state to disk
        state = context.getState(getPositions=True,
                                 getEnergy=True,
                                 getVelocities=True,
                                 getForces=True,
                                 getParameters=True,
                                 enforcePeriodicBox=self.enable_pbc)

        with open(self._state_path, 'w') as file:
            file.write(openmm.XmlSerializer.serialize(state))

        checkpoint = self._Checkpoint(self.output_frequency, self.checkpoint_frequency,
                                      self.steps_per_iteration, current_step_number)

        with open(self._checkpoint_path, 'w') as file:
            json.dump(checkpoint, file, cls=TypedJSONEncoder)

    def _truncate_statistics_file(self, number_of_frames):
        """Truncates the statistics file to the specified number
        of frames.

        Parameters
        ----------
        number_of_frames: int
            The number of frames to truncate to.
        """
        with open(self._local_statistics_path) as file:

            header_line = file.readline()
            file_contents = re.sub('#.*\n', '', file.read())

            with io.StringIO(file_contents) as string_object:
                existing_statistics_array = pd.read_csv(string_object, index_col=False, header=None)

        statistics_length = len(existing_statistics_array)

        if statistics_length < number_of_frames:

            raise ValueError(f'The saved number of statistics frames ({statistics_length}) '
                             f'is less than expected ({number_of_frames}).')

        elif statistics_length == number_of_frames:
            return

        truncated_statistics_array = existing_statistics_array[0:number_of_frames]

        with open(self._local_statistics_path, 'w') as file:

            file.write(f'{header_line}')
            truncated_statistics_array.to_csv(file, index=False, header=False)

    def _truncate_trajectory_file(self, number_of_frames):
        """Truncates the trajectory file to the specified number
        of frames.

        Parameters
        ----------
        number_of_frames: int
            The number of frames to truncate to.
        """
        import mdtraj
        from mdtraj.formats.dcd import DCDTrajectoryFile
        from mdtraj.utils import in_units_of

        # Load in the required topology object.
        topology = mdtraj.load_topology(self.input_coordinate_file)

        # Parse the internal mdtraj distance unit. While private access is
        # undesirable, this is never publicly defined and I believe this
        # route to be preferable over hard coding this unit here.
        base_distance_unit = mdtraj.Trajectory._distance_unit

        # Get an accurate measurement of the length of the trajectory
        # without reading it into memory.
        trajectory_length = 0

        for chunk in mdtraj.iterload(self._local_trajectory_path, top=topology):
            trajectory_length += len(chunk)

        # Make sure there is at least the expected number of frames.
        if trajectory_length < number_of_frames:

            raise ValueError(f'The saved number of trajectory frames ({trajectory_length}) '
                             f'is less than expected ({number_of_frames}).')

        elif trajectory_length == number_of_frames:
            return

        # Truncate the trajectory by streaming one frame of the trajectory at
        # a time.
        temporary_trajectory_path = f'{self._local_trajectory_path}.tmp'

        with DCDTrajectoryFile(self._local_trajectory_path, 'r') as input_file:

            with DCDTrajectoryFile(temporary_trajectory_path, 'w') as output_file:

                for frame_index in range(0, number_of_frames):

                    frame = input_file.read_as_traj(topology, n_frames=1, stride=1)

                    output_file.write(
                        xyz=in_units_of(frame.xyz, base_distance_unit, output_file.distance_unit),
                        cell_lengths=in_units_of(frame.unitcell_lengths, base_distance_unit,
                                                 output_file.distance_unit),
                        cell_angles=frame.unitcell_angles[0]
                    )

        os.replace(temporary_trajectory_path, self._local_trajectory_path)

        # Do a sanity check to make sure the trajectory was successfully truncated.
        new_trajectory_length = 0

        for chunk in mdtraj.iterload(self._local_trajectory_path, top=self.input_coordinate_file):
            new_trajectory_length += len(chunk)

        if new_trajectory_length != number_of_frames:
            raise ValueError('The trajectory was incorrectly truncated.')

    def _resume_from_checkpoint(self, context):
        """Resumes the simulation from a checkpoint file.

        Parameters
        ----------
        context: simtk.openmm.Context
            The current OpenMM context.

        Returns
        -------
        int
            The current step number.
        """
        current_step_number = 0

        # Check whether the checkpoint files actually exists.
        if (not os.path.isfile(self._checkpoint_path) or
            not os.path.isfile(self._state_path)):

            logging.info('No checkpoint files were found.')
            return current_step_number

        if (not os.path.isfile(self._local_statistics_path) or
            not os.path.isfile(self._local_trajectory_path)):

            raise ValueError('Checkpoint files were correctly found, but the trajectory '
                             'or statistics files seem to be missing. This should not happen.')

        logging.info('Restoring the system state from checkpoint files.')

        # If they do, load the current state from disk.
        with open(self._state_path, 'r') as file:
            current_state = openmm.XmlSerializer.deserialize(file.read())

        with open(self._checkpoint_path, 'r') as file:
            checkpoint = json.load(file, cls=TypedJSONDecoder)

        if (self.output_frequency != checkpoint.output_frequency or
            self.checkpoint_frequency != checkpoint.checkpoint_frequency or
            self.steps_per_iteration != checkpoint.steps_per_iteration):

            raise ValueError('Neither the output frequency, the checkpoint '
                             'frequency, nor the steps per iteration can '
                             'currently be changed during the course of the '
                             'simulation. Only the number of iterations is '
                             'allowed to change.')

        # Make sure this simulation hasn't already finished.
        total_expected_number_of_steps = self.total_number_of_iterations * self.steps_per_iteration

        if checkpoint.current_step_number == total_expected_number_of_steps:
            return checkpoint.current_step_number

        context.setState(current_state)

        # Make sure that the number of frames in the trajectory /
        # statistics file correspond to the recorded number of steps.
        # This is to handle possible cases where only some of the files
        # have been written from the current step (i.e only the trajectory may
        # have been written to before this protocol gets unexpectedly killed.
        expected_number_of_frames = int(checkpoint.current_step_number / self.output_frequency)

        # Handle the truncation of the statistics file.
        self._truncate_statistics_file(expected_number_of_frames)

        # Handle the truncation of the trajectory file.
        self._truncate_trajectory_file(expected_number_of_frames)

        logging.info('System state restored from checkpoint files.')

        return checkpoint.current_step_number

    def _save_final_statistics(self, path, temperature, pressure):
        """Converts the openmm statistic csv file into a propertyestimator
        StatisticsArray csv file, making sure to fill in any missing entries.

        Parameters
        ----------
        path: str
            The path to save the statistics to.
        temperature: unit.Quantity
            The temperature that the simulation is being run at.
        pressure: unit.Quantity
            The pressure that the simulation is being run at.
        """
        statistics = StatisticsArray.from_openmm_csv(self._local_statistics_path, pressure)

        reduced_potentials = statistics[ObservableType.PotentialEnergy] / unit.avogadro_number

        if pressure is not None:

            pv_terms = pressure * statistics[ObservableType.Volume]
            reduced_potentials += pv_terms

        beta = 1.0 / (unit.boltzmann_constant * temperature)
        statistics[ObservableType.ReducedPotential] = (beta * reduced_potentials).to(unit.dimensionless)

        statistics.to_pandas_csv(path)

    def _simulate(self, directory, context, integrator):
        """Performs the simulation using a given context
        and integrator.

        Parameters
        ----------
        directory: str
            The directory the trajectory is being run in.
        context: simtk.openmm.Context
            The OpenMM context to run with.
        integrator: simtk.openmm.Integrator
            The integrator to evolve the simulation with.
        """

        # Define how many steps should be taken.
        total_number_of_steps = self.total_number_of_iterations * self.steps_per_iteration

        # Try to load the current state from any available checkpoint information
        current_step = self._resume_from_checkpoint(context)

        if current_step == total_number_of_steps:
            return None

        # Build the reporters which we will use to report the state
        # of the simulation.
        append_trajectory = os.path.isfile(self._local_trajectory_path)
        dcd_reporter = app.DCDReporter(self._local_trajectory_path, 0, append_trajectory)

        statistics_file = open(self._local_statistics_path, 'a+')

        statistics_reporter = app.StateDataReporter(statistics_file, 0,
                                                    step=True,
                                                    potentialEnergy=True,
                                                    kineticEnergy=True,
                                                    totalEnergy=True,
                                                    temperature=True,
                                                    volume=True,
                                                    density=True)

        # Create the object which will transfer simulation output to the
        # reporters.
        topology = app.PDBFile(self.input_coordinate_file).topology

        with open(self.system_path, 'r') as file:
            system = openmm.XmlSerializer.deserialize(file.read())

        simulation = self._Simulation(integrator, topology, system, current_step)

        # Perform the simulation.
        checkpoint_counter = 0

        try:

            while current_step < total_number_of_steps:

                steps_to_take = min(self.output_frequency, total_number_of_steps - current_step)
                integrator.step(steps_to_take)

                current_step += steps_to_take

                state = context.getState(getPositions=True,
                                         getEnergy=True,
                                         getVelocities=False,
                                         getForces=False,
                                         getParameters=False,
                                         enforcePeriodicBox=self.enable_pbc)

                simulation.currentStep = current_step

                # Write out the current state using the reporters.
                dcd_reporter.report(simulation, state)
                statistics_reporter.report(simulation, state)

                if checkpoint_counter >= self.checkpoint_frequency:
                    # Save to the checkpoint file if needed.
                    self._write_checkpoint_file(current_step, context)
                    checkpoint_counter = 0

                checkpoint_counter += 1

        except Exception as e:

            formatted_exception = f'{traceback.format_exception(None, e, e.__traceback__)}'

            return PropertyEstimatorException(directory=directory,
                                              message=f'The simulation failed unexpectedly: '
                                                      f'{formatted_exception}')

        # Save out the final positions.
        self._write_checkpoint_file(current_step, context)

        final_state = context.getState(getPositions=True)

        positions = final_state.getPositions()
        topology.setPeriodicBoxVectors(final_state.getPeriodicBoxVectors())

        self.output_coordinate_file = os.path.join(directory, 'output.pdb')

        with open(self.output_coordinate_file, 'w+') as configuration_file:
            app.PDBFile.writeFile(topology, positions, configuration_file)

        logging.info(f'Simulation performed in the {str(self.ensemble)} ensemble: {self._id}')
        return None


@register_calculation_protocol()
class OpenMMParallelTempering(BaseOpenMMSimulation):
    """Performs a parallel tempering simulation using the utilities
    provided by the `openmmtools <https://openmmtools.readthedocs.io>`_ package.
    """

    output_frequency = protocol_input(
        docstring='The frequency (in number of iterations) with which to write to the '
                  'output statistics and trajectory files.',
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=1
    )

    replica_temperatures = protocol_input(
        docstring='The intermediate temperatures that a replica should sample at. '
                  'These values must be greater than the temperature defined by the '
                  'thermodynamic state, and list than the maximum temperature. If '
                  'unset, then these will be chosen as a geometric progression between '
                  'the defined start and end temperatures.',
        type_hint=list,
        default_value=UNDEFINED,
        optional=True
    )

    number_of_replicas = protocol_input(
        docstring='The number of exponentially-spaced temperatures between the '
                  'temperature of interest and the maximum temperature.',
        type_hint=int,
        default_value=UNDEFINED
    )

    maximum_temperature = protocol_input(
        docstring='The maximum temperature that a replica should sample at.',
        type_hint=unit.Quantity,
        default_value=UNDEFINED
    )

    def execute(self, directory, available_resources):

        from openmmtools import cache, mcmc, states
        from openmmtools.multistate import MultiStateReporter, ParallelTemperingSampler

        # Make sure the temperature range makes sense.
        if self.maximum_temperature < self.thermodynamic_state.temperature:

            return PropertyEstimatorException(directory, 'The maximum temperature cannot be lower than '
                                                         'the temperature of interest defined by the '
                                                         '`thermodynamic_state` input.')

        openmm_temperature = pint_quantity_to_openmm(self.thermodynamic_state.temperature)
        openmm_max_temperature = pint_quantity_to_openmm(self.maximum_temperature)

        openmm_pressure = None

        if self.thermodynamic_state.pressure is not None:
            openmm_pressure = pint_quantity_to_openmm(self.thermodynamic_state.pressure)

        # Determine the temperatures to simulate at.
        temperatures = None

        if self.replica_temperatures != UNDEFINED:

            if len(self.replica_temperatures) == 0:

                return PropertyEstimatorException(directory, 'At least one temperature must be defined in the '
                                                             '`replica_temperatures` list.')

            if not all(isinstance(x, unit.Quantity) for x in self.replica_temperatures):

                return PropertyEstimatorException(directory, 'The replica temperatures must be of type '
                                                             '`unit.Quantity` and be compatible with units '
                                                             'of kelvin.')

            sorted_temperatures = ([self.thermodynamic_state.temperature] +
                                   list(sorted(self.replica_temperatures)) +
                                   [self.maximum_temperature])

            if (sorted_temperatures[0] < self.thermodynamic_state.temperature or
                sorted_temperatures[0] > self.maximum_temperature or
                sorted_temperatures[-1] < self.thermodynamic_state.temperature or
                sorted_temperatures[-1] > self.maximum_temperature):

                return PropertyEstimatorException(directory, 'The replica temperatures are outside of the allowed '
                                                             'range.')

            temperatures = [temperature.to(unit.kelvin).magnitude for
                            temperature in sorted_temperatures] * simtk_unit.kelvin

        elif self.number_of_replicas == UNDEFINED:

            return PropertyEstimatorException(directory, 'The `number_of_replicas` must be set if '
                                                         '`replica_temperatures` is UNDEFINED.')

        # Load in the system object.
        with open(self.system_path, 'r') as file:
            system = openmm.XmlSerializer.deserialize(file.read())

        if not self.enable_pbc:

            disable_pbc(system)
            openmm_pressure = None

        # Create a platform with the correct resources.
        if not self.allow_gpu_platforms:

            from propertyestimator.backends import ComputeResources
            available_resources = ComputeResources(available_resources.number_of_threads)

        platform = setup_platform_with_resources(available_resources, self.high_precision)
        context_cache = cache.ContextCache(platform=platform)

        # Set up the thermodynamic states to sample.
        reference_state = states.ThermodynamicState(system, openmm_temperature, openmm_pressure)

        initial_pdb_file = app.PDBFile(self.input_coordinate_file)
        sampler_state = [states.SamplerState(positions=initial_pdb_file.positions)]

        # Propagate the replicas with Langevin dynamics.
        langevin_move = mcmc.GHMCMove(
            timestep=pint_quantity_to_openmm(self.timestep),
            collision_rate=pint_quantity_to_openmm(self.thermostat_friction),
            n_steps=self.steps_per_iteration,
            context_cache=context_cache
        )

        # Run the parallel tempering simulation.
        parallel_tempering = ParallelTemperingSampler(mcmc_moves=langevin_move,
                                                      number_of_iterations=self.total_number_of_iterations)

        storage_path = os.path.join(directory, 'replicas.nc')
        reporter = MultiStateReporter(storage_path, checkpoint_interval=self.output_frequency)

        if temperatures is None:

            parallel_tempering.create(reference_state,
                                      sampler_state,
                                      reporter,
                                      min_temperature=openmm_temperature,
                                      max_temperature=openmm_max_temperature,
                                      n_temperatures=self.number_of_replicas)

        else:

            parallel_tempering.create(reference_state,
                                      sampler_state,
                                      reporter,
                                      temperatures=temperatures,
                                      n_temperatures=len(temperatures))

        parallel_tempering.run()

        mdtraj_trajectory, statistics = self._extract_trajectory_statistics(os.path.join(directory, 'replicas.nc'),
                                                                            system,
                                                                            replica_index=0)

        self.statistics_file_path = os.path.join(directory, 'statistics.csv')
        self.trajectory_file_path = os.path.join(directory, 'trajectory.dcd')

        mdtraj_trajectory.save_dcd(self.trajectory_file_path)
        statistics.to_pandas_csv(self.statistics_file_path)

        return self._get_output_dictionary()

    def _extract_trajectory_statistics(self, nc_path, reference_system, replica_index):
        """Extract the trajectory and statistics of a replica from the NetCDF4 file.

        Parameters
        ----------
        nc_path : str
            Path to the primary nc_file storing the analysis options
        reference_system: simtk.openmm.System
            The system object describing the system which was simulated.
        replica_index : int, optional
            The index of the replica for which to extract the trajectory. One and
            only one between state_index and replica_index must be not None (default
            is None).

        Returns
        -------
        mdtraj.Trajectory
            The trajectory extracted from the netcdf file.
        StatisticsArray
            The statistics extracted from the netcdf file.

        Notes
        -----
        This method is mainly repurposed from the `yank.analyze.extract_trajectory`
        method. It should be remove if the same functionality is moved into
        `openmmtools`.
        """

        import mdtraj
        from openmmtools import multistate

        # Check correct input
        if not os.path.isfile(nc_path):
            raise ValueError('Cannot find file {}'.format(nc_path))

        # Import simulation data
        topology = mdtraj.load_topology(self.input_coordinate_file)

        reporter = None

        try:

            reporter = multistate.MultiStateReporter(nc_path, open_mode='r')

            all_reduced_potentials, _, _ = reporter.read_energies()
            replica_reduced_potentials = all_reduced_potentials[:, 0, 0]

            statistics_array = StatisticsArray()
            statistics_array[ObservableType.ReducedPotential] = replica_reduced_potentials * unit.dimensionless
            statistics_array[ObservableType.PotentialEnergy] = (replica_reduced_potentials *
                                                                self.thermodynamic_state.inverse_beta)

            # Determine if system is periodic
            is_periodic = reference_system.usesPeriodicBoundaryConditions()

            # Assume full iteration until proven otherwise
            reporter_storage = reporter._storage_checkpoint

            n_frames = reporter_storage.variables['positions'].shape[0]
            n_atoms = reporter_storage.variables['positions'].shape[2]

            # Determine the number of frames that the trajectory will have.
            frame_indices = range(0, n_frames)
            n_trajectory_frames = len(frame_indices)

            # Initialize positions and box vectors arrays. MDTraj Cython code
            # expects float32 positions.
            positions = np.zeros((n_trajectory_frames, n_atoms, 3), dtype=np.float32)
            box_vectors = None

            if is_periodic:
                box_vectors = np.zeros((n_trajectory_frames, 3, 3), dtype=np.float32)

            # Extract state positions and box vectors.
            for i, iteration in enumerate(frame_indices):

                positions[i, :, :] = reporter_storage.variables['positions'][
                                     iteration, replica_index, :, :].astype(np.float32)

                if box_vectors is not None:

                    box_vectors[i, :, :] = reporter_storage.variables['box_vectors'][
                                           iteration, replica_index, :, :].astype(np.float32)

        finally:

            if reporter is not None:
                reporter.close()

        # Create trajectory object
        trajectory = mdtraj.Trajectory(positions, topology)

        if is_periodic:
            trajectory.unitcell_vectors = box_vectors

        return trajectory, statistics_array
