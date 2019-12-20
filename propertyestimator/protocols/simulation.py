"""
A collection of protocols for running molecular simulations.
"""
import io
import json
import logging
import os
import re

import pandas as pd
from simtk import openmm
from simtk import unit as simtk_unit
from simtk.openmm import app

from propertyestimator import unit
from propertyestimator.attributes import UNDEFINED
from propertyestimator.thermodynamics import Ensemble, ThermodynamicState
from propertyestimator.utils.openmm import (
    disable_pbc,
    pint_quantity_to_openmm,
    setup_platform_with_resources,
)
from propertyestimator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder
from propertyestimator.utils.statistics import ObservableType, StatisticsArray
from propertyestimator.workflow.attributes import (
    InequalityMergeBehaviour,
    InputAttribute,
    OutputAttribute,
)
from propertyestimator.workflow.plugins import workflow_protocol
from propertyestimator.workflow.protocols import WorkflowProtocol


@workflow_protocol()
class RunEnergyMinimisation(WorkflowProtocol):
    """A protocol to minimise the potential energy of a system.
    """

    input_coordinate_file = InputAttribute(
        docstring="The coordinates to minimise.", type_hint=str, default_value=UNDEFINED
    )
    system_path = InputAttribute(
        docstring="The path to the XML system object which defines the forces present "
        "in the system.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    tolerance = InputAttribute(
        docstring="The energy tolerance to which the system should be minimized.",
        type_hint=unit.Quantity,
        default_value=10 * unit.kilojoules / unit.mole,
    )
    max_iterations = InputAttribute(
        docstring="The maximum number of iterations to perform. If this is 0, "
        "minimization is continued until the results converge without regard to "
        "how many iterations it takes.",
        type_hint=int,
        default_value=0,
    )

    enable_pbc = InputAttribute(
        docstring="If true, periodic boundary conditions will be enabled.",
        type_hint=bool,
        default_value=True,
    )

    output_coordinate_file = OutputAttribute(
        docstring="The file path to the minimised coordinates.", type_hint=str
    )

    def execute(self, directory, available_resources):

        platform = setup_platform_with_resources(available_resources)

        input_pdb_file = app.PDBFile(self.input_coordinate_file)

        with open(self.system_path, "rb") as file:
            system = openmm.XmlSerializer.deserialize(file.read().decode())

        if not self.enable_pbc:

            for force_index in range(system.getNumForces()):

                force = system.getForce(force_index)

                if not isinstance(force, openmm.NonbondedForce):
                    continue

                force.setNonbondedMethod(
                    0
                )  # NoCutoff = 0, NonbondedMethod.CutoffNonPeriodic = 1

        # TODO: Expose the constraint tolerance
        integrator = openmm.VerletIntegrator(0.002 * simtk_unit.picoseconds)
        simulation = app.Simulation(
            input_pdb_file.topology, system, integrator, platform
        )

        box_vectors = input_pdb_file.topology.getPeriodicBoxVectors()

        if box_vectors is None:
            box_vectors = simulation.system.getDefaultPeriodicBoxVectors()

        simulation.context.setPeriodicBoxVectors(*box_vectors)
        simulation.context.setPositions(input_pdb_file.positions)

        simulation.minimizeEnergy(
            pint_quantity_to_openmm(self.tolerance), self.max_iterations
        )

        positions = simulation.context.getState(getPositions=True).getPositions()

        self.output_coordinate_file = os.path.join(directory, "minimised.pdb")

        with open(self.output_coordinate_file, "w+") as minimised_file:
            app.PDBFile.writeFile(simulation.topology, positions, minimised_file)

        return self._get_output_dictionary()


@workflow_protocol()
class RunOpenMMSimulation(WorkflowProtocol):
    """Performs a molecular dynamics simulation in a given ensemble using
    an OpenMM backend.
    """

    class _Checkpoint:
        """A temporary checkpoint file which keeps track
        of the parts of the simulation state not stored in
        the checkpoint state xml file.
        """

        def __init__(
            self,
            output_frequency=-1,
            checkpoint_frequency=-1,
            steps_per_iteration=-1,
            current_step_number=0,
        ):

            self.output_frequency = output_frequency
            self.checkpoint_frequency = checkpoint_frequency
            self.steps_per_iteration = steps_per_iteration
            self.current_step_number = current_step_number

        def __getstate__(self):
            return {
                "output_frequency": self.output_frequency,
                "checkpoint_frequency": self.checkpoint_frequency,
                "steps_per_iteration": self.steps_per_iteration,
                "current_step_number": self.current_step_number,
            }

        def __setstate__(self, state):
            self.output_frequency = state["output_frequency"]
            self.checkpoint_frequency = state["checkpoint_frequency"]
            self.steps_per_iteration = state["steps_per_iteration"]
            self.current_step_number = state["current_step_number"]

    class _Simulation:
        """A fake simulation class to use with the
        openmm file reporters.
        """

        def __init__(self, integrator, topology, system, current_step):
            self.integrator = integrator
            self.topology = topology
            self.system = system
            self.currentStep = current_step

    steps_per_iteration = InputAttribute(
        docstring="The number of steps to propogate the system by at "
        "each iteration. The total number of steps performed "
        "by this protocol will be `total_number_of_iterations * "
        "steps_per_iteration`.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=1000000,
    )
    total_number_of_iterations = InputAttribute(
        docstring="The number of times to propogate the system forward by the "
        "`steps_per_iteration` number of steps. The total number of "
        "steps performed by this protocol will be `total_number_of_iterations * "
        "steps_per_iteration`.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=1,
    )

    output_frequency = InputAttribute(
        docstring="The frequency (in number of steps) with which to write to the "
        "output statistics and trajectory files.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=3000,
    )
    checkpoint_frequency = InputAttribute(
        docstring="The frequency (in multiples of `output_frequency`) with which to "
        "write to a checkpoint file, e.g. if `output_frequency=100` and "
        "`checkpoint_frequency==2`, a checkpoint file would be saved every "
        "200 steps.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        optional=True,
        default_value=10,
    )

    timestep = InputAttribute(
        docstring="The timestep to evolve the system by at each step.",
        type_hint=unit.Quantity,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=2.0 * unit.femtosecond,
    )

    thermodynamic_state = InputAttribute(
        docstring="The thermodynamic conditions to simulate under",
        type_hint=ThermodynamicState,
        default_value=UNDEFINED,
    )
    ensemble = InputAttribute(
        docstring="The thermodynamic ensemble to simulate in.",
        type_hint=Ensemble,
        default_value=Ensemble.NPT,
    )

    thermostat_friction = InputAttribute(
        docstring="The thermostat friction coefficient.",
        type_hint=unit.Quantity,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=1.0 / unit.picoseconds,
    )

    input_coordinate_file = InputAttribute(
        docstring="The file path to the starting coordinates.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    system_path = InputAttribute(
        docstring="A path to the XML system object which defines the forces present "
        "in the system.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    enable_pbc = InputAttribute(
        docstring="If true, periodic boundary conditions will be enabled.",
        type_hint=bool,
        default_value=True,
    )

    allow_gpu_platforms = InputAttribute(
        docstring="If true, OpenMM will be allowed to run using a GPU if available, "
        "otherwise it will be constrained to only using CPUs.",
        type_hint=bool,
        default_value=True,
    )
    high_precision = InputAttribute(
        docstring="If true, OpenMM will be run using a platform with high precision "
        "settings. This will be the Reference platform when only a CPU is "
        "available, or double precision mode when a GPU is available.",
        type_hint=bool,
        default_value=False,
    )

    output_coordinate_file = OutputAttribute(
        docstring="The file path to the coordinates of the final system configuration.",
        type_hint=str,
    )
    trajectory_file_path = OutputAttribute(
        docstring="The file path to the trajectory sampled during the simulation.",
        type_hint=str,
    )
    statistics_file_path = OutputAttribute(
        docstring="The file path to the statistics sampled during the simulation.",
        type_hint=str,
    )

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._checkpoint_path = None
        self._state_path = None

        self._local_trajectory_path = None
        self._local_statistics_path = None

        self._context = None
        self._integrator = None

    def execute(self, directory, available_resources):

        # We handle most things in OMM units here.
        temperature = self.thermodynamic_state.temperature
        openmm_temperature = pint_quantity_to_openmm(temperature)

        pressure = (
            None if self.ensemble == Ensemble.NVT else self.thermodynamic_state.pressure
        )

        openmm_pressure = pint_quantity_to_openmm(pressure)

        if openmm_temperature is None:

            raise ValueError(
                "A temperature must be set to perform a simulation in any ensemble"
            )

        if Ensemble(self.ensemble) == Ensemble.NPT and openmm_pressure is None:
            raise ValueError("A pressure must be set to perform an NPT simulation")

        if Ensemble(self.ensemble) == Ensemble.NPT and self.enable_pbc is False:
            raise ValueError("PBC must be enabled when running in the NPT ensemble.")

        # Set up the internal file paths
        self._checkpoint_path = os.path.join(directory, "checkpoint.json")
        self._state_path = os.path.join(directory, "checkpoint_state.xml")

        self._local_trajectory_path = os.path.join(directory, "trajectory.dcd")
        self._local_statistics_path = os.path.join(directory, "openmm_statistics.csv")

        # Set up the simulation objects.
        if self._context is None or self._integrator is None:

            self._context, self._integrator = self._setup_simulation_objects(
                openmm_temperature, openmm_pressure, available_resources
            )

        # Save a copy of the starting configuration if it doesn't already exist
        local_input_coordinate_path = os.path.join(directory, "input.pdb")

        if not os.path.isfile(local_input_coordinate_path):

            input_pdb_file = app.PDBFile(self.input_coordinate_file)

            with open(local_input_coordinate_path, "w+") as configuration_file:
                app.PDBFile.writeFile(
                    input_pdb_file.topology,
                    input_pdb_file.positions,
                    configuration_file,
                )

        # Run the simulation.
        self._simulate(directory, self._context, self._integrator)

        # Set the output paths.
        self.trajectory_file_path = self._local_trajectory_path
        self.statistics_file_path = os.path.join(directory, "statistics.csv")

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

            available_resources = ComputeResources(
                available_resources.number_of_threads
            )

        platform = setup_platform_with_resources(
            available_resources, self.high_precision
        )

        # Load in the system object from the provided xml file.
        with open(self.system_path, "r") as file:
            system = XmlSerializer.deserialize(file.read())

        # Disable the periodic boundary conditions if requested.
        if not self.enable_pbc:

            disable_pbc(system)
            pressure = None

        # Use the openmmtools ThermodynamicState object to help
        # set up a system which contains the correct barostat if
        # one should be present.
        openmm_state = openmmtools.states.ThermodynamicState(
            system=system, temperature=temperature, pressure=pressure
        )

        system = openmm_state.get_system(remove_thermostat=True)

        # Set up the integrator.
        thermostat_friction = pint_quantity_to_openmm(self.thermostat_friction)
        timestep = pint_quantity_to_openmm(self.timestep)

        integrator = openmmtools.integrators.LangevinIntegrator(
            temperature=temperature,
            collision_rate=thermostat_friction,
            timestep=timestep,
        )

        # Create the simulation context.
        context = openmm.Context(system, integrator, platform)

        # Initialize the context with the correct positions etc.
        input_pdb_file = app.PDBFile(self.input_coordinate_file)

        if self.enable_pbc:

            # Optionally set up the box vectors.
            box_vectors = input_pdb_file.topology.getPeriodicBoxVectors()

            if box_vectors is None:

                raise ValueError(
                    "The input file must contain box vectors when running with PBC."
                )

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
        state = context.getState(
            getPositions=True,
            getEnergy=True,
            getVelocities=True,
            getForces=True,
            getParameters=True,
            enforcePeriodicBox=self.enable_pbc,
        )

        with open(self._state_path, "w") as file:
            file.write(openmm.XmlSerializer.serialize(state))

        checkpoint = self._Checkpoint(
            self.output_frequency,
            self.checkpoint_frequency,
            self.steps_per_iteration,
            current_step_number,
        )

        with open(self._checkpoint_path, "w") as file:
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
            file_contents = re.sub("#.*\n", "", file.read())

            with io.StringIO(file_contents) as string_object:
                existing_statistics_array = pd.read_csv(
                    string_object, index_col=False, header=None
                )

        statistics_length = len(existing_statistics_array)

        if statistics_length < number_of_frames:

            raise ValueError(
                f"The saved number of statistics frames ({statistics_length}) "
                f"is less than expected ({number_of_frames})."
            )

        elif statistics_length == number_of_frames:
            return

        truncated_statistics_array = existing_statistics_array[0:number_of_frames]

        with open(self._local_statistics_path, "w") as file:

            file.write(f"{header_line}")
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
        # noinspection PyProtectedMember
        base_distance_unit = mdtraj.Trajectory._distance_unit

        # Get an accurate measurement of the length of the trajectory
        # without reading it into memory.
        trajectory_length = 0

        for chunk in mdtraj.iterload(self._local_trajectory_path, top=topology):
            trajectory_length += len(chunk)

        # Make sure there is at least the expected number of frames.
        if trajectory_length < number_of_frames:

            raise ValueError(
                f"The saved number of trajectory frames ({trajectory_length}) "
                f"is less than expected ({number_of_frames})."
            )

        elif trajectory_length == number_of_frames:
            return

        # Truncate the trajectory by streaming one frame of the trajectory at
        # a time.
        temporary_trajectory_path = f"{self._local_trajectory_path}.tmp"

        with DCDTrajectoryFile(self._local_trajectory_path, "r") as input_file:

            with DCDTrajectoryFile(temporary_trajectory_path, "w") as output_file:

                for frame_index in range(0, number_of_frames):

                    frame = input_file.read_as_traj(topology, n_frames=1, stride=1)

                    output_file.write(
                        xyz=in_units_of(
                            frame.xyz, base_distance_unit, output_file.distance_unit
                        ),
                        cell_lengths=in_units_of(
                            frame.unitcell_lengths,
                            base_distance_unit,
                            output_file.distance_unit,
                        ),
                        cell_angles=frame.unitcell_angles[0],
                    )

        os.replace(temporary_trajectory_path, self._local_trajectory_path)

        # Do a sanity check to make sure the trajectory was successfully truncated.
        new_trajectory_length = 0

        for chunk in mdtraj.iterload(
            self._local_trajectory_path, top=self.input_coordinate_file
        ):
            new_trajectory_length += len(chunk)

        if new_trajectory_length != number_of_frames:
            raise ValueError("The trajectory was incorrectly truncated.")

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
        if not os.path.isfile(self._checkpoint_path) or not os.path.isfile(
            self._state_path
        ):

            logging.info("No checkpoint files were found.")
            return current_step_number

        if not os.path.isfile(self._local_statistics_path) or not os.path.isfile(
            self._local_trajectory_path
        ):

            raise ValueError(
                "Checkpoint files were correctly found, but the trajectory "
                "or statistics files seem to be missing. This should not happen."
            )

        logging.info("Restoring the system state from checkpoint files.")

        # If they do, load the current state from disk.
        with open(self._state_path, "r") as file:
            current_state = openmm.XmlSerializer.deserialize(file.read())

        with open(self._checkpoint_path, "r") as file:
            checkpoint = json.load(file, cls=TypedJSONDecoder)

        if (
            self.output_frequency != checkpoint.output_frequency
            or self.checkpoint_frequency != checkpoint.checkpoint_frequency
            or self.steps_per_iteration != checkpoint.steps_per_iteration
        ):

            raise ValueError(
                "Neither the output frequency, the checkpoint "
                "frequency, nor the steps per iteration can "
                "currently be changed during the course of the "
                "simulation. Only the number of iterations is "
                "allowed to change."
            )

        # Make sure this simulation hasn't already finished.
        total_expected_number_of_steps = (
            self.total_number_of_iterations * self.steps_per_iteration
        )

        if checkpoint.current_step_number == total_expected_number_of_steps:
            return checkpoint.current_step_number

        context.setState(current_state)

        # Make sure that the number of frames in the trajectory /
        # statistics file correspond to the recorded number of steps.
        # This is to handle possible cases where only some of the files
        # have been written from the current step (i.e only the trajectory may
        # have been written to before this protocol gets unexpectedly killed.
        expected_number_of_frames = int(
            checkpoint.current_step_number / self.output_frequency
        )

        # Handle the truncation of the statistics file.
        self._truncate_statistics_file(expected_number_of_frames)

        # Handle the truncation of the trajectory file.
        self._truncate_trajectory_file(expected_number_of_frames)

        logging.info("System state restored from checkpoint files.")

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
        statistics = StatisticsArray.from_openmm_csv(
            self._local_statistics_path, pressure
        )

        reduced_potentials = (
            statistics[ObservableType.PotentialEnergy] / unit.avogadro_number
        )

        if pressure is not None:

            pv_terms = pressure * statistics[ObservableType.Volume]
            reduced_potentials += pv_terms

        beta = 1.0 / (unit.boltzmann_constant * temperature)
        statistics[ObservableType.ReducedPotential] = (beta * reduced_potentials).to(
            unit.dimensionless
        )

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
        total_number_of_steps = (
            self.total_number_of_iterations * self.steps_per_iteration
        )

        # Try to load the current state from any available checkpoint information
        current_step = self._resume_from_checkpoint(context)

        if current_step == total_number_of_steps:
            return

        # Build the reporters which we will use to report the state
        # of the simulation.
        append_trajectory = os.path.isfile(self._local_trajectory_path)
        dcd_reporter = app.DCDReporter(
            self._local_trajectory_path, 0, append_trajectory
        )

        statistics_file = open(self._local_statistics_path, "a+")

        statistics_reporter = app.StateDataReporter(
            statistics_file,
            0,
            step=True,
            potentialEnergy=True,
            kineticEnergy=True,
            totalEnergy=True,
            temperature=True,
            volume=True,
            density=True,
        )

        # Create the object which will transfer simulation output to the
        # reporters.
        topology = app.PDBFile(self.input_coordinate_file).topology

        with open(self.system_path, "r") as file:
            system = openmm.XmlSerializer.deserialize(file.read())

        simulation = self._Simulation(integrator, topology, system, current_step)

        # Perform the simulation.
        checkpoint_counter = 0

        while current_step < total_number_of_steps:

            steps_to_take = min(
                self.output_frequency, total_number_of_steps - current_step
            )
            integrator.step(steps_to_take)

            current_step += steps_to_take

            state = context.getState(
                getPositions=True,
                getEnergy=True,
                getVelocities=False,
                getForces=False,
                getParameters=False,
                enforcePeriodicBox=self.enable_pbc,
            )

            simulation.currentStep = current_step

            # Write out the current state using the reporters.
            dcd_reporter.report(simulation, state)
            statistics_reporter.report(simulation, state)

            if checkpoint_counter >= self.checkpoint_frequency:
                # Save to the checkpoint file if needed.
                self._write_checkpoint_file(current_step, context)
                checkpoint_counter = 0

            checkpoint_counter += 1

        # Save out the final positions.
        self._write_checkpoint_file(current_step, context)

        final_state = context.getState(getPositions=True)

        positions = final_state.getPositions()
        topology.setPeriodicBoxVectors(final_state.getPeriodicBoxVectors())

        self.output_coordinate_file = os.path.join(directory, "output.pdb")

        with open(self.output_coordinate_file, "w+") as configuration_file:
            app.PDBFile.writeFile(topology, positions, configuration_file)
