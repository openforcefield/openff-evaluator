"""
A collection of protocols which employs OpenMM to evaluate and propagate the
state of molecular systems.
"""
import copy
import io
import json
import logging
import os
import re

import numpy as np
import pandas as pd
from simtk import openmm
from simtk import unit as simtk_unit
from simtk.openmm import app

from openff.evaluator import unit
from openff.evaluator.forcefield import ForceFieldSource, SmirnoffForceFieldSource
from openff.evaluator.protocols.gradients import BaseGradientPotentials
from openff.evaluator.protocols.reweighting import BaseReducedPotentials
from openff.evaluator.protocols.simulation import BaseEnergyMinimisation, BaseSimulation
from openff.evaluator.thermodynamics import Ensemble
from openff.evaluator.utils.openmm import (
    disable_pbc,
    openmm_quantity_to_pint,
    pint_quantity_to_openmm,
    setup_platform_with_resources,
)
from openff.evaluator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder
from openff.evaluator.utils.statistics import ObservableType, StatisticsArray
from openff.evaluator.utils.utils import is_file_and_not_empty
from openff.evaluator.workflow import workflow_protocol

logger = logging.getLogger(__name__)


@workflow_protocol()
class OpenMMEnergyMinimisation(BaseEnergyMinimisation):
    """A protocol to minimise the potential energy of a system using
    OpenMM.
    """

    def _execute(self, directory, available_resources):

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


@workflow_protocol()
class OpenMMSimulation(BaseSimulation):
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

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._checkpoint_path = None
        self._state_path = None

        self._local_trajectory_path = None
        self._local_statistics_path = None

        self._context = None
        self._integrator = None

    def _execute(self, directory, available_resources):

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

        if not is_file_and_not_empty(local_input_coordinate_path):

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

        # Save out the final statistics in the openff-evaluator format
        self._save_final_statistics(self.statistics_file_path, temperature, pressure)

    def _setup_simulation_objects(self, temperature, pressure, available_resources):
        """Initializes the objects needed to perform the simulation.
        This comprises of a context, and an integrator.

        Parameters
        ----------
        temperature: simtk.pint.Quantity
            The temperature to run the simulation at.
        pressure: simtk.pint.Quantity
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

            from openff.evaluator.backends import ComputeResources

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
        if not is_file_and_not_empty(
            self._checkpoint_path
        ) or not is_file_and_not_empty(self._state_path):

            logger.info("No checkpoint files were found.")
            return current_step_number

        if not is_file_and_not_empty(
            self._local_statistics_path
        ) or not is_file_and_not_empty(self._local_trajectory_path):

            raise ValueError(
                "Checkpoint files were correctly found, but the trajectory "
                "or statistics files seem to be missing. This should not happen."
            )

        logger.info("Restoring the system state from checkpoint files.")

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

        logger.info("System state restored from checkpoint files.")

        return checkpoint.current_step_number

    def _save_final_statistics(self, path, temperature, pressure):
        """Converts the openmm statistic csv file into an openff-evaluator
        StatisticsArray csv file, making sure to fill in any missing entries.

        Parameters
        ----------
        path: str
            The path to save the statistics to.
        temperature: pint.Quantity
            The temperature that the simulation is being run at.
        pressure: pint.Quantity
            The pressure that the simulation is being run at.
        """
        statistics = StatisticsArray.from_openmm_csv(
            self._local_statistics_path, pressure
        )

        reduced_potentials = (
            statistics[ObservableType.PotentialEnergy] / unit.avogadro_constant
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
        append_trajectory = is_file_and_not_empty(self._local_trajectory_path)
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
            speed=True,
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


@workflow_protocol()
class OpenMMReducedPotentials(BaseReducedPotentials):
    """Calculates the reduced potential for a given set of
    configurations using OpenMM.
    """

    def _execute(self, directory, available_resources):

        import mdtraj
        import openmmtools

        trajectory = mdtraj.load_dcd(
            self.trajectory_file_path, self.coordinate_file_path
        )

        with open(self.system_path, "r") as file:
            system = openmm.XmlSerializer.deserialize(file.read())

        temperature = pint_quantity_to_openmm(self.thermodynamic_state.temperature)
        pressure = pint_quantity_to_openmm(self.thermodynamic_state.pressure)

        if self.enable_pbc:
            system.setDefaultPeriodicBoxVectors(*trajectory.openmm_boxes(0))
        else:
            pressure = None

        openmm_state = openmmtools.states.ThermodynamicState(
            system=system, temperature=temperature, pressure=pressure
        )

        integrator = openmmtools.integrators.VelocityVerletIntegrator(
            0.01 * simtk_unit.femtoseconds
        )

        # Setup the requested platform:
        platform = setup_platform_with_resources(
            available_resources, self.high_precision
        )
        openmm_system = openmm_state.get_system(True, True)

        if not self.enable_pbc:
            disable_pbc(openmm_system)

        openmm_context = openmm.Context(openmm_system, integrator, platform)

        potential_energies = np.zeros(trajectory.n_frames)
        reduced_potentials = np.zeros(trajectory.n_frames)

        for frame_index in range(trajectory.n_frames):

            if self.enable_pbc:
                box_vectors = trajectory.openmm_boxes(frame_index)
                openmm_context.setPeriodicBoxVectors(*box_vectors)

            positions = trajectory.xyz[frame_index]
            openmm_context.setPositions(positions)

            potential_energy = openmm_context.getState(
                getEnergy=True
            ).getPotentialEnergy()

            potential_energies[frame_index] = potential_energy.value_in_unit(
                simtk_unit.kilojoule_per_mole
            )
            reduced_potentials[frame_index] = openmm_state.reduced_potential(
                openmm_context
            )

        kinetic_energies = StatisticsArray.from_pandas_csv(self.kinetic_energies_path)[
            ObservableType.KineticEnergy
        ]

        statistics_array = StatisticsArray()
        statistics_array[ObservableType.PotentialEnergy] = (
            potential_energies * unit.kilojoule / unit.mole
        )
        statistics_array[ObservableType.KineticEnergy] = kinetic_energies
        statistics_array[ObservableType.ReducedPotential] = (
            reduced_potentials * unit.dimensionless
        )

        statistics_array[ObservableType.TotalEnergy] = (
            statistics_array[ObservableType.PotentialEnergy]
            + statistics_array[ObservableType.KineticEnergy]
        )

        statistics_array[ObservableType.Enthalpy] = (
            statistics_array[ObservableType.ReducedPotential]
            * self.thermodynamic_state.inverse_beta
            + kinetic_energies
        )

        if self.use_internal_energy:
            statistics_array[ObservableType.ReducedPotential] += (
                kinetic_energies * self.thermodynamic_state.beta
            )

        self.statistics_file_path = os.path.join(directory, "statistics.csv")
        statistics_array.to_pandas_csv(self.statistics_file_path)


@workflow_protocol()
class OpenMMGradientPotentials(BaseGradientPotentials):
    """A protocol to estimates the the reduced potential of the configurations
    of a trajectory using reverse and forward perturbed simulation parameters for
    use with estimating reweighted gradients using the central difference method.
    """

    def _build_reduced_system(self, original_force_field, topology, scale_amount=None):
        """Produces an OpenMM system containing only forces for the specified parameter,
         optionally perturbed by the amount specified by `scale_amount`.

        Parameters
        ----------
        original_force_field: openforcefield.typing.engines.smirnoff.ForceField
            The force field to create the system from (and optionally perturb).
        topology: openforcefield.topology.Topology
            The topology of the system to apply the force field to.
        scale_amount: float, optional
            The optional amount to perturb the parameter by.

        Returns
        -------
        simtk.openmm.System
            The created system.
        simtk.pint.Quantity
            The new value of the perturbed parameter.
        """
        # As this method deals mainly with the toolkit, we stick to
        # simtk units here.
        from openforcefield.typing.engines.smirnoff import ForceField

        parameter_tag = self.parameter_key.tag
        parameter_smirks = self.parameter_key.smirks
        parameter_attribute = self.parameter_key.attribute

        original_handler = original_force_field.get_parameter_handler(parameter_tag)
        original_parameter = original_handler.parameters[parameter_smirks]

        if self.use_subset_of_force_field:

            force_field = ForceField()
            handler = copy.deepcopy(
                original_force_field.get_parameter_handler(parameter_tag)
            )
            force_field.register_parameter_handler(handler)

        else:

            force_field = copy.deepcopy(original_force_field)
            handler = force_field.get_parameter_handler(parameter_tag)

        parameter_index = None
        value_list = None

        if hasattr(original_parameter, parameter_attribute):
            parameter_value = getattr(original_parameter, parameter_attribute)
        else:
            attribute_split = re.split(r"(\d+)", parameter_attribute)

            assert len(parameter_attribute) == 2
            assert hasattr(original_parameter, attribute_split[0])

            parameter_attribute = attribute_split[0]
            parameter_index = int(attribute_split[1]) - 1

            value_list = getattr(original_parameter, parameter_attribute)
            parameter_value = value_list[parameter_index]

        if scale_amount is not None:

            existing_parameter = handler.parameters[parameter_smirks]

            if np.isclose(parameter_value.value_in_unit(parameter_value.unit), 0.0):
                # Careful thought needs to be given to this. Consider cases such as
                # epsilon or sigma where negative values are not allowed.
                parameter_value = (
                    scale_amount if scale_amount > 0.0 else 0.0
                ) * parameter_value.unit
            else:
                parameter_value *= 1.0 + scale_amount

            if value_list is None:
                setattr(existing_parameter, parameter_attribute, parameter_value)
            else:
                value_list[parameter_index] = parameter_value
                setattr(existing_parameter, parameter_attribute, value_list)

        system = force_field.create_openmm_system(topology)

        if not self.enable_pbc:
            disable_pbc(system)

        return system, parameter_value

    def _evaluate_reduced_potential(
        self,
        system,
        trajectory,
        file_path,
        compute_resources,
        subset_energy_corrections=None,
    ):
        """Computes the reduced potential of each frame in a trajectory
        using the provided system.

        Parameters
        ----------
        system: simtk.openmm.System
            The system which encodes the interaction forces for the
            specified parameter.
        trajectory: mdtraj.Trajectory
            A trajectory of configurations to evaluate.
        file_path: str
            The path to save the reduced potentials to.
        compute_resources: ComputeResources
            The compute resources available to execute on.
        subset_energy_corrections: pint.Quantity, optional
            A pint.Quantity wrapped numpy.ndarray which contains a set
            of energies to add to the re-evaluated potential energies.
            This is mainly used to correct the potential energies evaluated
            using a subset of the force field back to energies as if evaluated
            using the full thing.

        Returns
        ---------
        StatisticsArray
            The array containing the reduced potentials.
        """
        integrator = openmm.VerletIntegrator(0.1 * simtk_unit.femtoseconds)

        platform = setup_platform_with_resources(compute_resources, True)
        openmm_context = openmm.Context(system, integrator, platform)

        potentials = np.zeros(trajectory.n_frames, dtype=np.float64)
        reduced_potentials = np.zeros(trajectory.n_frames, dtype=np.float64)

        temperature = pint_quantity_to_openmm(self.thermodynamic_state.temperature)
        beta = 1.0 / (simtk_unit.BOLTZMANN_CONSTANT_kB * temperature)

        pressure = pint_quantity_to_openmm(self.thermodynamic_state.pressure)

        if subset_energy_corrections is None:
            subset_energy_corrections = (
                np.zeros(trajectory.n_frames, dtype=np.float64)
                * simtk_unit.kilojoules_per_mole
            )
        else:
            subset_energy_corrections = pint_quantity_to_openmm(
                subset_energy_corrections
            )

        for frame_index in range(trajectory.n_frames):

            positions = trajectory.xyz[frame_index]
            box_vectors = trajectory.openmm_boxes(frame_index)

            if self.enable_pbc:
                openmm_context.setPeriodicBoxVectors(*box_vectors)

            openmm_context.setPositions(positions)

            state = openmm_context.getState(getEnergy=True)

            potential_energy = (
                state.getPotentialEnergy() + subset_energy_corrections[frame_index]
            )
            unreduced_potential = potential_energy / simtk_unit.AVOGADRO_CONSTANT_NA

            if pressure is not None and self.enable_pbc:
                unreduced_potential += pressure * state.getPeriodicBoxVolume()

            potentials[frame_index] = potential_energy.value_in_unit(
                simtk_unit.kilojoule_per_mole
            )
            reduced_potentials[frame_index] = unreduced_potential * beta

        potentials *= unit.kilojoule / unit.mole
        reduced_potentials *= unit.dimensionless

        statistics_array = StatisticsArray()
        statistics_array[ObservableType.ReducedPotential] = reduced_potentials
        statistics_array[ObservableType.PotentialEnergy] = potentials
        statistics_array.to_pandas_csv(file_path)

        return statistics_array

    def _execute(self, directory, available_resources):

        import mdtraj
        from openforcefield.topology import Molecule, Topology

        with open(self.force_field_path) as file:
            force_field_source = ForceFieldSource.parse_json(file.read())

        if not isinstance(force_field_source, SmirnoffForceFieldSource):

            raise ValueError(
                "Only SMIRNOFF force fields are supported by this protocol.",
            )

        # Load in the inputs
        force_field = force_field_source.to_force_field()

        trajectory = mdtraj.load_dcd(
            self.trajectory_file_path, self.coordinate_file_path
        )

        unique_molecules = []

        for component in self.substance.components:

            molecule = Molecule.from_smiles(smiles=component.smiles)
            unique_molecules.append(molecule)

        pdb_file = app.PDBFile(self.coordinate_file_path)
        topology = Topology.from_openmm(
            pdb_file.topology, unique_molecules=unique_molecules
        )

        # Compute the difference between the energies using the reduced force field,
        # and the full force field.
        energy_corrections = None

        if self.use_subset_of_force_field:

            target_system, _ = self._build_reduced_system(force_field, topology)

            subset_potentials_path = os.path.join(directory, "subset.csv")
            subset_potentials = self._evaluate_reduced_potential(
                target_system, trajectory, subset_potentials_path, available_resources
            )

            full_statistics = StatisticsArray.from_pandas_csv(self.statistics_path)

            energy_corrections = (
                full_statistics[ObservableType.PotentialEnergy]
                - subset_potentials[ObservableType.PotentialEnergy]
            )

        # Build the slightly perturbed system.
        reverse_system, reverse_parameter_value = self._build_reduced_system(
            force_field, topology, -self.perturbation_scale
        )

        forward_system, forward_parameter_value = self._build_reduced_system(
            force_field, topology, self.perturbation_scale
        )

        self.reverse_parameter_value = openmm_quantity_to_pint(reverse_parameter_value)
        self.forward_parameter_value = openmm_quantity_to_pint(forward_parameter_value)

        # Calculate the reduced potentials.
        self.reverse_potentials_path = os.path.join(directory, "reverse.csv")
        self.forward_potentials_path = os.path.join(directory, "forward.csv")

        self._evaluate_reduced_potential(
            reverse_system,
            trajectory,
            self.reverse_potentials_path,
            available_resources,
            energy_corrections,
        )
        self._evaluate_reduced_potential(
            forward_system,
            trajectory,
            self.forward_potentials_path,
            available_resources,
            energy_corrections,
        )
