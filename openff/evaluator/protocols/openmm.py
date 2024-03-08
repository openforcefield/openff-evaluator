"""
A collection of protocols which employs OpenMM to evaluate and propagate the
state of molecular systems.
"""
import io
import json
import logging
import os
import re
from collections import defaultdict
from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd
from openff.units import unit
from openff.units.openmm import from_openmm, to_openmm

from openff.evaluator.backends import ComputeResources
from openff.evaluator.forcefield import (
    ParameterGradient,
    ParameterGradientKey,
    SmirnoffForceFieldSource,
)
from openff.evaluator.protocols.reweighting import BaseEvaluateEnergies
from openff.evaluator.protocols.simulation import BaseEnergyMinimisation, BaseSimulation
from openff.evaluator.thermodynamics import Ensemble, ThermodynamicState
from openff.evaluator.utils.observables import (
    ObservableArray,
    ObservableFrame,
    ObservableType,
)
from openff.evaluator.utils.openmm import (
    disable_pbc,
    extract_atom_indices,
    extract_positions,
    setup_platform_with_resources,
    system_subset,
    update_context_with_pdb,
    update_context_with_positions,
)
from openff.evaluator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder
from openff.evaluator.utils.utils import is_file_and_not_empty
from openff.evaluator.workflow import workflow_protocol

if TYPE_CHECKING:
    import openmm
    from mdtraj import Trajectory
    from openff.toolkit.topology import Topology
    from openff.toolkit.typing.engines.smirnoff import ForceField

logger = logging.getLogger(__name__)


def _evaluate_energies(
    thermodynamic_state: ThermodynamicState,
    system: "openmm.System",
    trajectory: "Trajectory",
    compute_resources: ComputeResources,
    enable_pbc: bool = True,
    high_precision: bool = True,
) -> ObservableFrame:
    """Evaluates the reduced and potential energies of each frame in a trajectory
    using the specified system and at a particular state.

    Parameters
    ----------
    thermodynamic_state
        The thermodynamic state to evaluate the reduced potentials at.
    system
        The system to evaluate the energies using.
    trajectory
        A trajectory of configurations to evaluate.
    compute_resources: ComputeResources
        The compute resources available to execute on.
    enable_pbc
        Whether PBC are enabled. This controls whether box vectors
        are set or not.
    high_precision
        Whether to compute the energies using double precision.

    Returns
    -------
        The array containing the evaluated potentials.
    """
    import openmm
    import openmm.unit

    integrator = openmm.VerletIntegrator(0.1 * openmm.unit.femtoseconds)

    platform = setup_platform_with_resources(compute_resources, high_precision)
    openmm_context = openmm.Context(system, integrator, platform)

    potentials = np.zeros(trajectory.n_frames, dtype=np.float64)
    reduced_potentials = np.zeros(trajectory.n_frames, dtype=np.float64)

    temperature = to_openmm(thermodynamic_state.temperature)
    beta = 1.0 / (openmm.unit.BOLTZMANN_CONSTANT_kB * temperature)

    if thermodynamic_state.pressure is None:
        pressure = None
    else:
        pressure = to_openmm(thermodynamic_state.pressure)

    for frame_index in range(trajectory.n_frames):
        positions = trajectory.xyz[frame_index]
        box_vectors = None

        if enable_pbc:
            box_vectors = trajectory.openmm_boxes(frame_index)

        update_context_with_positions(openmm_context, positions, box_vectors)

        state = openmm_context.getState(getEnergy=True)

        potential_energy = state.getPotentialEnergy()
        unreduced_potential = potential_energy / openmm.unit.AVOGADRO_CONSTANT_NA

        if pressure is not None and enable_pbc:
            unreduced_potential += pressure * state.getPeriodicBoxVolume()

        potentials[frame_index] = potential_energy.value_in_unit(
            openmm.unit.kilojoule_per_mole
        )
        reduced_potentials[frame_index] = unreduced_potential * beta

    potentials *= unit.kilojoule / unit.mole
    reduced_potentials *= unit.dimensionless

    observables_frame = ObservableFrame(
        {
            ObservableType.PotentialEnergy: ObservableArray(potentials),
            ObservableType.ReducedPotential: ObservableArray(reduced_potentials),
        }
    )

    return observables_frame


def _compute_gradients(
    gradient_parameters: List[ParameterGradientKey],
    observables: ObservableFrame,
    force_field: "ForceField",
    thermodynamic_state: ThermodynamicState,
    topology: "Topology",
    trajectory: "Trajectory",
    compute_resources: ComputeResources,
    enable_pbc: bool = True,
    perturbation_amount: float = 0.0001,
):
    """Computes the gradients of the provided observables with respect to
    the set of specified force field parameters using the central difference
    finite difference method.

    Notes
    -----
    The ``observables`` object will be modified in-place.

    Parameters
    ----------
    gradient_parameters
        The parameters to differentiate with respect to.
    observables
        The observables to differentiate.
    force_field
        The full set force field parameters which contain the parameters to
        differentiate.
    thermodynamic_state
        The state at which the trajectory was sampled
    topology
        The topology of the system the observables were collected for.
    trajectory
        The trajectory over which the observables were collected.
    compute_resources
        The compute resources available for the computations.
    enable_pbc
        Whether PBC should be enabled when re-evaluating system energies.
    perturbation_amount
        The amount to perturb for the force field parameter by.
    """

    import openmm

    gradients = defaultdict(list)
    observables.clear_gradients()

    if enable_pbc:
        # Make sure the PBC are set on the topology otherwise the cut-off will be
        # set incorrectly.
        topology.box_vectors = from_openmm(trajectory.openmm_boxes(0))

    for parameter_key in gradient_parameters:
        # Build the slightly perturbed systems.
        reverse_system, reverse_parameter_value = system_subset(
            parameter_key, force_field, topology, -perturbation_amount
        )
        forward_system, forward_parameter_value = system_subset(
            parameter_key, force_field, topology, perturbation_amount
        )

        # Perform a cheap check to try and catch most cases where the systems energy
        # does not depend on this parameter.
        reverse_xml = openmm.XmlSerializer.serialize(reverse_system)
        forward_xml = openmm.XmlSerializer.serialize(forward_system)

        if not enable_pbc:
            disable_pbc(reverse_system)
            disable_pbc(forward_system)

        # Evaluate the energies using the reverse and forward sub-systems.
        if reverse_xml != forward_xml:
            reverse_energies = _evaluate_energies(
                thermodynamic_state,
                reverse_system,
                trajectory,
                compute_resources,
                enable_pbc,
            )
            forward_energies = _evaluate_energies(
                thermodynamic_state,
                forward_system,
                trajectory,
                compute_resources,
                enable_pbc,
            )
        else:
            zeros = np.zeros(len(trajectory))

            reverse_energies = forward_energies = ObservableFrame(
                {
                    ObservableType.PotentialEnergy: ObservableArray(
                        zeros * unit.kilojoule / unit.mole,
                        [
                            ParameterGradient(
                                key=parameter_key,
                                value=(
                                    zeros
                                    * unit.kilojoule
                                    / unit.mole
                                    / reverse_parameter_value.units
                                ),
                            )
                        ],
                    ),
                    ObservableType.ReducedPotential: ObservableArray(
                        zeros * unit.dimensionless,
                        [
                            ParameterGradient(
                                key=parameter_key,
                                value=(
                                    zeros
                                    * unit.dimensionless
                                    / reverse_parameter_value.units
                                ),
                            )
                        ],
                    ),
                }
            )

        potential_gradient = ParameterGradient(
            key=parameter_key,
            value=(
                forward_energies[ObservableType.PotentialEnergy].value
                - reverse_energies[ObservableType.PotentialEnergy].value
            )
            / (forward_parameter_value - reverse_parameter_value),
        )
        reduced_potential_gradient = ParameterGradient(
            key=parameter_key,
            value=(
                forward_energies[ObservableType.ReducedPotential].value
                - reverse_energies[ObservableType.ReducedPotential].value
            )
            / (forward_parameter_value - reverse_parameter_value),
        )

        gradients[ObservableType.PotentialEnergy].append(potential_gradient)
        gradients[ObservableType.TotalEnergy].append(potential_gradient)
        gradients[ObservableType.Enthalpy].append(potential_gradient)
        gradients[ObservableType.ReducedPotential].append(reduced_potential_gradient)

        if ObservableType.KineticEnergy in observables:
            gradients[ObservableType.KineticEnergy].append(
                ParameterGradient(
                    key=parameter_key,
                    value=(
                        np.zeros(potential_gradient.value.shape)
                        * observables[ObservableType.KineticEnergy].value.units
                        / reverse_parameter_value.units
                    ),
                )
            )
        if ObservableType.Density in observables:
            gradients[ObservableType.Density].append(
                ParameterGradient(
                    key=parameter_key,
                    value=(
                        np.zeros(potential_gradient.value.shape)
                        * observables[ObservableType.Density].value.units
                        / reverse_parameter_value.units
                    ),
                )
            )
        if ObservableType.Volume in observables:
            gradients[ObservableType.Volume].append(
                ParameterGradient(
                    key=parameter_key,
                    value=(
                        np.zeros(potential_gradient.value.shape)
                        * observables[ObservableType.Volume].value.units
                        / reverse_parameter_value.units
                    ),
                )
            )

    for observable_type in observables:
        observables[observable_type] = ObservableArray(
            value=observables[observable_type].value,
            gradients=gradients[observable_type],
        )


@workflow_protocol()
class OpenMMEnergyMinimisation(BaseEnergyMinimisation):
    """A protocol to minimise the potential energy of a system using
    OpenMM.
    """

    def _execute(self, directory, available_resources):
        import openmm
        import openmm.unit
        from openmm import app

        platform = setup_platform_with_resources(available_resources)

        input_pdb_file = app.PDBFile(self.input_coordinate_file)
        system = self.parameterized_system.system

        if not self.enable_pbc:
            disable_pbc(system=system)

        # TODO: Expose the constraint tolerance
        integrator = openmm.VerletIntegrator(0.002 * openmm.unit.picoseconds)
        simulation = app.Simulation(
            input_pdb_file.topology, system, integrator, platform
        )

        update_context_with_pdb(simulation.context, input_pdb_file)

        simulation.minimizeEnergy(to_openmm(self.tolerance), self.max_iterations)

        positions = extract_positions(
            simulation.context.getState(getPositions=True),
            # Discard any v-sites.
            extract_atom_indices(system),
        )

        self.output_coordinate_file = os.path.join(directory, "minimised.pdb")

        with open(self.output_coordinate_file, "w+") as minimised_file:
            app.PDBFile.writeFile(simulation.topology, positions, minimised_file)


@workflow_protocol()
class OpenMMSimulation(BaseSimulation):
    """Performs a molecular dynamics simulation in a given ensemble using
    an OpenMM backend.

    This protocol employs the Langevin integrator implemented in the ``openmmtools``
    package to propagate the state of the system using the default BAOAB splitting [1]_.
    Further, simulations which are run in the NPT simulation will have a Monte Carlo
    barostat (openmm.MonteCarloBarostat) applied every 25 steps (the OpenMM
    default).

    References
    ----------
    [1] Leimkuhler, Ben, and Charles Matthews. "Numerical methods for stochastic
        molecular dynamics." Molecular Dynamics. Springer, Cham, 2015. 261-328.
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

        def __init__(self, integrator, topology, system, context, current_step):
            self.integrator = integrator
            self.topology = topology
            self.system = system
            self.context = context
            self.currentStep = current_step

    class _DCDReporter:
        def __init__(self, file, append=False):
            self._append = append

            mode = "r+b" if append else "wb"

            self._out = open(file, mode)

            self._dcd = None
            self._atom_indices = None

        def report(self, simulation, state):
            from openmm import app

            if self._dcd is None:
                self._dcd = app.DCDFile(
                    self._out,
                    simulation.topology,
                    simulation.integrator.getStepSize(),
                    simulation.currentStep,
                    0,
                    self._append,
                )

                system = simulation.system

                self._atom_indices = extract_atom_indices(system)

            self._dcd.writeModel(
                extract_positions(state, self._atom_indices),
                periodicBoxVectors=state.getPeriodicBoxVectors(),
            )

        def __del__(self):
            self._out.close()

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._checkpoint_path = None
        self._state_path = None

        self._local_trajectory_path = None
        self._local_statistics_path = None

        self._context = None
        self._integrator = None

    def _execute(self, directory, available_resources):
        import mdtraj
        from openmm import app

        # We handle most things in OMM units here.
        temperature = self.thermodynamic_state.temperature
        openmm_temperature = to_openmm(temperature)

        if self.ensemble == Ensemble.NVT:
            pressure = None
            openmm_pressure = None
        else:
            pressure = self.thermodynamic_state.pressure
            openmm_pressure = to_openmm(pressure)

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

        self.output_coordinate_file = os.path.join(directory, "output.pdb")

        # Run the simulation.
        self._simulate(self._context, self._integrator)

        # Set the output paths.
        self.trajectory_file_path = self._local_trajectory_path
        self.observables_file_path = os.path.join(directory, "observables.json")

        # Compute the final observables
        self.observables = self._compute_final_observables(temperature, pressure)

        # Optionally compute any gradients.
        if len(self.gradient_parameters) == 0:
            return

        if not isinstance(
            self.parameterized_system.force_field, SmirnoffForceFieldSource
        ):
            raise ValueError(
                "Derivates can only be computed for systems parameterized with SMIRNOFF "
                "force fields."
            )

        _compute_gradients(
            self.gradient_parameters,
            self.observables,
            self.parameterized_system.force_field.to_force_field(),
            self.thermodynamic_state,
            self.parameterized_system.topology,
            mdtraj.load_dcd(self.trajectory_file_path, top=self.input_coordinate_file),
            available_resources,
            self.enable_pbc,
        )

    def _setup_simulation_objects(self, temperature, pressure, available_resources):
        """Initializes the objects needed to perform the simulation.
        This comprises of a context, and an integrator.

        Parameters
        ----------
        temperature: openff.evaluator.unit.Quantity
            The temperature to run the simulation at.
        pressure: openff.evaluator.unit.Quantity
            The pressure to run the simulation at.
        available_resources: ComputeResources
            The resources available to run on.

        Returns
        -------
        openmm.Context
            The created openmm context which takes advantage
            of the available compute resources.
        openmmtools.integrators.LangevinIntegrator
            The Langevin integrator which will propogate
            the simulation.
        """

        import openmm
        import openmmtools
        from openmm import app

        # Create a platform with the correct resources.
        if not self.allow_gpu_platforms:
            from openff.evaluator.backends import ComputeResources

            available_resources = ComputeResources(
                available_resources.number_of_threads
            )

        platform = setup_platform_with_resources(
            available_resources, self.high_precision
        )

        # Load in the system object.
        system = self.parameterized_system.system

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
        thermostat_friction = to_openmm(self.thermostat_friction)
        timestep = to_openmm(self.timestep)

        integrator = openmmtools.integrators.LangevinIntegrator(
            temperature=temperature,
            collision_rate=thermostat_friction,
            timestep=timestep,
        )

        # Create the simulation context.
        context = openmm.Context(system, integrator, platform)

        # Initialize the context with the correct positions etc.
        input_pdb_file = app.PDBFile(self.input_coordinate_file)
        box_vectors = None

        if self.enable_pbc:
            # Optionally set up the box vectors.
            box_vectors = input_pdb_file.topology.getPeriodicBoxVectors()

            if box_vectors is None:
                raise ValueError(
                    "The input file must contain box vectors when running with PBC."
                )

        update_context_with_positions(
            context, input_pdb_file.getPositions(asNumpy=True), box_vectors
        )

        context.setVelocitiesToTemperature(temperature)

        return context, integrator

    def _write_checkpoint_file(self, current_step_number, context):
        """Writes a simulation checkpoint file to disk.

        Parameters
        ----------
        current_step_number: int
            The total number of steps which have been taken so
            far.
        context: openmm.Context
            The current OpenMM context.
        """
        import openmm

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
        context: openmm.Context
            The current OpenMM context.

        Returns
        -------
        int
            The current step number.
        """
        import openmm

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

    def _compute_final_observables(self, temperature, pressure) -> ObservableFrame:
        """Converts the openmm statistic csv file into an openff-evaluator
        ``ObservableFrame`` and computes additional missing data, such as reduced
        potentials and derivatives of the energies with respect to any requested
        force field parameters.

        Parameters
        ----------
        temperature: openff.evaluator.unit.Quantity
            The temperature that the simulation is being run at.
        pressure: openff.evaluator.unit.Quantity
            The pressure that the simulation is being run at.
        """
        observables = ObservableFrame.from_openmm(self._local_statistics_path, pressure)

        reduced_potentials = (
            observables[ObservableType.PotentialEnergy].value / unit.avogadro_constant
        )

        if pressure is not None:
            pv_terms = pressure * observables[ObservableType.Volume].value
            reduced_potentials += pv_terms

        beta = 1.0 / (unit.boltzmann_constant * temperature)

        observables[ObservableType.ReducedPotential] = ObservableArray(
            value=(beta * reduced_potentials).to(unit.dimensionless)
        )

        if pressure is not None:
            observables[ObservableType.Enthalpy] = observables[
                ObservableType.TotalEnergy
            ] + observables[ObservableType.Volume] * pressure * (
                1.0 * unit.avogadro_constant
            )

        return observables

    def _simulate(self, context, integrator):
        """Performs the simulation using a given context
        and integrator.

        Parameters
        ----------
        context: openmm.Context
            The OpenMM context to run with.
        integrator: openmm.Integrator
            The integrator to evolve the simulation with.
        """
        from openmm import app

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
        dcd_reporter = self._DCDReporter(self._local_trajectory_path, append_trajectory)

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
        system = self.parameterized_system.system
        simulation = self._Simulation(
            integrator,
            topology,
            system,
            context,
            current_step,
        )

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

        positions = extract_positions(final_state, extract_atom_indices(system))
        topology.setPeriodicBoxVectors(final_state.getPeriodicBoxVectors())

        with open(self.output_coordinate_file, "w+") as configuration_file:
            app.PDBFile.writeFile(topology, positions, configuration_file)


@workflow_protocol()
class OpenMMEvaluateEnergies(BaseEvaluateEnergies):
    """Re-evaluates the energy of a series of configurations for a given set of force
    field parameters using OpenMM.
    """

    def _execute(self, directory, available_resources):
        import mdtraj

        # Load in the inputs.
        trajectory = mdtraj.load_dcd(
            self.trajectory_file_path, self.parameterized_system.topology_path
        )
        system = self.parameterized_system.system

        # Re-evaluate the energies.
        self.output_observables = _evaluate_energies(
            self.thermodynamic_state,
            system,
            trajectory,
            available_resources,
            self.enable_pbc,
            False,
        )

        # Optionally compute any gradients.
        if len(self.gradient_parameters) == 0:
            return

        if not isinstance(
            self.parameterized_system.force_field, SmirnoffForceFieldSource
        ):
            raise ValueError(
                "Derivates can only be computed for systems parameterized with SMIRNOFF "
                "force fields."
            )

        force_field = self.parameterized_system.force_field.to_force_field()

        _compute_gradients(
            self.gradient_parameters,
            self.output_observables,
            force_field,
            self.thermodynamic_state,
            self.parameterized_system.topology,
            trajectory,
            available_resources,
            self.enable_pbc,
        )
