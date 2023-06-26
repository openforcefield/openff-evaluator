import os
from enum import Enum
from typing import Union

import numpy as np
import openmm.app as app
from openff.units import unit
from openff.units.openmm import to_openmm
from paprika.analysis.utils import get_block_sem
from paprika.io import load_trajectory, read_restraint_data

from openff.evaluator.forcefield.system import ParameterizedSystem
from openff.evaluator.protocols.openmm import OpenMMSimulation
from openff.evaluator.protocols.paprika.analysis import ComputePotentialEnergyGradient
from openff.evaluator.protocols.paprika.restraints import ApplyRestraints
from openff.evaluator.thermodynamics import Ensemble
from openff.evaluator.utils import is_file_and_not_empty
from openff.evaluator.workflow import workflow_protocol
from openff.evaluator.workflow.attributes import InputAttribute, OutputAttribute


class ConvergenceType(Enum):
    """Enum class for choosing the property type for convergence calculation."""

    restraints = "restraints"
    potential_energy = "potential_energy"
    gradient = "gradient"


class APRSimulationSteps:
    """A helper class to define the length of simulation (thermalization,
    equilibration, production) for host-guest binding affinity calculation
    with the pAPRika protocol. The production run can be extended by
    `number_of_extra_steps` up to a maximum of `max_number_of_steps` or
    until convergence is reached.

    Examples
    --------

    To create a protocol with (0.1, 1.0, 2.0) ns for (thermalization, equilbiration, production)
    runs

    >>> APR_setting = APRSimulationSteps(
    >>>     n_thermalization_steps=50000,
    >>>     n_equilibration_steps=500000,
    >>>     n_production_steps=1000000,
    >>> )

    Add this to the pAPRika schema

    >>> host_guest_schema = HostGuestBindingAffinity.default_paprika_schema(
    >>>     simulation_time_steps=APR_setting,
    >>> )

    """

    class _MDSteps:
        @property
        def number_of_steps(self) -> int:
            """Number of MD steps."""
            return self._number_of_steps

        @number_of_steps.setter
        def number_of_steps(self, value: int):
            self._number_of_steps = value

        @property
        def time_step(self) -> unit.Quantity:
            """The integration time step in femtosecond (can be a `float` or `unit.Quantity`)."""
            return self._time_step

        @time_step.setter
        def time_step(self, value: Union[float, unit.Quantity]):
            if isinstance(value, float):
                self._time_step = value * unit.femtosecond
            elif isinstance(value, unit.Quantity):
                self._time_step = value

        @property
        def output_frequency(self) -> int:
            """Number of steps between saving coordinates and statistics."""
            return self._output_frequency

        @output_frequency.setter
        def output_frequency(self, value: int):
            self._output_frequency = value

        @property
        def number_of_extra_steps(self) -> int:
            """Number of extra steps to run after the initial `number_of_steps`."""
            return self._number_of_extra_steps

        @number_of_extra_steps.setter
        def number_of_extra_steps(self, value: int):
            self._number_of_extra_steps = value

        @property
        def max_number_of_steps(self) -> int:
            """Maximum number of steps (i.e. upper limit of the simulation length)."""
            return self._max_number_of_steps

        @max_number_of_steps.setter
        def max_number_of_steps(self, value: int):
            self._max_number_of_steps = value

        def __init__(
            self,
            number_of_steps,
            time_step,
            output_frequency,
            number_of_extra_steps=None,
            max_number_of_steps=None,
        ):
            self._number_of_steps = number_of_steps
            self._time_step = time_step
            self._output_frequency = output_frequency

            # Useful for extending production run until convergence
            self._number_of_extra_steps = number_of_extra_steps
            self._max_number_of_steps = max_number_of_steps

        def __str__(self):
            return f"number_of_steps={self._number_of_steps}, time_step={self._time_step}, output_frequency={self._output_frequency}, number_of_extra_steps={self._number_of_extra_steps}, max_number_of_steps={self._max_number_of_steps}"

        def __repr__(self):
            return f"<_MDSteps {str(self)}>"

    @property
    def thermalization(self) -> _MDSteps:
        """The settings for thermalization run."""
        return self._thermalization

    @property
    def equilibration(self) -> _MDSteps:
        """The settings for equilibration run."""
        return self._equilibration

    @property
    def production(self) -> _MDSteps:
        """The settings for production run."""
        return self._production

    @property
    def convergence_criteria(self) -> ConvergenceType:
        """The settings for convergence in production runs."""
        return self._convergence_criteria

    @property
    def convergence_tolerance(self) -> float:
        """The settings for convergence in production runs."""
        return self._convergence_tolerance

    def __init__(
        self,
        n_thermalization_steps=50000,
        n_equilibration_steps=500000,
        n_production_steps=1000000,
        dt_thermalization=1.0 * unit.femtosecond,
        dt_equilibration=2.0 * unit.femtosecond,
        dt_production=2.0 * unit.femtosecond,
        out_thermalization=10000,
        out_equilibration=10000,
        out_production=5000,
        production_extra_steps=None,
        production_max_steps=None,
        convergence_criteria=None,
        convergence_tolerance=None,
    ):
        self._thermalization = self._MDSteps(
            n_thermalization_steps, dt_thermalization, out_thermalization
        )
        self._equilibration = self._MDSteps(
            n_equilibration_steps, dt_equilibration, out_equilibration
        )
        self._production = self._MDSteps(
            n_production_steps,
            dt_production,
            out_production,
            production_extra_steps,
            production_max_steps,
        )
        self._convergence_criteria = convergence_criteria
        self._convergence_tolerance = convergence_tolerance


# class HFESimulationSteps:
#     def __init__(self):
#         self._equilibration_steps = 100000
#         self._equilibration_output_frequency = 10000
#         self._equilibration_time_step = 2.0 * unit.femtosecond
#         self._number_of_iterations = 500
#         self._steps_per_iteration = 50
#         self._total_iterations = 2000
#         self._max_iterations = 20
#         self._electrostatic_lambdas_1 = None
#         self._electrostatic_lambdas_2 = None
#         self._steric_lambdas_1 = None
#         self._steric_lambdas_2 = None


@workflow_protocol()
class PaprikaOpenMMSimulation(OpenMMSimulation):
    pristine_system = InputAttribute(
        docstring="The parameterized system object without dummy atoms.",
        type_hint=Union[ParameterizedSystem, None],
        default_value=None,
    )
    number_of_extra_steps = InputAttribute(
        docstring="The number of steps to run after `steps_per_iteration`. "
        "The simulations will continue to run until convergence is reached "
        "or until `max_number_of_steps`.",
        type_hint=Union[int, None],
        default_value=None,
    )
    max_number_of_steps = InputAttribute(
        docstring="Maximum number of steps to propagate the system by "
        "at each iteration.",
        type_hint=Union[int, None],
        default_value=None,
    )
    phase = InputAttribute(
        docstring="The phase of the current window",
        type_hint=Union[str, None],
        default_value=None,
    )
    window_number = InputAttribute(
        docstring="The current window index",
        type_hint=Union[int, None],
        default_value=None,
    )
    lambda_scaling = InputAttribute(
        docstring="A dictionary containing the list of scaling factor "
        "to multiply the SEM of the forces with.",
        type_hint=Union[dict, None],
        default_value=None,
    )
    convergence_criteria = InputAttribute(
        docstring="The convergence criteria to use - (1) use restraint forces (`restraints`),"
        "(2) potential energy, and (3) free-energy gradient (`gradient`) to estimate the SEM.",
        type_hint=Union[ConvergenceType, None],
        default_value=None,
    )
    convergence_tolerance = InputAttribute(
        docstring="The convergence tolerance for the SEM.",
        type_hint=Union[dict, float, None],
        default_value=None,
    )
    restraints_path = InputAttribute(
        docstring="The file path to the JSON file which contains the restraint "
        "definitions. This will usually have been generated by a "
        "`GenerateXXXRestraints` protocol.",
        type_hint=Union[str, None],
        default_value=None,
    )
    total_steps = OutputAttribute(
        docstring="Total number of steps run for this simulation.",
        type_hint=int,
    )
    standard_error = OutputAttribute(
        docstring="The standard error of the mean.",
        type_hint=Union[np.ndarray, None],
    )

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._directory = None
        self._available_resources = None
        self._local_input_coordinate_path = None

    def _estimate_restraint_sem(self):
        """Estimate the SEM of all active APR restraints for the current window."""

        # Load the restraints
        restraints = ApplyRestraints.load_restraints(self.restraints_path)
        restraint_list = []
        if self.phase == "attach":
            restraint_list = restraints["guest"] + restraints["conformational"]
        elif self.phase == "pull":
            for restraint in restraints["guest"]:
                # Find the distance `r` colvar restraint
                if not restraint.mask3 and not restraint.mask4:
                    restraint_list.append(restraint)
        elif self.phase == "release":
            restraint_list = restraints["conformational"]

        if len(restraint_list) == 0:
            ValueError(
                "There are no restraints configured, APR calculations require restraints."
            )

        # Load the trajectory
        trajectory = load_trajectory(
            "", self._local_trajectory_path, self._local_input_coordinate_path
        )

        # Estimate the sum of all active restraint forces
        restraint_forces = []

        for restraint_index, restraint in enumerate(restraint_list):
            restraint_forces.append([])
            restraint_forces[restraint_index] = read_restraint_data(
                trajectory, restraint
            )

            # Equilibrium position
            equilibrium = restraint.phase[self.phase]["targets"][self.window_number]

            # Phase correction for torsions
            if restraint.mask4:
                restraint_forces[restraint_index] = np.minimum(
                    np.minimum(
                        np.abs(restraint_forces[restraint_index] - equilibrium),
                        np.abs(restraint_forces[restraint_index] - equilibrium + 360),
                    ),
                    np.abs(restraint_forces[restraint_index] - equilibrium - 360),
                )
            else:
                restraint_forces[restraint_index] -= equilibrium

            # Force constant - take the value at lambda = 1
            force_constant = restraint.phase[self.phase]["force_constants"][-1]
            if self.phase == "release":
                force_constant = restraint.phase[self.phase]["force_constants"][0]

            # Convert kcal/mol/radians**2 to kcal/mol/degrees**2
            if restraint.mask3 or restraint.mask4:
                force_constant *= (np.pi / 180) ** 2

            # Compute the Forces due to restraining potential
            if self.phase == "attach" or self.phase == "release":
                # Harmonic restraint potential : k * (x - x0)^2
                # Restraint forces (deriv w.r.t. lambda): k * <(x - x0)^2>_l
                restraint_forces[restraint_index] = (
                    force_constant * restraint_forces[restraint_index] ** 2
                )
            elif self.phase == "pull":
                # Harmonic restraint potential: k * (x - x0)^2
                # Restraint forces (deriv w.r.t. x): 2 * k <(x - x0)>_i
                restraint_forces[restraint_index] *= 2 * force_constant

        restraint_forces = np.sum(np.array(restraint_forces), axis=0)

        sem = (
            get_block_sem(restraint_forces)
            * self.lambda_scaling[self.phase][self.window_number]
        )

        return np.array(sem)

    def _estimate_gradient_sem(self, return_percentage=False):
        """Estimate the SEM of the gradient of the potential energy w.r.t. FF
        parameters. The function can return the absolute value or the fractional
        (as a percentage) value of the SEM.

        .. note ::
            It may be better to return the fractional value of the SEM since the
            magnitude may vary depending on the FF parameter.
        """

        gradient = ComputePotentialEnergyGradient("")
        gradient.input_system = self.pristine_system
        gradient.topology_path = self._local_input_coordinate_path
        gradient.trajectory_path = self._local_trajectory_path
        gradient.thermodynamic_state = self.thermodynamic_state
        gradient.enable_pbc = self.enable_pbc
        gradient.gradient_parameters = self.gradient_parameters
        gradient.execute(self._directory, self._available_resources)

        ave = []
        sem = []
        for gradient in gradient.potential_energy_gradients_data:
            ave.append(gradient.value.magnitude.mean())
            sem.append(get_block_sem(gradient.value.magnitude))

        if return_percentage:
            return np.array(sem) / np.array(ave) * 100

        return np.array(sem)

    def _estimate_potential_energy_sem(self):
        """Estimate the SEM of the total potential energy."""
        raise NotImplementedError()

    def _estimate_sem(self):
        """Estimate the SEM based on either (1) restraint forces, (2) gradient or
        (3) total potential energy.
        """
        if self.convergence_criteria == ConvergenceType.restraints:
            return self._estimate_restraint_sem()
        elif self.convergence_criteria == ConvergenceType.gradient:
            return self._estimate_gradient_sem(return_percentage=True)
        elif self.convergence_criteria == ConvergenceType.potential_energy:
            self._estimate_potential_energy_sem()

    def _simulate(self, context, integrator):
        """Performs the simulation using a given context and integrator.
        This overwrites the original method to include an automated
        convergence checking.

        Parameters
        ----------
        context: simtk.openmm.Context
            The OpenMM context to run with.
        integrator: simtk.openmm.Integrator
            The integrator to evolve the simulation with.
        """

        # Define how many steps should be taken.
        total_number_of_steps = (
            self.total_number_of_iterations * self.steps_per_iteration
        )
        if self.max_number_of_steps:
            total_number_of_steps = self.max_number_of_steps

        # Try to load the current state from any available checkpoint information
        current_step = self._resume_from_checkpoint(context)
        self.total_steps = current_step

        # If there is already a file then check for convergence
        if (
            self.convergence_criteria
            and current_step >= self.steps_per_iteration
            and is_file_and_not_empty(self._local_trajectory_path)
        ):
            self.standard_error = self._estimate_sem()
            if isinstance(self.convergence_tolerance, dict):
                if self.standard_error < self.convergence_tolerance[self.phase]:
                    return
            elif isinstance(self.convergence_tolerance, float):
                if np.all(self.standard_error < self.convergence_tolerance):
                    return

        # Return if maximum number steps is reached
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
        system = self.parameterized_system.system
        simulation = self._Simulation(
            integrator, topology, system, context, current_step
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

            # Convergence checking
            if (
                self.convergence_criteria
                and current_step >= self.steps_per_iteration
                and current_step % self.number_of_extra_steps == 0
            ):
                self.standard_error = self._estimate_sem()
                if isinstance(self.convergence_tolerance, dict):
                    if self.standard_error < self.convergence_tolerance[self.phase]:
                        break
                elif isinstance(self.convergence_tolerance, float):
                    if np.all(self.standard_error < self.convergence_tolerance):
                        break

        self.total_steps = current_step

        # Save out the final positions.
        self._write_checkpoint_file(current_step, context)

        final_state = context.getState(getPositions=True)

        positions = final_state.getPositions()
        topology.setPeriodicBoxVectors(final_state.getPeriodicBoxVectors())

        with open(self.output_coordinate_file, "w+") as configuration_file:
            app.PDBFile.writeFile(topology, positions, configuration_file)

    def _execute(self, directory, available_resources):
        self._directory = directory
        self._available_resources = available_resources

        if self.thermodynamic_state.temperature is None:
            raise ValueError(
                "A temperature must be set to perform a simulation in any ensemble"
            )

        # We handle most things in OMM units here.
        temperature = self.thermodynamic_state.temperature
        openmm_temperature = to_openmm(temperature)

        if self.ensemble == Ensemble.NVT:
            pressure = None
            openmm_pressure = None
        else:
            pressure = self.thermodynamic_state.pressure
            openmm_pressure = to_openmm(pressure)

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
        self._local_input_coordinate_path = os.path.join(directory, "input.pdb")

        if not is_file_and_not_empty(self._local_input_coordinate_path):
            input_pdb_file = app.PDBFile(self.input_coordinate_file)

            with open(self._local_input_coordinate_path, "w+") as configuration_file:
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
