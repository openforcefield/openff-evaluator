"""
A collection of protocols for running molecular simulations.
"""
import logging
import math
import os
import shutil
import traceback

import numpy as np
from propertyestimator import unit
from propertyestimator.thermodynamics import ThermodynamicState, Ensemble
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import setup_platform_with_resources, openmm_quantity_to_pint, \
    pint_quantity_to_openmm, disable_pbc
from propertyestimator.utils.statistics import StatisticsArray, ObservableType
from propertyestimator.utils.utils import safe_unlink
from propertyestimator.workflow.decorators import protocol_input, protocol_output, InequalityMergeBehaviour, UNDEFINED
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol
from simtk import openmm, unit as simtk_unit
from simtk.openmm import app


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
class RunOpenMMSimulation(BaseProtocol):
    """Performs a molecular dynamics simulation in a given ensemble using
    an OpenMM backend.
    """

    steps = protocol_input(
        docstring='The number of timesteps to evolve the system by.',
        type_hint=int, merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=1000000
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

    output_frequency = protocol_input(
        docstring='The frequency with which to write to the output statistics and '
                  'trajectory files.',
        type_hint=int, merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=3000
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

    save_rolling_statistics = protocol_input(
        docstring='If True, the statistics file will be written to every '
                  '`output_frequency` number of steps, rather than just once at '
                  'the end of the simulation.',
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

        self._temporary_statistics_path = None
        self._temporary_trajectory_path = None

        self._checkpoint_path = None

        self._context = None
        self._integrator = None

    def execute(self, directory, available_resources):

        # We handle most things in OMM units here.
        temperature = pint_quantity_to_openmm(self.thermodynamic_state.temperature)
        pressure = pint_quantity_to_openmm(None if self.ensemble == Ensemble.NVT else
                                           self.thermodynamic_state.pressure)

        if temperature is None:

            return PropertyEstimatorException(directory=directory,
                                              message='A temperature must be set to perform '
                                                      'a simulation in any ensemble')

        if Ensemble(self.ensemble) == Ensemble.NPT and pressure is None:

            return PropertyEstimatorException(directory=directory,
                                              message='A pressure must be set to perform an NPT simulation')

        if Ensemble(self.ensemble) == Ensemble.NPT and self.enable_pbc is False:

            return PropertyEstimatorException(directory=directory,
                                              message='PBC must be enabled when running in the NPT ensemble.')

        logging.info('Performing a simulation in the ' + str(self.ensemble) + ' ensemble: ' + self.id)

        # Clean up any temporary files from previous (possibly failed)
        # simulations.
        self._temporary_statistics_path = os.path.join(directory, 'temp_statistics.csv')
        self._temporary_trajectory_path = os.path.join(directory, 'temp_trajectory.dcd')

        safe_unlink(self._temporary_statistics_path)
        safe_unlink(self._temporary_trajectory_path)

        self._checkpoint_path = os.path.join(directory, 'checkpoint.xml')

        # Set up the output file paths
        self.trajectory_file_path = os.path.join(directory, 'trajectory.dcd')
        self.statistics_file_path = os.path.join(directory, 'statistics.csv')

        # Set up the simulation objects.
        if self._context is None or self._integrator is None:

            self._context, self._integrator = self._setup_simulation_objects(temperature,
                                                                             pressure,
                                                                             available_resources)

        result = self._simulate(directory, temperature, pressure, self._context, self._integrator)
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
        if os.path.isfile(self._checkpoint_path):

            # Load the simulation state from a checkpoint file.
            logging.info(f'Loading the checkpoint from {self._checkpoint_path}.')

            with open(self._checkpoint_path, 'r') as file:
                checkpoint_state = XmlSerializer.deserialize(file.read())

            context.setState(checkpoint_state)

        else:

            logging.info(f'No checkpoint file was found at {self._checkpoint_path}.')

            # Populate the simulation object from the starting input files.
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

    def _write_statistics_array(self, raw_statistics, current_step, temperature, pressure,
                                degrees_of_freedom, total_mass):
        """Appends a set of statistics to an existing `statistics_array`.
        Those statistics are potential energy, kinetic energy, total energy,
        volume, density and reduced potential.

        Parameters
        ----------
        raw_statistics: dict of ObservableType and numpy.ndarray
            A dictionary of potential energies (kJ/mol), kinetic
            energies (kJ/mol) and volumes (angstrom**3).
        current_step: int
            The index of the current step.
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
        temperature = openmm_quantity_to_pint(temperature)
        pressure = openmm_quantity_to_pint(pressure)

        beta = 1.0 / (unit.boltzmann_constant * temperature)

        raw_potential_energies = raw_statistics[ObservableType.PotentialEnergy][0:current_step + 1]
        raw_kinetic_energies = raw_statistics[ObservableType.KineticEnergy][0:current_step + 1]
        raw_volumes = raw_statistics[ObservableType.Volume][0:current_step + 1]

        potential_energies = raw_potential_energies * unit.kilojoules / unit.mole
        kinetic_energies = raw_kinetic_energies * unit.kilojoules / unit.mole
        volumes = raw_volumes * unit.angstrom ** 3

        # Calculate the instantaneous temperature, taking account the
        # systems degrees of freedom.
        temperatures = 2.0 * kinetic_energies / (degrees_of_freedom * unit.molar_gas_constant)

        # Calculate the systems enthalpy and reduced potential.
        total_energies = potential_energies + kinetic_energies
        enthalpies = None

        reduced_potentials = potential_energies / unit.avogadro_number

        if pressure is not None:

            pv_terms = pressure * volumes

            reduced_potentials += pv_terms
            enthalpies = total_energies + pv_terms * unit.avogadro_number

        reduced_potentials = (beta * reduced_potentials) * unit.dimensionless

        # Calculate the systems density.
        densities = total_mass / (volumes * unit.avogadro_number)

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

        # Build the reporters which we will use to report the state
        # of the simulation.
        input_pdb_file = app.PDBFile(self.input_coordinate_file)
        topology = input_pdb_file.topology

        with open(os.path.join(directory, 'input.pdb'), 'w+') as configuration_file:
            app.PDBFile.writeFile(input_pdb_file.topology, input_pdb_file.positions, configuration_file)

        # Make a copy of the existing trajectory to append to if one already exists.
        append_trajectory = False

        if os.path.isfile(self.trajectory_file_path):

            shutil.copyfile(self.trajectory_file_path, self._temporary_trajectory_path)
            append_trajectory = True

        elif os.path.isfile(self._temporary_trajectory_path):
            os.unlink(self._temporary_trajectory_path)

        if append_trajectory:
            trajectory_file_object = open(self._temporary_trajectory_path, 'r+b')
        else:
            trajectory_file_object = open(self._temporary_trajectory_path, 'w+b')

        trajectory_dcd_object = app.DCDFile(trajectory_file_object,
                                            topology,
                                            integrator.getStepSize(),
                                            0,
                                            self.output_frequency,
                                            append_trajectory)

        expected_number_of_statistics = math.ceil(self.steps / self.output_frequency)

        raw_statistics = {
            ObservableType.PotentialEnergy: np.zeros(expected_number_of_statistics),
            ObservableType.KineticEnergy: np.zeros(expected_number_of_statistics),
            ObservableType.Volume: np.zeros(expected_number_of_statistics),
        }

        # Define any constants needed for extracting system statistics
        # Compute the instantaneous temperature of degrees of freedom.
        # This snipped is taken from the build in OpenMM `StateDataReporter`
        system = context.getSystem()

        degrees_of_freedom = sum([3 for i in range(system.getNumParticles()) if
                                  system.getParticleMass(i) > 0 * simtk_unit.dalton])

        degrees_of_freedom -= system.getNumConstraints()

        if any(type(system.getForce(i)) == openmm.CMMotionRemover for i in range(system.getNumForces())):
            degrees_of_freedom -= 3

        total_mass = 0.0 * simtk_unit.dalton

        for i in range(system.getNumParticles()):
            total_mass += system.getParticleMass(i)

        total_mass = openmm_quantity_to_pint(total_mass)

        # Perform the simulation.
        current_step_count = 0
        current_step = 0

        result = None

        try:

            while current_step_count < self.steps:

                steps_to_take = min(self.output_frequency, self.steps - current_step_count)
                integrator.step(steps_to_take)

                state = context.getState(getPositions=True,
                                         getEnergy=True,
                                         getVelocities=False,
                                         getForces=False,
                                         getParameters=False,
                                         enforcePeriodicBox=self.enable_pbc)

                # Write out the current frame of the trajectory.
                trajectory_dcd_object.writeModel(positions=state.getPositions(),
                                                 periodicBoxVectors=state.getPeriodicBoxVectors())

                # Write out the energies and system statistics.
                raw_statistics[ObservableType.PotentialEnergy][current_step] = \
                    state.getPotentialEnergy().value_in_unit(simtk_unit.kilojoules_per_mole)
                raw_statistics[ObservableType.KineticEnergy][current_step] = \
                    state.getKineticEnergy().value_in_unit(simtk_unit.kilojoules_per_mole)
                raw_statistics[ObservableType.Volume][current_step] = \
                    state.getPeriodicBoxVolume().value_in_unit(simtk_unit.angstrom ** 3)

                if self.save_rolling_statistics:

                    self._write_statistics_array(raw_statistics, current_step, temperature,
                                                 pressure, degrees_of_freedom, total_mass)

                current_step_count += steps_to_take
                current_step += 1

        except Exception as e:

            formatted_exception = f'{traceback.format_exception(None, e, e.__traceback__)}'

            result = PropertyEstimatorException(directory=directory,
                                                message=f'The simulation failed unexpectedly: '
                                                        f'{formatted_exception}')

        # Create a checkpoint file.
        state = context.getState(getPositions=True,
                                 getEnergy=True,
                                 getVelocities=True,
                                 getForces=True,
                                 getParameters=True,
                                 enforcePeriodicBox=self.enable_pbc)

        state_xml = openmm.XmlSerializer.serialize(state)

        with open(self._checkpoint_path, 'w') as file:
            file.write(state_xml)

        # Make sure to close the open trajectory stream.
        trajectory_file_object.close()

        # Save the final statistics
        self._write_statistics_array(raw_statistics, current_step, temperature,
                                     pressure, degrees_of_freedom, total_mass)

        if isinstance(result, PropertyEstimatorException):
            return result

        # Move the trajectory and statistics files to their
        # final location.
        os.replace(self._temporary_trajectory_path, self.trajectory_file_path)

        if not os.path.isfile(self.statistics_file_path):
            os.replace(self._temporary_statistics_path, self.statistics_file_path)
        else:

            existing_statistics = StatisticsArray.from_pandas_csv(self.statistics_file_path)
            current_statistics = StatisticsArray.from_pandas_csv(self._temporary_statistics_path)

            concatenated_statistics = StatisticsArray.join(existing_statistics,
                                                           current_statistics)

            concatenated_statistics.to_pandas_csv(self.statistics_file_path)

        # Save out the final positions.
        final_state = context.getState(getPositions=True)
        positions = final_state.getPositions()
        topology.setPeriodicBoxVectors(final_state.getPeriodicBoxVectors())

        self.output_coordinate_file = os.path.join(directory, 'output.pdb')

        with open(self.output_coordinate_file, 'w+') as configuration_file:
            app.PDBFile.writeFile(topology, positions, configuration_file)

        logging.info(f'Simulation performed in the {str(self.ensemble)} ensemble: {self._id}')
        return self._get_output_dictionary()
