"""
A collection of protocols for running molecular simulations.
"""
import logging
import os

import numpy as np
import yaml
from simtk import openmm, unit as simtk_unit
from simtk.openmm import app

from propertyestimator import unit
from propertyestimator.thermodynamics import ThermodynamicState, Ensemble
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import setup_platform_with_resources, openmm_quantity_to_pint, \
    pint_quantity_to_openmm
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import StatisticsArray, ObservableType
from propertyestimator.utils.utils import safe_unlink
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

        self._tolerance = 10*unit.kilojoules / unit.mole
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
        integrator = openmm.VerletIntegrator(0.002 * simtk_unit.picoseconds)
        simulation = app.Simulation(input_pdb_file.topology, self._system, integrator, platform)

        box_vectors = input_pdb_file.topology.getPeriodicBoxVectors()

        if box_vectors is None:
            box_vectors = simulation.system.getDefaultPeriodicBoxVectors()

        simulation.context.setPeriodicBoxVectors(*box_vectors)
        simulation.context.setPositions(input_pdb_file.positions)

        simulation.minimizeEnergy(pint_quantity_to_openmm(self._tolerance), self._max_iterations)

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

        self._context = None
        self._integrator = None

    def execute(self, directory, available_resources):

        # We handle most things in OMM units here.
        temperature = pint_quantity_to_openmm(self._thermodynamic_state.temperature)
        pressure = pint_quantity_to_openmm(None if self._ensemble == Ensemble.NVT else
                                           self._thermodynamic_state.pressure)

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
        thermostat_friction = pint_quantity_to_openmm(self._thermostat_friction)
        timestep = pint_quantity_to_openmm(self._timestep)

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
        temperature = openmm_quantity_to_pint(temperature)
        pressure = openmm_quantity_to_pint(pressure)

        beta = 1.0 / (unit.boltzmann_constant * temperature)

        potential_energies = np.array(raw_statistics[ObservableType.PotentialEnergy]) * unit.kilojoules / unit.mole
        kinetic_energies = np.array(raw_statistics[ObservableType.KineticEnergy]) * unit.kilojoules / unit.mole
        volumes = np.array(raw_statistics[ObservableType.Volume]) * unit.angstrom ** 3

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
        result = None

        try:

            while current_step_count < self.steps:

                steps_to_take = min(self._output_frequency, self.steps - current_step_count)
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
                raw_statistics[ObservableType.PotentialEnergy].append(state.getPotentialEnergy().
                                                                      value_in_unit(simtk_unit.kilojoules_per_mole))
                raw_statistics[ObservableType.KineticEnergy].append(state.getKineticEnergy().
                                                                    value_in_unit(simtk_unit.kilojoules_per_mole))
                raw_statistics[ObservableType.Volume].append(state.getPeriodicBoxVolume().
                                                             value_in_unit(simtk_unit.angstrom ** 3))

                self._write_statistics_array(raw_statistics, temperature, pressure,
                                             degrees_of_freedom, total_mass)

                current_step_count += steps_to_take

        except Exception as e:

            result = PropertyEstimatorException(directory=directory,
                                                message='Simulation failed: {}'.format(e))

        # Make sure to close the open trajectory stream.
        trajectory_file_object.close()

        if isinstance(result, PropertyEstimatorException):
            return result

        # Create a checkpoint file.
        state = context.getState(getPositions=True,
                                 getEnergy=True,
                                 getVelocities=False,
                                 getForces=False,
                                 getParameters=False,
                                 enforcePeriodicBox=self.enable_pbc)

        state_xml = openmm.XmlSerializer.serialize(state)

        with open(self._checkpoint_path, 'w') as file:
            file.write(state_xml)

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
            app.PDBFile.writeFile(topology, positions, configuration_file)

        logging.info(f'Simulation performed in the {str(self._ensemble)} ensemble: {self._id}')
        return self._get_output_dictionary()
