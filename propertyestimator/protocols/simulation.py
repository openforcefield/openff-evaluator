"""
A collection of protocols for running molecular simulations.
"""
import logging
from os import path

from simtk import unit, openmm
from simtk.openmm import app

from propertyestimator.thermodynamics import ThermodynamicState, Ensemble
from propertyestimator.utils import statistics
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import setup_platform_with_resources
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

    @protocol_output(str)
    def output_coordinate_file(self):
        """The file path to the minimised coordinates."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._input_coordinate_file = None

        self._system_path = None
        self._system = None

        self._tolerance = 10*unit.kilojoules_per_mole
        self._max_iterations = 0

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

        simulation.minimizeEnergy(self._tolerance, self._max_iterations)

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
