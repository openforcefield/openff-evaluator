"""
A collection of protocols for reweighting cached simulation data.
"""

import json
import logging
import sys
from os import path

import numpy as np
import pymbar
from simtk import unit

from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import setup_platform_with_resources
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.serialization import TypedJSONDecoder
from propertyestimator.utils.statistics import bootstrap, StatisticsArray, ObservableType
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class UnpackStoredSimulationData(BaseProtocol):
    """Loads a `StoredSimulationData` object from disk,
    and makes its attributes easily accessible to other protocols.
    """

    @protocol_input(tuple)
    def simulation_data_path(self):
        """A tuple which contains both the path to the simulation data directory,
        and the force field which was used to generate the stored data."""
        pass

    @protocol_output(Substance)
    def substance(self):
        """The substance which was stored."""
        pass

    @protocol_output(int)
    def total_number_of_molecules(self):
        """The total number of molecules in the stored system."""
        pass

    @protocol_output(ThermodynamicState)
    def thermodynamic_state(self):
        """The thermodynamic state which was stored."""
        pass

    @protocol_output(float)
    def statistical_inefficiency(self):
        """The statistical inefficiency of the stored data."""
        pass

    @protocol_output(str)
    def coordinate_file_path(self):
        """A path to the stored simulation trajectory."""
        pass

    @protocol_output(str)
    def trajectory_file_path(self):
        """A path to the stored simulation trajectory."""
        pass

    @protocol_output(str)
    def statistics_file_path(self):
        """A path to the stored simulation statistics array."""
        pass

    @protocol_output(str)
    def force_field_path(self):
        """A path to the force field parameters used to generate
        the stored data."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new UnpackStoredSimulationData object."""
        super().__init__(protocol_id)

        self._simulation_data_path = None

        self._substance = None
        self._total_number_of_molecules = None

        self._thermodynamic_state = None

        self._statistical_inefficiency = None

        self._coordinate_file_path = None
        self._trajectory_file_path = None

        self._statistics_file_path = None

        self._force_field_path = None

    def execute(self, directory, available_resources):

        if len(self._simulation_data_path) != 2:

            return PropertyEstimatorException(directory=directory,
                                              message='The simulation data path should be a tuple '
                                                      'of a path to the data directory, and a path '
                                                      'to the force field used to generate it.')

        data_directory = self._simulation_data_path[0]
        force_field_path = self._simulation_data_path[1]

        if not path.isdir(data_directory):

            return PropertyEstimatorException(directory=directory,
                                              message='The path to the data directory'
                                                      'is invalid: {}'.format(data_directory))

        if not path.isfile(force_field_path):

            return PropertyEstimatorException(directory=directory,
                                              message='The path to the force field'
                                                      'is invalid: {}'.format(force_field_path))

        data_object = None

        with open(path.join(data_directory, 'data.json'), 'r') as file:
            data_object = json.load(file, cls=TypedJSONDecoder)

        self._substance = data_object.substance
        self._total_number_of_molecules = data_object.total_number_of_molecules

        self._thermodynamic_state = data_object.thermodynamic_state

        self._statistical_inefficiency = data_object.statistical_inefficiency

        self._coordinate_file_path = path.join(data_directory, data_object.coordinate_file_name)
        self._trajectory_file_path = path.join(data_directory, data_object.trajectory_file_name)

        self._statistics_file_path = path.join(data_directory, data_object.statistics_file_name)

        self._force_field_path = force_field_path

        return self._get_output_dictionary()


@register_calculation_protocol()
class ConcatenateTrajectories(BaseProtocol):
    """A protocol which concatenates multiple trajectories into
    a single one.
    """

    @protocol_input(list)
    def input_coordinate_paths(self):
        """A list of paths to the starting coordinates for each of the trajectories."""
        pass

    @protocol_input(list)
    def input_trajectory_paths(self):
        """A list of paths to the trajectories to concatenate."""
        pass

    @protocol_output(str)
    def output_coordinate_path(self):
        """The path the coordinate file which contains the topology of
        the concatenated trajectory."""
        pass

    @protocol_output(str)
    def output_trajectory_path(self):
        """The path to the concatenated trajectory."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new AddValues object."""
        super().__init__(protocol_id)

        self._input_coordinate_paths = None
        self._input_trajectory_paths = None

        self._output_coordinate_path = None
        self._output_trajectory_path = None

    def execute(self, directory, available_resources):

        import mdtraj

        if len(self._input_coordinate_paths) != len(self._input_trajectory_paths):

            return PropertyEstimatorException(directory=directory, message='There should be the same number of '
                                                                           'coordinate and trajectory paths.')

        if len(self._input_trajectory_paths) == 0:

            return PropertyEstimatorException(directory=directory, message='No trajectories were '
                                                                           'given to concatenate.')

        trajectories = []

        for coordinate_path, trajectory_path in zip(self._input_coordinate_paths,
                                                    self._input_trajectory_paths):

            self._output_coordinate_path = self._output_coordinate_path or coordinate_path
            trajectories.append(mdtraj.load_dcd(trajectory_path, coordinate_path))

        output_trajectory = trajectories[0] if len(trajectories) == 1 else mdtraj.join(trajectories, True, False)

        self._output_trajectory_path = path.join(directory, 'output_trajectory.dcd')
        output_trajectory.save_dcd(self._output_trajectory_path)

        return self._get_output_dictionary()


@register_calculation_protocol()
class CalculateReducedPotentialOpenMM(BaseProtocol):
    """Calculates the reduced potential for a given
    set of configurations.
    """

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self):
        pass

    @protocol_input(str)
    def system_path(self):
        pass

    @protocol_input(str)
    def coordinate_file_path(self):
        pass

    @protocol_input(str)
    def trajectory_file_path(self):
        pass

    @protocol_input(bool)
    def high_precision(self):
        pass

    @protocol_output(str)
    def statistics_file_path(self):
        """A file path to the StatisticsArray file which contains the reduced potentials."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new UnpackStoredSimulationData object."""
        super().__init__(protocol_id)

        self._thermodynamic_state = None

        self._system_path = None
        self._system = None

        self._coordinate_file_path = None
        self._trajectory_file_path = None

        self._statistics_file_path = None

        self._high_precision = False

    def execute(self, directory, available_resources):

        import openmmtools
        import mdtraj

        from simtk.openmm import XmlSerializer

        with open(self._system_path, 'rb') as file:
            self._system = XmlSerializer.deserialize(file.read().decode())

        trajectory = mdtraj.load_dcd(self._trajectory_file_path, self._coordinate_file_path)
        self._system.setDefaultPeriodicBoxVectors(*trajectory.openmm_boxes(0))

        openmm_state = openmmtools.states.ThermodynamicState(system=self._system,
                                                             temperature=self._thermodynamic_state.temperature,
                                                             pressure=self._thermodynamic_state.pressure)

        integrator = openmmtools.integrators.VelocityVerletIntegrator(0.01*unit.femtoseconds)

        # Setup the requested platform:
        platform = setup_platform_with_resources(available_resources, self._high_precision)

        context_cache = openmmtools.cache.ContextCache(platform)
        openmm_context, openmm_context_integrator = context_cache.get_context(openmm_state,
                                                                              integrator)

        reduced_potentials = np.zeros(trajectory.n_frames)

        for frame_index in range(trajectory.n_frames):

            # positions = trajectory.openmm_positions(frame_index)
            positions = trajectory.xyz[frame_index]
            box_vectors = trajectory.openmm_boxes(frame_index)

            openmm_context.setPeriodicBoxVectors(*box_vectors)
            openmm_context.setPositions(positions)

            # set box vectors
            reduced_potentials[frame_index] = openmm_state.reduced_potential(openmm_context)

        statistics_array = StatisticsArray()
        statistics_array[ObservableType.ReducedPotential] = reduced_potentials * unit.dimensionless

        self._statistics_file_path = path.join(directory, 'statistics.csv')
        statistics_array.to_pandas_csv(self._statistics_file_path)

        return self._get_output_dictionary()


@register_calculation_protocol()
class ReweightWithMBARProtocol(BaseProtocol):
    """Reweights a set of observables using MBAR to calculate
    the average value of the observables at a different state
    than they were originally measured.
    """

    @protocol_input(list)
    def reference_reduced_potentials(self):
        """A list of paths to the reduced potentials of each reference state."""
        pass

    @protocol_input(list)
    def reference_observables(self):
        """A list of the observables to be reweighted from each reference state."""
        pass

    @protocol_input(list)
    def target_reduced_potentials(self):
        """A list of paths to the reduced potentials of the target state."""
        pass

    @protocol_input(bool)
    def bootstrap_uncertainties(self):
        """If true, bootstrapping will be used to estimated the total uncertainty"""
        pass

    @protocol_input(int)
    def bootstrap_iterations(self):
        """The number of bootstrap iterations to perform if bootstraped
        uncertainties have been requested"""
        pass

    @protocol_input(float)
    def bootstrap_sample_size(self):
        """The relative bootstrap sample size to use if bootstraped
        uncertainties have been requested"""
        pass

    @protocol_input(int)
    def required_effective_samples(self):
        """The minimum number of MBAR effective samples for the reweighted
        value to be trusted. If this minimum is not met then the uncertainty
        will be set to sys.float_info.max"""
        pass

    @protocol_output(EstimatedQuantity)
    def value(self):
        pass

    def __init__(self, protocol_id):
        """Constructs a new ReweightWithMBARProtocol object."""
        super().__init__(protocol_id)

        self._reference_reduced_potentials = None
        self._reference_observables = None

        self._target_reduced_potentials = None

        self._bootstrap_uncertainties = False
        self._bootstrap_iterations = 1
        self._bootstrap_sample_size = 1.0

        self._required_effective_samples = 50

        self._value = None

    def execute(self, directory, available_resources):

        if len(self._reference_observables) == 0:

            return PropertyEstimatorException(directory=directory,
                                              message='There were no observables to reweight.')

        if not isinstance(self._reference_observables[0], unit.Quantity):

            return PropertyEstimatorException(directory=directory,
                                              message='The reference_observables input should be'
                                                      'a list of unit.Quantity wrapped ndarray\'s.')

        observables = self._prepare_observables_array(self._reference_observables)
        observable_unit = self._reference_observables[0].unit

        reference_reduced_potentials = []
        target_reduced_potentials = []

        for file_path in self._reference_reduced_potentials:

            statistics_array = StatisticsArray.from_pandas_csv(file_path)
            reduced_potentials = statistics_array[ObservableType.ReducedPotential]

            reference_reduced_potentials.append(reduced_potentials.value_in_unit(unit.dimensionless))

        for file_path in self._target_reduced_potentials:

            statistics_array = StatisticsArray.from_pandas_csv(file_path)
            reduced_potentials = statistics_array[ObservableType.ReducedPotential]

            target_reduced_potentials.append(reduced_potentials.value_in_unit(unit.dimensionless))

        if self._bootstrap_uncertainties:

            reference_potentials = np.transpose(np.array(reference_reduced_potentials))
            target_potentials = np.transpose(np.array(target_reduced_potentials))

            frame_counts = np.array([len(observable) for observable in self._reference_observables])

            # Construct an mbar object to get out the number of effective samples.
            mbar = pymbar.MBAR(reference_reduced_potentials,
                               frame_counts, verbose=False, relative_tolerance=1e-12)

            effective_samples = mbar.computeEffectiveSampleNumber().max()

            value, uncertainty = bootstrap(self._bootstrap_function,
                                           self._bootstrap_iterations,
                                           self._bootstrap_sample_size,
                                           frame_counts,
                                           reference_reduced_potentials=reference_potentials,
                                           target_reduced_potentials=target_potentials,
                                           observables=np.transpose(observables))

            if effective_samples < self._required_effective_samples:

                logging.info(f'{self.id}: There was not enough effective samples '
                             f'to reweight - {effective_samples} < {self._required_effective_samples}')

                uncertainty = sys.float_info.max

            self._value = EstimatedQuantity(value * observable_unit,
                                            uncertainty * observable_unit,
                                            self.id)

        else:

            values, uncertainties, effective_samples = self._reweight_observables(reference_reduced_potentials,
                                                                                  target_reduced_potentials,
                                                                                  observables=observables)

            uncertainty = uncertainties['observables']

            if effective_samples < self._required_effective_samples:

                logging.info(f'{self.id}: There was not enough effective samples '
                             f'to reweight - {effective_samples} < {self._required_effective_samples}')

                uncertainty = sys.float_info.max

            self._value = EstimatedQuantity(values['observables'] * observable_unit,
                                            uncertainty * observable_unit,
                                            self.id)

        return self._get_output_dictionary()

    @staticmethod
    def _prepare_observables_array(reference_observables):
        """Takes a list of reference observables, and concatenates them
        into a single Quantity wrapped numpy array.

        Parameters
        ----------
        reference_observables: List of unit.Quantity
            A list of observables for each reference state,
            which each observable is a Quantity wrapped numpy
            array.

        Returns
        -------
        np.ndarray
            A unitless numpy array of all of the observables.
        """
        frame_counts = np.array([len(observable) for observable in reference_observables])
        number_of_configurations = frame_counts.sum()

        observable_dimensions = 1 if len(reference_observables[0].shape) == 1 else reference_observables[0].shape[1]
        observable_unit = reference_observables[0].unit

        observables = np.zeros((observable_dimensions, number_of_configurations))

        # Build up an array which contains the observables from all
        # of the reference states.
        for index_k, observables_k in enumerate(reference_observables):

            start_index = np.array(frame_counts[0:index_k]).sum()

            for index in range(0, frame_counts[index_k]):

                value = observables_k[index].value_in_unit(observable_unit)

                if not isinstance(value, np.ndarray):
                    observables[0][start_index + index] = value
                    continue

                for dimension in range(observable_dimensions):
                    observables[dimension][start_index + index] = value[dimension]

        return observables

    def _bootstrap_function(self, reference_reduced_potentials, target_reduced_potentials, **reference_observables):
        """The function which will be called after each bootstrap
        iteration, if bootstrapping is being employed to estimated
        the reweighting uncertainty.

        Parameters
        ----------
        reference_reduced_potentials
        target_reduced_potentials
        reference_observables

        Returns
        -------
        float
            The bootstrapped value,
        """
        assert len(reference_observables) == 1

        transposed_observables = {}

        for key in reference_observables:
            transposed_observables[key] = np.transpose(reference_observables[key])

        values, _, _ = self._reweight_observables(np.transpose(reference_reduced_potentials),
                                                               np.transpose(target_reduced_potentials),
                                                               **transposed_observables)

        return next(iter(values.values()))

    def _reweight_observables(self, reference_reduced_potentials, target_reduced_potentials, **reference_observables):
        """Reweights a set of reference observables to
        the target state.

        Returns
        -------
        dict of str and float or list of float
            The reweighted values.
        dict of str and float or list of float
            The MBAR calculated uncertainties in the reweighted values.
        int
            The number of effective samples.
        """

        frame_counts = np.array([len(observable) for observable in self._reference_observables])

        # Construct the mbar object.
        mbar = pymbar.MBAR(reference_reduced_potentials,
                           frame_counts, verbose=False, relative_tolerance=1e-12)

        max_effective_samples = mbar.computeEffectiveSampleNumber().max()

        values = {}
        uncertainties = {}

        for observable_key in reference_observables:

            observable = reference_observables[observable_key]
            observable_dimensions = observable.shape[0]

            if observable_dimensions == 1:

                results = mbar.computeExpectations(observable,
                                                   target_reduced_potentials,
                                                   state_dependent=True)

                values[observable_key] = results[0][0]
                uncertainties[observable_key] = results[1][0]

            else:

                value = []
                uncertainty = []

                for dimension in range(observable_dimensions):

                    results = mbar.computeExpectations(observable[dimension],
                                                       target_reduced_potentials,
                                                       state_dependent=True)

                    value.append(results[0][0])
                    uncertainty.append(results[1][0])

                values[observable_key] = np.array(value)
                uncertainties[observable_key] = np.array(uncertainty)

        return values, uncertainties, max_effective_samples
