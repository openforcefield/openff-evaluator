"""
A collection of protocols for reweighting cached simulation data.
"""

from os import path

import numpy as np
import pymbar
from scipy.special import logsumexp

from propertyestimator import unit
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.openmm import pint_quantity_to_openmm, setup_platform_with_resources, disable_pbc
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import bootstrap, StatisticsArray, ObservableType
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


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
        """Constructs a new ConcatenateTrajectories object."""
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

        output_trajectory = trajectories[0] if len(trajectories) == 1 else mdtraj.join(trajectories, False, False)

        self._output_trajectory_path = path.join(directory, 'output_trajectory.dcd')
        output_trajectory.save_dcd(self._output_trajectory_path)

        return self._get_output_dictionary()


@register_calculation_protocol()
class ConcatenateStatistics(BaseProtocol):
    """A protocol which concatenates multiple trajectories into
    a single one.
    """

    @protocol_input(list)
    def input_statistics_paths(self):
        """A list of paths to the different statistics arrays."""
        pass

    @protocol_output(str)
    def output_statistics_path(self):
        """The path the csv file which contains the concatenated statistics."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new ConcatenateStatistics object."""
        super().__init__(protocol_id)

        self._input_statistics_paths = None
        self._output_statistics_path = None

    def execute(self, directory, available_resources):

        if len(self._input_statistics_paths) == 0:

            return PropertyEstimatorException(directory=directory, message='No statistics arrays were '
                                                                           'given to concatenate.')

        arrays = [StatisticsArray.from_pandas_csv(file_path) for
                  file_path in self._input_statistics_paths]

        if len(arrays) > 1:
            output_array = StatisticsArray.join(*arrays)
        else:
            output_array = arrays[0]

        self._output_statistics_path = path.join(directory, 'output_statistics.csv')
        output_array.to_pandas_csv(self._output_statistics_path)

        return self._get_output_dictionary()


@register_calculation_protocol()
class CalculateReducedPotentialOpenMM(BaseProtocol):
    """Calculates the reduced potential for a given
    set of configurations.
    """

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self):
        """The state to calculate the reduced potential at."""
        pass

    @protocol_input(str)
    def system_path(self):
        """The path to the system object which describes the systems
        potential energy function."""
        pass

    @protocol_input(bool)
    def enable_pbc(self):
        """If true, periodic boundary conditions will be enabled."""
        pass

    @protocol_input(str)
    def coordinate_file_path(self):
        """The path to the coordinate file which contains topology
        information about the system."""
        pass

    @protocol_input(str)
    def trajectory_file_path(self):
        """The path to the trajectory file which contains the
        configurations to calculate the energies of."""
        pass

    @protocol_input(str)
    def kinetic_energies_path(self):
        """The file path to a statistics array which contain the kinetic energies
        of each frame in the trajectory."""
        pass

    @protocol_input(bool)
    def high_precision(self):
        """If true, OpenMM will be run in double precision mode."""
        pass

    @protocol_input(bool)
    def use_internal_energy(self):
        """If true the internal energy, rather than the potential energy
        will be used when calculating the reduced potential. This is required
        when reweighting properties which depend on the total energy, such as
        enthalpy."""
        pass

    @protocol_output(str)
    def statistics_file_path(self):
        """A file path to the StatisticsArray file which contains the reduced potentials,
        and the potential, kinetic and total energies and enthalpies evaluated at the specified
        state and using the specified system object."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new CalculateReducedPotentialOpenMM object."""
        super().__init__(protocol_id)

        self._thermodynamic_state = None

        self._system_path = None
        self._system = None
        self._enable_pbc = True

        self._coordinate_file_path = None
        self._trajectory_file_path = None

        self._kinetic_energies_path = None
        self._use_internal_energy = False

        self._statistics_file_path = None

        self._high_precision = False

    def execute(self, directory, available_resources):

        import openmmtools
        import mdtraj

        from simtk import openmm, unit as simtk_unit
        from simtk.openmm import XmlSerializer

        trajectory = mdtraj.load_dcd(self._trajectory_file_path, self._coordinate_file_path)

        with open(self._system_path, 'rb') as file:
            self._system = XmlSerializer.deserialize(file.read().decode())

        temperature = pint_quantity_to_openmm(self._thermodynamic_state.temperature)
        pressure = pint_quantity_to_openmm(self._thermodynamic_state.pressure)

        if self._enable_pbc:
            self._system.setDefaultPeriodicBoxVectors(*trajectory.openmm_boxes(0))
        else:
            pressure = None

        openmm_state = openmmtools.states.ThermodynamicState(system=self._system,
                                                             temperature=temperature,
                                                             pressure=pressure)

        integrator = openmmtools.integrators.VelocityVerletIntegrator(0.01*simtk_unit.femtoseconds)

        # Setup the requested platform:
        platform = setup_platform_with_resources(available_resources, self._high_precision)
        openmm_system = openmm_state.get_system(True, True)

        if not self._enable_pbc:
            disable_pbc(openmm_system)

        openmm_context = openmm.Context(openmm_system, integrator, platform)

        potential_energies = np.zeros(trajectory.n_frames)
        reduced_potentials = np.zeros(trajectory.n_frames)

        for frame_index in range(trajectory.n_frames):

            if self._enable_pbc:
                box_vectors = trajectory.openmm_boxes(frame_index)
                openmm_context.setPeriodicBoxVectors(*box_vectors)

            positions = trajectory.xyz[frame_index]
            openmm_context.setPositions(positions)

            potential_energy = openmm_context.getState(getEnergy=True).getPotentialEnergy()

            potential_energies[frame_index] = potential_energy.value_in_unit(simtk_unit.kilojoule_per_mole)
            reduced_potentials[frame_index] = openmm_state.reduced_potential(openmm_context)

        kinetic_energies = StatisticsArray.from_pandas_csv(self._kinetic_energies_path)[ObservableType.KineticEnergy]

        statistics_array = StatisticsArray()
        statistics_array[ObservableType.PotentialEnergy] = potential_energies * unit.kilojoule / unit.mole
        statistics_array[ObservableType.KineticEnergy] = kinetic_energies
        statistics_array[ObservableType.ReducedPotential] = reduced_potentials * unit.dimensionless

        statistics_array[ObservableType.TotalEnergy] = (statistics_array[ObservableType.PotentialEnergy] +
                                                        statistics_array[ObservableType.KineticEnergy])

        statistics_array[ObservableType.Enthalpy] = (statistics_array[ObservableType.ReducedPotential] *
                                                     self._thermodynamic_state.inverse_beta + kinetic_energies)

        if self._use_internal_energy:
            statistics_array[ObservableType.ReducedPotential] += kinetic_energies * self._thermodynamic_state.beta

        self._statistics_file_path = path.join(directory, 'statistics.csv')
        statistics_array.to_pandas_csv(self._statistics_file_path)

        return self._get_output_dictionary()


@register_calculation_protocol()
class BaseMBARProtocol(BaseProtocol):
    """Reweights a set of observables using MBAR to calculate
    the average value of the observables at a different state
    than they were originally measured.
    """

    @protocol_input(list)
    def reference_reduced_potentials(self):
        """A list of paths to the reduced potentials of each reference state."""
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
        """The reweighted average value of the observable at the target state."""
        pass

    @protocol_output(int)
    def effective_samples(self):
        """The number of effective samples which were reweighted."""
        pass

    @protocol_output(list)
    def effective_sample_indices(self):
        """The indices of those samples which have a non-zero weight."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new BaseMBARProtocol object."""
        super().__init__(protocol_id)

        self._reference_reduced_potentials = None
        self._reference_observables = None

        self._target_reduced_potentials = None

        self._bootstrap_uncertainties = False
        self._bootstrap_iterations = 1
        self._bootstrap_sample_size = 1.0

        self._required_effective_samples = 50

        self._value = None

        self._effective_samples = 0
        self._effective_sample_indices = None

    def execute(self, directory, available_resources):

        if len(self._reference_observables) == 0:

            return PropertyEstimatorException(directory=directory,
                                              message='There were no observables to reweight.')

        if not isinstance(self._reference_observables[0], unit.Quantity):

            return PropertyEstimatorException(directory=directory,
                                              message='The reference_observables input should be'
                                                      'a list of unit.Quantity wrapped ndarray\'s.')

        observables = self._prepare_observables_array(self._reference_observables)
        observable_unit = self._reference_observables[0].units

        if self._bootstrap_uncertainties:
            error = self._execute_with_bootstrapping(observable_unit, observables=observables)
        else:
            error = self._execute_without_bootstrapping(observable_unit, observables=observables)

        if error is not None:

            error.directory = directory
            return error

        return self._get_output_dictionary()

    def _load_reduced_potentials(self):
        """Loads the target and reference reduced potentials
        from the specified statistics files.

        Returns
        -------
        numpy.ndarray
            The reference reduced potentials array with dtype=double and
            shape=(1,)
        numpy.ndarray
            The target reduced potentials array with dtype=double and
            shape=(1,)
        """

        reference_reduced_potentials = []
        target_reduced_potentials = []

        # Load in the reference reduced potentials.
        for file_path in self._reference_reduced_potentials:

            statistics_array = StatisticsArray.from_pandas_csv(file_path)
            reduced_potentials = statistics_array[ObservableType.ReducedPotential]

            reference_reduced_potentials.append(reduced_potentials.to(unit.dimensionless).magnitude)

        # Load in the target reduced potentials.
        if len(target_reduced_potentials) > 1:

            raise ValueError('This protocol currently only supports reweighting to '
                             'a single target state.')

        for file_path in self._target_reduced_potentials:

            statistics_array = StatisticsArray.from_pandas_csv(file_path)
            reduced_potentials = statistics_array[ObservableType.ReducedPotential]

            target_reduced_potentials.append(reduced_potentials.to(unit.dimensionless).magnitude)

        reference_reduced_potentials = np.array(reference_reduced_potentials)
        target_reduced_potentials = np.array(target_reduced_potentials)

        return reference_reduced_potentials, target_reduced_potentials

    def _execute_with_bootstrapping(self, observable_unit, **observables):
        """Calculates the average reweighted observables at the target state,
        using bootstrapping to estimate uncertainties.

        Parameters
        ----------
        observable_unit: propertyestimator.unit.Unit:
            The expected unit of the reweighted observable.
        observables: dict of str and numpy.ndarray
            The observables to reweight which have been stripped of their units.

        Returns
        -------
        PropertyEstimatorException, optional
            None if the method executed normally, otherwise the exception that was raised.
        """

        reference_reduced_potentials, target_reduced_potentials = self._load_reduced_potentials()

        frame_counts = np.array([len(observable) for observable in self._reference_observables])

        # Construct a dummy mbar object to get out the number of effective samples.
        mbar = self._construct_mbar_object(reference_reduced_potentials)

        (self._effective_samples,
         effective_sample_indices) = self._compute_effective_samples(mbar, target_reduced_potentials)

        if self._effective_samples < self._required_effective_samples:

            return PropertyEstimatorException(message=f'{self.id}: There was not enough effective samples '
                                                      f'to reweight - {self._effective_samples} < '
                                                      f'{self._required_effective_samples}')

        # Transpose the observables ready for bootstrapping.
        reference_reduced_potentials = np.transpose(reference_reduced_potentials)
        target_reduced_potentials = np.transpose(target_reduced_potentials)

        transposed_observables = {}

        for observable_key in observables:
            transposed_observables[observable_key] = np.transpose(observables[observable_key])

        value, uncertainty = bootstrap(self._bootstrap_function,
                                       self._bootstrap_iterations,
                                       self._bootstrap_sample_size,
                                       frame_counts,
                                       reference_reduced_potentials=reference_reduced_potentials,
                                       target_reduced_potentials=target_reduced_potentials,
                                       **transposed_observables)

        self._effective_sample_indices = effective_sample_indices

        self._value = EstimatedQuantity(value * observable_unit,
                                        uncertainty * observable_unit,
                                        self.id)

    def _execute_without_bootstrapping(self, observable_unit, **observables):
        """Calculates the average reweighted observables at the target state,
        using the built-in pymbar method to estimate uncertainties.

        Parameters
        ----------
        observables: dict of str and numpy.ndarray
            The observables to reweight which have been stripped of their units.
        """

        if len(observables) > 1:

            raise ValueError('Currently only a single observable can be reweighted at'
                             'any one time.')

        reference_reduced_potentials, target_reduced_potentials = self._load_reduced_potentials()

        values, uncertainties, self._effective_samples = self._reweight_observables(reference_reduced_potentials,
                                                                                    target_reduced_potentials,
                                                                                    **observables)

        observable_key = next(iter(observables))
        uncertainty = uncertainties[observable_key]

        if self._effective_samples < self._required_effective_samples:

            return PropertyEstimatorException(message=f'{self.id}: There was not enough effective samples '
                                                      f'to reweight - {self._effective_samples} < '
                                                      f'{self._required_effective_samples}')

        self._value = EstimatedQuantity(values[observable_key] * observable_unit,
                                        uncertainty * observable_unit,
                                        self.id)

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
        observable_unit = reference_observables[0].units

        observables = np.zeros((observable_dimensions, number_of_configurations))

        # Build up an array which contains the observables from all
        # of the reference states.
        for index_k, observables_k in enumerate(reference_observables):

            start_index = np.array(frame_counts[0:index_k]).sum()

            for index in range(0, frame_counts[index_k]):

                value = observables_k[index].to(observable_unit).magnitude

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

    def _construct_mbar_object(self, reference_reduced_potentials):
        """Constructs a new `pymbar.MBAR` object for a given set of reference
        and target reduced potentials

        Parameters
        -------
        reference_reduced_potentials: numpy.ndarray
            The reference reduced potentials.

        Returns
        -------
        pymbar.MBAR
            The constructed `MBAR` object.
        """

        frame_counts = np.array([len(observables) for observables in self._reference_observables])

        # Construct the mbar object.
        mbar = pymbar.MBAR(reference_reduced_potentials,
                           frame_counts, verbose=False, relative_tolerance=1e-12)

        return mbar

    @staticmethod
    def _compute_effective_samples(mbar, target_reduced_potentials):
        """Compute the effective number of samples which contribute to the final
        reweighted estimate.

        Parameters
        ----------
        mbar: pymbar.MBAR
            The MBAR object which contains the sample weights.
        target_reduced_potentials: numpy.ndarray
            The target reduced potentials.

        Returns
        -------
        int
            The effective number of samples.
        list of int
            The indices of samples which have non-zero weights.
        """

        states_with_samples = (mbar.N_k > 0)

        log_ref_q_k = mbar.f_k[states_with_samples] - mbar.u_kn[states_with_samples].T
        log_denominator_n = logsumexp(log_ref_q_k, b=mbar.N_k[states_with_samples], axis=1)

        target_f_hat = -logsumexp(-target_reduced_potentials[:len(target_reduced_potentials)] -
                                  log_denominator_n, axis=1)

        log_tar_q_k = target_f_hat - target_reduced_potentials

        # Calculate the weights
        weights = np.exp(log_tar_q_k - log_denominator_n)

        effective_samples = 1.0 / np.sum(weights**2)

        effective_sample_indices = [index for index in range(weights.shape[1])
                                    if not np.isclose(weights[0][index], 0.0)]

        return effective_samples, effective_sample_indices

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

        # Construct the mbar object.
        mbar = self._construct_mbar_object(reference_reduced_potentials)

        (effective_samples,
         self._effective_sample_indices) = self._compute_effective_samples(mbar, target_reduced_potentials)

        values = {}
        uncertainties = {}

        for observable_key in reference_observables:

            reference_observable = reference_observables[observable_key]
            observable_dimensions = reference_observable.shape[0]

            values[observable_key] = np.zeros((observable_dimensions, 1))
            uncertainties[observable_key] = np.zeros((observable_dimensions, 1))

            for dimension in range(observable_dimensions):

                results = mbar.computeExpectations(reference_observable[dimension],
                                                   target_reduced_potentials,
                                                   state_dependent=True)

                values[observable_key][dimension] = results[0][-1]
                uncertainties[observable_key][dimension] = results[1][-1]

            if observable_dimensions == 1:
                values[observable_key] = values[observable_key][0][0].item()
                uncertainties[observable_key] = uncertainties[observable_key][0][0].item()

        return values, uncertainties, effective_samples


@register_calculation_protocol()
class ReweightStatistics(BaseMBARProtocol):
    """Reweights a set of observables from a `StatisticsArray` using MBAR.
    """

    @protocol_input(list)
    def statistics_paths(self):
        """The file paths to the statistics array which contains the observables
        of interest from each state. If the observable of interest is dependant
        on the changing variable (e.g. the potential energy) then this must be a
        path to the observable re-evaluated at the new state."""
        pass

    @protocol_input(ObservableType)
    def statistics_type(self):
        """The type of observable to reweight."""
        pass

    @protocol_input(list)
    def frame_counts(self):
        """An optional list which describes how many of the statistics in the
        array belong to each reference state. If this input is used, only a
        single file path should be passed to the `statistics_paths` input."""
        pass

    def __init__(self, protocol_id):
        """Constructs a new ReweightWithMBARProtocol object."""
        super().__init__(protocol_id)

        self._statistics_paths = None
        self._statistics_type = None

        self._frame_counts = []

    def execute(self, directory, available_resources):

        if self._statistics_paths is None or len(self._statistics_paths) == 0:
            return PropertyEstimatorException(directory, 'No statistics paths were provided.')

        if len(self._frame_counts) > 0 and len(self._statistics_paths) != 1:
            return PropertyEstimatorException(directory, 'The frame counts input can only be used when only'
                                                         'a single path is passed to the `statistics_paths`'
                                                         'input.')

        if self._statistics_type == ObservableType.KineticEnergy:
            return PropertyEstimatorException(directory, f'Kinetic energies cannot be reweighted.')

        statistics_arrays = [StatisticsArray.from_pandas_csv(file_path) for file_path in self._statistics_paths]

        self._reference_observables = []

        if len(self._frame_counts) > 0:

            statistics_array = statistics_arrays[0]
            current_index = 0

            for frame_count in self._frame_counts:

                if frame_count <= 0:
                    return PropertyEstimatorException(directory, 'The frame counts must be > 0.')

                observables = statistics_array[self._statistics_type][current_index:current_index + frame_count]
                self._reference_observables.append(observables)

                current_index += frame_count

        else:

            for statistics_array in statistics_arrays:

                observables = statistics_array[self._statistics_type]
                self._reference_observables.append(observables)

        return super(ReweightStatistics, self).execute(directory, available_resources)
