"""
A collection of protocols for running analysing the results of molecular simulations.
"""

import logging
from os import path

import numpy as np

from propertyestimator import unit
from propertyestimator.utils import statistics, timeseries
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import StatisticsArray, bootstrap
from propertyestimator.workflow.decorators import protocol_input, protocol_output, MergeBehaviour
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class AveragePropertyProtocol(BaseProtocol):
    """An abstract base class for protocols which will calculate the
    average of a property and its uncertainty via bootstrapping.
    """

    @protocol_input(int, merge_behavior=MergeBehaviour.GreatestValue)
    def bootstrap_iterations(self):
        """The number of bootstrap iterations to perform."""
        pass

    @protocol_input(float, merge_behavior=MergeBehaviour.GreatestValue)
    def bootstrap_sample_size(self):
        """The relative sample size to use for bootstrapping."""
        pass

    @protocol_output(EstimatedQuantity)
    def value(self):
        """The averaged value."""
        pass

    @protocol_output(int)
    def equilibration_index(self):
        """The index in the data set after which the data is stationary."""
        pass

    @protocol_output(float)
    def statistical_inefficiency(self):
        """The statistical inefficiency in the data set."""
        pass

    @protocol_output(unit.Quantity)
    def uncorrelated_values(self):
        """The uncorrelated values which the average was calculated from."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._bootstrap_iterations = 250
        self._bootstrap_sample_size = 1.0

        self._value = None

        self._equilibration_index = None
        self._statistical_inefficiency = None

        self._uncorrelated_values = None

    def _bootstrap_function(self, **sample_kwargs):
        """The function to perform on the data set being sampled by
        bootstrapping.

        Parameters
        ----------
        sample_kwargs: dict of str and np.ndarray
            A key words dictionary of the bootstrap sample data, where the
            sample data is a numpy array of shape=(num_frames, num_dimensions)
            with dtype=float.

        Returns
        -------
        float
            The result of evaluating the data.
        """

        assert len(sample_kwargs) == 1
        sample_data = next(iter(sample_kwargs.values()))

        return sample_data.mean()

    def execute(self, directory, available_resources):
        return self._get_output_dictionary()


@register_calculation_protocol()
class AverageTrajectoryProperty(AveragePropertyProtocol):
    """An abstract base class for protocols which will calculate the
    average of a property from a simulation trajectory.
    """

    @protocol_input(str)
    def input_coordinate_file(self):
        """The file path to the starting coordinates of a trajectory."""
        pass

    @protocol_input(str)
    def trajectory_path(self):
        """The file path to the trajectory to average over."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._input_coordinate_file = None
        self._trajectory_path = None

    def execute(self, directory, available_resources):

        if self._trajectory_path is None:

            return PropertyEstimatorException(directory=directory,
                                              message='The AverageTrajectoryProperty protocol '
                                                       'requires a previously calculated trajectory')

        return self._get_output_dictionary()


@register_calculation_protocol()
class ExtractAverageStatistic(AveragePropertyProtocol):
    """Extracts the average value from a statistics file which was generated
    during a simulation.
    """

    @protocol_input(str)
    def statistics_path(self):
        """The file path to the trajectory to average over."""
        pass

    @protocol_input(statistics.ObservableType)
    def statistics_type(self):
        """The file path to the trajectory to average over."""
        pass

    @protocol_input(object)
    def divisor(self):
        """A divisor to divide the statistic by. This is useful
        if a statistic (such as enthalpy) needs to be normalised
        by the number of molecules."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._statistics_path = None
        self._statistics_type = statistics.ObservableType.PotentialEnergy

        self._divisor = 1

        self._statistics = None

    def execute(self, directory, available_resources):

        logging.info('Extracting {}: {}'.format(self._statistics_type, self.id))

        if self._statistics_path is None:

            return PropertyEstimatorException(directory=directory,
                                              message='The ExtractAverageStatistic protocol '
                                                       'requires a previously calculated statistics file')

        self._statistics = statistics.StatisticsArray.from_pandas_csv(self.statistics_path)

        if self._statistics_type not in self._statistics:

            return PropertyEstimatorException(directory=directory,
                                              message=f'The {self._statistics_path} statistics file contains no '
                                                      f'data of type {self._statistics_type}.')

        values = self._statistics[self._statistics_type]

        statistics_unit = values[0].units
        unitless_values = values.to(statistics_unit).magnitude

        divisor = self._divisor

        if isinstance(self._divisor, unit.Quantity):
            statistics_unit /= self._divisor.units
            divisor = self._divisor.magnitude

        unitless_values = np.array(unitless_values) / divisor

        unitless_values, self._equilibration_index, self._statistical_inefficiency = \
            timeseries.decorrelate_time_series(unitless_values)

        final_value, final_uncertainty = bootstrap(self._bootstrap_function,
                                                   self._bootstrap_iterations,
                                                   self._bootstrap_sample_size,
                                                   values=unitless_values)

        self._uncorrelated_values = unitless_values * statistics_unit

        self._value = EstimatedQuantity(final_value * statistics_unit,
                                        final_uncertainty * statistics_unit, self.id)

        logging.info('Extracted {}: {}'.format(self._statistics_type, self.id))

        return self._get_output_dictionary()


@register_calculation_protocol()
class ExtractUncorrelatedData(BaseProtocol):
    """An abstract base class for protocols which will subsample
    a data set, yielding only equilibrated, uncorrelated data.
    """

    @protocol_input(int)
    def equilibration_index(self):
        """The index in the data set after which the data is stationary."""
        pass

    @protocol_input(float)
    def statistical_inefficiency(self):
        """The statistical inefficiency in the data set."""
        pass

    @protocol_output(int)
    def number_of_uncorrelated_samples(self):
        """The number of uncorrelated samples."""
        pass

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._equilibration_index = None
        self._statistical_inefficiency = None

        self._number_of_uncorrelated_samples = None

    def execute(self, directory, available_resources):
        raise NotImplementedError


@register_calculation_protocol()
class ExtractUncorrelatedTrajectoryData(ExtractUncorrelatedData):
    """A protocol which will subsample frames from a trajectory, yielding only uncorrelated
    frames as determined from a provided statistical inefficiency and equilibration time.
    """

    @protocol_input(str)
    def input_coordinate_file(self):
        """The file path to the starting coordinates of a trajectory."""
        pass

    @protocol_input(str)
    def input_trajectory_path(self):
        """The file path to the trajectory to subsample."""
        pass

    @protocol_output(str)
    def output_trajectory_path(self):
        """The file path to the subsampled trajectory."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._input_coordinate_file = None
        self._input_trajectory_path = None

        self._output_trajectory_path = None

    @staticmethod
    def _yield_frame(file, topology, stride):
        """A generator which yields frames of a DCD trajectory.

        Parameters
        ----------
        file: mdtraj.DCDTrajectoryFile
            The file object being used to read the trajectory.
        topology: mdtraj.Topology
            The object which describes the topology of the trajectory.
        stride
            Only read every stride-th frame.

        Returns
        -------
        mdtraj.Trajectory
            A trajectory containing only a single frame.
        """

        while True:

            frame = file.read_as_traj(topology, n_frames=1, stride=stride)

            if len(frame) == 0:
                return

            yield frame

    def execute(self, directory, available_resources):

        import mdtraj
        from mdtraj.formats.dcd import DCDTrajectoryFile
        from mdtraj.utils import in_units_of

        logging.info('Subsampling trajectory: {}'.format(self.id))

        if self._input_trajectory_path is None:

            return PropertyEstimatorException(directory=directory,
                                              message='The ExtractUncorrelatedTrajectoryData protocol '
                                                       'requires a previously calculated trajectory')

        # Set the output path.
        self._output_trajectory_path = path.join(directory, 'uncorrelated_trajectory.dcd')

        # Load in the trajectories topology.
        topology = mdtraj.load_frame(self._input_coordinate_file, 0).topology
        # Parse the internal mdtraj distance unit. While private access is undesirable,
        # this is never publicly defined and I believe this route to be preferable
        # over hard coding this unit.
        base_distance_unit = mdtraj.Trajectory._distance_unit

        # Determine the stride that needs to be taken to yield uncorrelated frames.
        stride = timeseries.get_uncorrelated_stride(self._statistical_inefficiency)
        frame_count = 0

        with DCDTrajectoryFile('trajectory.dcd', 'r') as input_file:

            # Skip the equilibration configurations.
            if self._equilibration_index > 0:
                input_file.seek(self._equilibration_index)

            with DCDTrajectoryFile(self._output_trajectory_path, 'w') as output_file:

                for frame in self._yield_frame(input_file, topology, stride):

                    output_file.write(
                        xyz=in_units_of(frame.xyz, base_distance_unit, output_file.distance_unit),
                        cell_lengths=in_units_of(frame.unitcell_lengths, base_distance_unit, output_file.distance_unit),
                        cell_angles=frame.unitcell_angles[0]
                    )

                    frame_count += 1

        self._number_of_uncorrelated_samples = frame_count

        logging.info('Trajectory subsampled: {}'.format(self.id))

        return self._get_output_dictionary()


@register_calculation_protocol()
class ExtractUncorrelatedStatisticsData(ExtractUncorrelatedData):
    """A protocol which will subsample entries from a statistics array, yielding only uncorrelated
    entries as determined from a provided statistical inefficiency and equilibration time.
    """

    @protocol_input(str)
    def input_statistics_path(self):
        """The file path to the statistics to subsample."""
        pass

    @protocol_output(str)
    def output_statistics_path(self):
        """The file path to the subsampled statistics."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._input_statistics_path = None
        self._output_statistics_path = None

    def execute(self, directory, available_resources):

        logging.info('Subsampling statistics: {}'.format(self.id))

        if self._input_statistics_path is None:

            return PropertyEstimatorException(directory=directory,
                                              message='The ExtractUncorrelatedStatisticsData protocol '
                                                       'requires a previously calculated statisitics file')

        statistics_array = StatisticsArray.from_pandas_csv(self._input_statistics_path)

        uncorrelated_indices = timeseries.get_uncorrelated_indices(len(statistics_array) - self._equilibration_index,
                                                                   self._statistical_inefficiency)

        uncorrelated_indices = [index + self._equilibration_index for index in uncorrelated_indices]
        uncorrelated_statistics = StatisticsArray.from_existing(statistics_array, uncorrelated_indices)

        self._output_statistics_path = path.join(directory, 'uncorrelated_statistics.csv')
        uncorrelated_statistics.to_pandas_csv(self._output_statistics_path)

        logging.info('Statistics subsampled: {}'.format(self.id))

        self._number_of_uncorrelated_samples = len(uncorrelated_statistics)

        return self._get_output_dictionary()
