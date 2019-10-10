"""
A collection of protocols for running analysing the results of molecular simulations.
"""

import logging
from os import path

import numpy as np
import typing

from propertyestimator import unit
from propertyestimator.utils import statistics, timeseries
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import StatisticsArray, bootstrap
from propertyestimator.workflow.decorators import protocol_input, protocol_output, InequalityMergeBehaviour
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class AveragePropertyProtocol(BaseProtocol):
    """An abstract base class for protocols which will calculate the
    average of a property and its uncertainty via bootstrapping.
    """

    bootstrap_iterations = protocol_input(docstring='The number of bootstrap iterations to perform.',
                                          type_hint=int,
                                          default_value=250,
                                          merge_behavior=InequalityMergeBehaviour.LargestValue)

    bootstrap_sample_size = protocol_input(docstring='The relative sample size to use for bootstrapping.',
                                           type_hint=float,
                                           default_value=1.0,
                                           merge_behavior=InequalityMergeBehaviour.LargestValue)

    value = protocol_output(docstring='The average value and its uncertainty.',
                            type_hint=EstimatedQuantity)

    equilibration_index = protocol_output(docstring='The index in the data set after which the data is stationary.',
                                          type_hint=int)

    statistical_inefficiency = protocol_output(docstring='The statistical inefficiency in the data set.',
                                               type_hint=float)

    uncorrelated_values = protocol_output(docstring='The uncorrelated values which the average was calculated from.',
                                          type_hint=unit.Quantity)

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

    input_coordinate_file = protocol_input(docstring='The file path to the starting coordinates of a trajectory.',
                                           type_hint=str,
                                           default_value=protocol_input.UNDEFINED)

    trajectory_path = protocol_input(docstring='The file path to the trajectory to average over.',
                                     type_hint=str,
                                     default_value=protocol_input.UNDEFINED)

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

    statistics_path = protocol_input(docstring='The file path to the statistics to average over.',
                                     type_hint=str,
                                     default_value=protocol_input.UNDEFINED)

    statistics_type = protocol_input(docstring='The type of statistic to average over.',
                                     type_hint=statistics.ObservableType,
                                     default_value=protocol_input.UNDEFINED)

    divisor = protocol_input(docstring='A value to divide the statistic by. This is useful if a statistic (such '
                                       'as enthalpy) needs to be normalised by the number of molecules.',
                             type_hint=typing.Union[int, float, unit.Quantity],
                             default_value=1.0)

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

    equilibration_index = protocol_input(docstring='The index in the data set after which the data is stationary.',
                                         type_hint=int,
                                         default_value=protocol_input.UNDEFINED,
                                         merge_behavior=InequalityMergeBehaviour.LargestValue)

    statistical_inefficiency = protocol_input(docstring='The statistical inefficiency in the data set.',
                                              type_hint=float,
                                              default_value=protocol_input.UNDEFINED,
                                              merge_behavior=InequalityMergeBehaviour.LargestValue)

    number_of_uncorrelated_samples = protocol_output(docstring='The number of uncorrelated samples.',
                                                     type_hint=int)

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

    input_coordinate_file = protocol_input(docstring='The file path to the starting coordinates of a trajectory.',
                                           type_hint=str,
                                           default_value=protocol_input.UNDEFINED)

    input_trajectory_path = protocol_input(docstring='The file path to the trajectory to subsample.',
                                           type_hint=str,
                                           default_value=protocol_input.UNDEFINED)

    output_trajectory_path = protocol_output(docstring='The file path to the subsampled trajectory.',
                                             type_hint=str)

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

        with DCDTrajectoryFile(self._input_trajectory_path, 'r') as input_file:

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

    input_statistics_path = protocol_input(docstring='The file path to the statistics to subsample.',
                                           type_hint=str,
                                           default_value=protocol_input.UNDEFINED)

    output_statistics_path = protocol_output(docstring='The file path to the subsampled statistics.',
                                             type_hint=str)

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
