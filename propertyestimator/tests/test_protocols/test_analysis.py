import tempfile

from propertyestimator.backends import ComputeResources
from propertyestimator.protocols.analysis import (
    ExtractAverageStatistic,
    ExtractUncorrelatedStatisticsData,
    ExtractUncorrelatedTrajectoryData,
)
from propertyestimator.utils import get_data_filename
from propertyestimator.utils.statistics import ObservableType, StatisticsArray


def test_extract_average_statistic():

    statistics_path = get_data_filename("test/statistics/stats_pandas.csv")

    with tempfile.TemporaryDirectory() as temporary_directory:

        extract_protocol = ExtractAverageStatistic("extract_protocol")
        extract_protocol.statistics_path = statistics_path
        extract_protocol.statistics_type = ObservableType.PotentialEnergy
        extract_protocol.execute(temporary_directory, ComputeResources())


def test_extract_uncorrelated_trajectory_data():

    import mdtraj

    coordinate_path = get_data_filename("test/trajectories/water.pdb")
    trajectory_path = get_data_filename("test/trajectories/water.dcd")

    original_trajectory = mdtraj.load(trajectory_path, top=coordinate_path)

    with tempfile.TemporaryDirectory() as temporary_directory:

        extract_protocol = ExtractUncorrelatedTrajectoryData("extract_protocol")
        extract_protocol.input_coordinate_file = coordinate_path
        extract_protocol.input_trajectory_path = trajectory_path
        extract_protocol.equilibration_index = 2
        extract_protocol.statistical_inefficiency = 2.0
        extract_protocol.execute(temporary_directory, ComputeResources())

        final_trajectory = mdtraj.load(
            extract_protocol.output_trajectory_path, top=coordinate_path
        )
        assert len(final_trajectory) == (len(original_trajectory) - 2) / 2
        assert (
            extract_protocol.number_of_uncorrelated_samples
            == (len(original_trajectory) - 2) / 2
        )


def test_extract_uncorrelated_statistics_data():

    statistics_path = get_data_filename("test/statistics/stats_pandas.csv")
    original_array = StatisticsArray.from_pandas_csv(statistics_path)

    with tempfile.TemporaryDirectory() as temporary_directory:

        extract_protocol = ExtractUncorrelatedStatisticsData("extract_protocol")
        extract_protocol.input_statistics_path = statistics_path
        extract_protocol.equilibration_index = 2
        extract_protocol.statistical_inefficiency = 2.0
        extract_protocol.execute(temporary_directory, ComputeResources())

        final_array = StatisticsArray.from_pandas_csv(
            extract_protocol.output_statistics_path
        )
        assert len(final_array) == (len(original_array) - 2) / 2
        assert (
            extract_protocol.number_of_uncorrelated_samples
            == (len(original_array) - 2) / 2
        )
