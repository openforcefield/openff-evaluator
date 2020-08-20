import tempfile

import pytest

from openff.evaluator import unit
from openff.evaluator.backends import ComputeResources
from openff.evaluator.protocols.analysis import (
    AverageFreeEnergies,
    ExtractAverageStatistic,
    ExtractUncorrelatedStatisticsData,
    ExtractUncorrelatedTrajectoryData,
)
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils import get_data_filename
from openff.evaluator.utils.statistics import ObservableType, StatisticsArray


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


def test_average_free_energies_protocol():
    """Tests adding together two free energies."""

    compute_resources = ComputeResources(number_of_threads=1)

    delta_g_one = (-10.0 * unit.kilocalorie / unit.mole).plus_minus(
        1.0 * unit.kilocalorie / unit.mole
    )
    delta_g_two = (-20.0 * unit.kilocalorie / unit.mole).plus_minus(
        2.0 * unit.kilocalorie / unit.mole
    )

    thermodynamic_state = ThermodynamicState(298 * unit.kelvin, 1 * unit.atmosphere)

    sum_protocol = AverageFreeEnergies("average_free_energies")

    sum_protocol.values = [delta_g_one, delta_g_two]
    sum_protocol.thermodynamic_state = thermodynamic_state

    sum_protocol.execute("", compute_resources)

    result_value = sum_protocol.result.value.to(unit.kilocalorie / unit.mole)
    result_uncertainty = sum_protocol.result.error.to(unit.kilocalorie / unit.mole)

    assert isinstance(sum_protocol.result, unit.Measurement)
    assert result_value.magnitude == pytest.approx(-20.0, abs=0.2)
    assert result_uncertainty.magnitude == pytest.approx(2.0, abs=0.2)
