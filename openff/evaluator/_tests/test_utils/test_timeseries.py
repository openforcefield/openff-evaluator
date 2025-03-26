"""
Units tests for openff.evaluator.utils.statistics
"""

import numpy as np

from openff.evaluator.utils.pymbar import detect_equilibration
from openff.evaluator.utils.timeseries import (
    analyze_time_series,
    get_uncorrelated_indices,
)

import pytest


def test_analyze_time_series_std():
    """Test the ``analyze_time_series`` utility with flat data."""

    statistics = analyze_time_series(np.ones(10))

    assert statistics.n_total_points == 10
    assert statistics.n_uncorrelated_points == 1
    assert np.isclose(statistics.statistical_inefficiency, 10.0)
    assert statistics.equilibration_index == 0


def test_analyze_time_series():
    """Compare the output of the ``analyze_time_series`` utility with ``pymbar``."""

    np.random.seed(4)
    random_array = np.random.rand(10)

    statistics = analyze_time_series(random_array, minimum_samples=3)
    expected_index, expected_value, _ = detect_equilibration(random_array, fast=False)

    assert expected_index == statistics.equilibration_index
    assert np.isclose(statistics.statistical_inefficiency, expected_value)
    assert statistics.n_total_points == 10
    assert 0 < statistics.n_uncorrelated_points <= 10
    assert 0 <= statistics.equilibration_index < 10


@pytest.mark.parametrize("n_frames", [0, 1, 2, 3, 4, 5, 6, 7, 8])
def test_analyze_time_series_discard_initial(n_frames):
    """Compare the output of the ``analyze_time_series`` utility with ``pymbar``."""

    sample_array = np.array([
        0.21773057, 0.49832794, 0.13265156, 0.30525485, 0.73907393,
        0.90003993, 0.0676494, 0.36543323, 0.56557613, 0.30557112
    ])

    statistics = analyze_time_series(
        sample_array,
        minimum_samples=3, 
        discard_initial_frames=n_frames
    )
    expected_index, expected_value, _ = detect_equilibration(sample_array, fast=False)

    assert expected_index == 0
    assert statistics.equilibration_index == n_frames
    assert np.isclose(statistics.statistical_inefficiency, expected_value)
    assert statistics.n_total_points == 10
    assert 0 < statistics.n_uncorrelated_points <= 10
    assert 0 <= statistics.equilibration_index < 10


def test_get_uncorrelated_indices():
    uncorrelated_indices = get_uncorrelated_indices(4, 2.0)
    assert uncorrelated_indices == [0, 2]


@pytest.mark.parametrize("n_frames, expected_indices", [
    (0, [0, 2, 4, 6, 8]),
    (1, [1, 3, 5, 7, 9]),
    (2, [2, 4, 6, 8]),
    (3, [3, 5, 7, 9]),
    (4, [4, 6, 8]),
    (5, [5, 7, 9]),
    (6, [6, 8]),
    (7, [7, 9]),
    (8, [8]),
])
def test_get_uncorrelated_indices_discard(n_frames, expected_indices):
    uncorrelated_indices = get_uncorrelated_indices(10, 1.2, discard_initial_frames=n_frames)
    assert uncorrelated_indices == expected_indices
