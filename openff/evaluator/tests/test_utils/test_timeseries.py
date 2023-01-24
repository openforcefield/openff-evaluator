"""
Units tests for openff.evaluator.utils.statistics
"""

import numpy as np

from openff.evaluator.utils.pymbar import detect_equilibration
from openff.evaluator.utils.timeseries import (
    analyze_time_series,
    get_uncorrelated_indices,
)


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


def test_get_uncorrelated_indices():

    uncorrelated_indices = get_uncorrelated_indices(4, 2.0)
    assert uncorrelated_indices == [0, 2]
