"""
Units tests for propertyestimator.utils.statistics
"""

import numpy as np
from pymbar import timeseries as pymbar_timeseries

from propertyestimator.utils import timeseries


def test_statistical_inefficiency():
    """Test the statistical inefficiency calculation utility."""

    data_size = 200000

    random_array = np.random.rand(data_size)
    numpy_vector_array = []

    for i in range(data_size):
        numpy_vector_array.append([random_array[i]])

    a = np.array(numpy_vector_array)

    statistical_inefficiency = timeseries.calculate_statistical_inefficiency(a, minimum_samples=3)
    pymbar_statistical_inefficiency = pymbar_timeseries.statisticalInefficiency(a, mintime=3)

    print('utils: {}, pymbar: {}', statistical_inefficiency, pymbar_statistical_inefficiency)

    assert abs(statistical_inefficiency - pymbar_statistical_inefficiency) < 0.00001
