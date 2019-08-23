"""
Units tests for propertyestimator.utils.statistics
"""
import os

import numpy as np
import pytest

from propertyestimator import unit
from propertyestimator.utils import get_data_filename
from propertyestimator.utils.statistics import StatisticsArray, bootstrap, ObservableType


def test_statistics_object():

    statistics_object = StatisticsArray.from_openmm_csv(get_data_filename('test/statistics/stats_openmm.csv'),
                                                        1*unit.atmosphere)

    statistics_object.to_pandas_csv('stats_pandas.csv')

    statistics_object = StatisticsArray.from_pandas_csv('stats_pandas.csv')
    assert statistics_object is not None

    subsampled_array = StatisticsArray.from_existing(statistics_object, [1, 2, 3])
    assert subsampled_array is not None and len(subsampled_array) == 3

    if os.path.isfile('stats_pandas.csv'):
        os.unlink('stats_pandas.csv')

    reduced_potential = np.array([0.1] * (len(statistics_object) - 1))

    with pytest.raises(ValueError):
        statistics_object[ObservableType.ReducedPotential] = reduced_potential

    reduced_potential = np.array([0.1] * len(statistics_object))

    with pytest.raises(ValueError):
        statistics_object[ObservableType.ReducedPotential] = reduced_potential

    statistics_object[ObservableType.ReducedPotential] = reduced_potential * unit.dimensionless

    assert ObservableType.ReducedPotential in statistics_object


def test_bootstrap():

    simple_data = np.array([1.0, 1.0, 1.0, 1.0])

    def bootstrap_function(values):
        return values.mean()

    value, uncertainty = bootstrap(bootstrap_function, 5, 1.0, values=simple_data)
    assert np.isclose(value, simple_data.mean()) and uncertainty == 0.0

    vector_data = np.array([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]])
    value, uncertainty = bootstrap(bootstrap_function, 5, 1.0, values=vector_data)
    assert np.isclose(value, vector_data.mean()) and uncertainty == 0.0

    simple_sub_data = np.array([1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0])
    value, uncertainty = bootstrap(bootstrap_function, 5, 1.0, np.array([2, 3, 4]), values=simple_sub_data)
    assert np.isclose(value, simple_sub_data.mean())

    vector_sub_data = np.array([
        [1.0, 2.0], [1.0, 2.0],
        [3.0, 4.0], [3.0, 4.0], [3.0, 4.0],
        [5.0, 6.0], [5.0, 6.0], [5.0, 6.0], [5.0, 6.0]
    ])
    value, uncertainty = bootstrap(bootstrap_function, 5, 1.0, np.array([2, 3, 4]), values=vector_sub_data)
    assert np.isclose(value, vector_sub_data.mean())
