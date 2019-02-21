"""
Units tests for propertyestimator.utils.statistics
"""

from simtk import unit

from propertyestimator.utils import get_data_filename
from propertyestimator.utils.statistics import StatisticsArray


def test_statistics_object():

    statistics_object = StatisticsArray.from_openmm_csv(get_data_filename('properties/stats_openmm.csv'), 1*unit.atmosphere)
    statistics_object.save_as_pandas_csv('stats_pandas.csv')

    statistics_object = StatisticsArray.from_pandas_csv('stats_pandas.csv')
    subsampled_array = StatisticsArray.from_statistics_array(statistics_object, [1, 2, 3])

    assert statistics_object is not None
    assert subsampled_array is not None and len(subsampled_array) == 3
