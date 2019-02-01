"""
Units tests for propertyestimator.utils.statistics
"""

from simtk import unit

from propertyestimator.utils import get_data_filename
from propertyestimator.utils.statistics import Statistics


def test_statistics_object():

    statistics_object = Statistics.from_openmm_csv(get_data_filename('properties/stats_openmm.csv'), 1*unit.atmosphere)
    print(statistics_object)

    statistics_object.save_as_pandas_csv('stats_pandas.csv')

    statistics_object = Statistics.from_pandas_csv('stats_pandas.csv')
    assert statistics_object is not None
