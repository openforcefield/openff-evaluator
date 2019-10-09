"""
Units tests for propertyestimator.datasets
"""

from propertyestimator.datasets.taproom import TaproomDataSet


def test_initialization():
    """Test the initialization of a TaproomDataSet."""

    # Mole fraction
    data_set = TaproomDataSet()

    assert data_set is not None
    assert len(data_set.properties) > 0
