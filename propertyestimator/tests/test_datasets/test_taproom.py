"""
Units tests for propertyestimator.datasets
"""
import pytest
from propertyestimator.datasets.taproom import TaproomDataSet

has_taproom = True

try:
    import taproom
except ImportError:
    has_taproom = False


@pytest.mark.skipif(not has_taproom, reason='The optional `taproom` package is not installed.')
def test_initialization():
    """Test the initialization of a TaproomDataSet."""

    data_set = TaproomDataSet()

    assert data_set is not None
    assert len(data_set.properties) > 0


@pytest.mark.skipif(not has_taproom, reason='The optional `taproom` package is not installed.')
def test_filtering():
    """Test filtering the data set"""

    data_set = TaproomDataSet()

    data_set.filter_by_host_identifiers('bcd', 'acd')
    data_set.filter_by_guest_identifiers('cbu')

    assert data_set is not None
    assert len(data_set.properties) == 2
