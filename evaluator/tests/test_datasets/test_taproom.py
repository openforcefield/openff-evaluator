import pytest

from evaluator.datasets.taproom import TaproomDataSet

pytest.importorskip("taproom")


def test_initialization():
    """Test the initialization of a TaproomDataSet."""

    data_set = TaproomDataSet()

    assert data_set is not None
    assert len(data_set.properties) > 0


def test_filtering():
    """Test filtering the data set"""

    data_set = TaproomDataSet()

    data_set.filter_by_host_identifiers("bcd", "acd")
    data_set.filter_by_guest_identifiers("cbu")

    assert data_set is not None
    assert len(data_set.properties) == 2
