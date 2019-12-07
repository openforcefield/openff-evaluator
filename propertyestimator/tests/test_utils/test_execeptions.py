"""
Units tests for propertyestimator.utils.exceptions
"""
import pytest

from propertyestimator.utils import exceptions


def test_xml_exceptions():
    """Test xml raised exceptions."""

    with pytest.raises(exceptions.XmlNodeMissingException):
        raise exceptions.XmlNodeMissingException("dummy_node")


def test_estimator_exceptions():
    """Test estimator, json based exceptions."""

    estimator_exception = exceptions.PropertyEstimatorException(
        directory="dummy_dir", message="dummy_message"
    )

    exception_state = estimator_exception.__getstate__()

    assert len(exception_state) == 2
    assert "directory" in exception_state and "message" in exception_state

    recreated_exception = exceptions.PropertyEstimatorException()
    recreated_exception.__setstate__(exception_state)

    assert estimator_exception.directory == recreated_exception.directory
    assert estimator_exception.message == recreated_exception.message
