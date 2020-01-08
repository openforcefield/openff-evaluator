"""
Units tests for propertyestimator.utils.exceptions
"""
from propertyestimator.utils import exceptions


def test_estimator_exceptions():
    """Test estimator, json based exceptions."""

    estimator_exception = exceptions.EvaluatorException(message="dummy_message")

    exception_state = estimator_exception.__getstate__()

    assert len(exception_state) == 1
    assert "message" in exception_state

    recreated_exception = exceptions.EvaluatorException()
    recreated_exception.__setstate__(exception_state)

    assert estimator_exception.message == recreated_exception.message
