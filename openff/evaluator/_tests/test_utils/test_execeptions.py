"""
Units tests for openff.evaluator.utils.exceptions
"""

from openff.evaluator.utils import exceptions


def test_evaluator_exceptions():
    """Test evaluator, json based exceptions."""

    evaluator_exception = exceptions.EvaluatorException(message="dummy_message")

    exception_state = evaluator_exception.__getstate__()

    assert len(exception_state) == 1
    assert "message" in exception_state

    recreated_exception = exceptions.EvaluatorException()
    recreated_exception.__setstate__(exception_state)

    assert evaluator_exception.message == recreated_exception.message
