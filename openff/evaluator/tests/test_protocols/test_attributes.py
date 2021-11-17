import pytest


def test_deprecation_warning():
    msg = ("InequalityMergeBehaviour is a DEPRECATED spelling and will "
           "be removed in a future release. Please use InequalityMergeBehavior instead")
    with pytest.warns(DeprecationWarning, match=msg):
        from openff.evaluator.workflow.attributes import InequalityMergeBehaviour
