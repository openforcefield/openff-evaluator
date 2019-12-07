"""
Units tests for propertyestimator.workflow.decorators
"""
import json

from propertyestimator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder
from propertyestimator.workflow.decorators import UNDEFINED, UndefinedAttribute


def test_undefined_singleton():
    """A test of the UNDEFINED singleton pattern"""

    value_a = UndefinedAttribute()
    value_b = UndefinedAttribute()

    assert value_a == value_b


def test_undefined_serialization():
    """A test of serializing the UNDEFINED placeholder"""

    value_a = UNDEFINED
    value_a_json = json.dumps(value_a, cls=TypedJSONEncoder)
    value_a_recreated = json.loads(value_a_json, cls=TypedJSONDecoder)

    assert value_a == value_a_recreated
