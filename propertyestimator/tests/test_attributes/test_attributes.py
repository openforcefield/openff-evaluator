"""
Units tests for propertyestimator.attributes
"""
import json

import pytest

from propertyestimator.attributes import UNDEFINED, Attribute, AttributeClass
from propertyestimator.utils.serialization import TypedJSONDecoder, TypedJSONEncoder
from propertyestimator.workflow.attributes import InputAttribute, OutputAttribute


class AttributeObject(AttributeClass):

    required_input = InputAttribute("", str, UNDEFINED, optional=False)
    optional_input = InputAttribute("", int, UNDEFINED, optional=True)

    some_output = OutputAttribute("", int)

    def __init__(self):
        self.some_output = 5


class NestedAttributeObject(AttributeClass):

    some_value = Attribute("", AttributeObject)

    some_list = Attribute("", list, UNDEFINED, optional=True)
    some_dict = Attribute("", dict, UNDEFINED, optional=True)


def test_undefined_singleton():
    """A test of the UNDEFINED singleton pattern"""

    from propertyestimator.attributes.attributes import UndefinedAttribute

    value_a = UndefinedAttribute()
    value_b = UndefinedAttribute()

    assert value_a == value_b


def test_undefined_serialization():
    """A test of serializing the UNDEFINED placeholder"""

    value_a = UNDEFINED
    value_a_json = json.dumps(value_a, cls=TypedJSONEncoder)
    value_a_recreated = json.loads(value_a_json, cls=TypedJSONDecoder)

    assert value_a == value_a_recreated


def test_get_attributes():

    input_attributes = AttributeObject.get_attributes(InputAttribute)
    assert input_attributes == ["required_input", "optional_input"]

    output_attributes = AttributeObject.get_attributes(OutputAttribute)
    assert output_attributes == ["some_output"]

    all_attributes = AttributeObject.get_attributes()
    assert all_attributes == ["required_input", "optional_input", "some_output"]


def test_type_check():

    some_object = AttributeObject()

    with pytest.raises(ValueError):
        some_object.required_input = 5


def test_state_methods():

    some_object = AttributeObject()
    some_object.required_input = "Set"

    state = some_object.__getstate__()

    assert len(state) == 2

    new_object = AttributeObject()
    new_object.required_input = ""
    new_object.optional_input = 10

    new_object.__setstate__(state)

    assert new_object.required_input == some_object.required_input
    assert new_object.optional_input == some_object.optional_input
    assert new_object.some_output == some_object.some_output


def test_nested_validation():

    nested_object = NestedAttributeObject()
    nested_object.some_value = AttributeObject()

    # Should fail
    with pytest.raises(ValueError):
        nested_object.validate()

    nested_object.some_value.required_input = ""
    nested_object.validate()

    nested_object.some_list = [AttributeObject()]

    # Should fail
    with pytest.raises(ValueError):
        nested_object.validate()

    nested_object.some_list[0].required_input = ""
    nested_object.validate()

    nested_object.some_dict = {"x": AttributeObject()}

    # Should fail
    with pytest.raises(ValueError):
        nested_object.validate()

    nested_object.some_dict["x"].required_input = ""
    nested_object.validate()
