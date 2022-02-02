"""
Units tests for openff.evaluator.utils.serialization
"""
import json
from datetime import datetime
from enum import Enum, IntEnum

import numpy as np
import pytest
from openff.units import unit

from openff.evaluator.client import EvaluatorClient
from openff.evaluator.utils.serialization import (
    TypedBaseModel,
    TypedJSONDecoder,
    TypedJSONEncoder,
    _type_string_to_object,
    _type_to_type_string,
    deserialize_quantity,
    serialize_quantity,
)


class Foo:
    def __init__(self):

        self.field1 = "field1"
        self.field2 = 2

    def __getstate__(self):

        return {"field1": self.field1, "field2": self.field2}

    def __setstate__(self, state):

        self.field1 = state["field1"]
        self.field2 = state["field2"]


class FooInherited(Foo):
    def __init__(self):

        super().__init__()
        self.field3 = 100

    def __getstate__(self):

        self_state = {"field3": self.field3}
        parent_state = super(FooInherited, self).__getstate__()

        self_state.update(parent_state)

        return self_state

    def __setstate__(self, state):

        self.field3 = state["field3"]
        super(FooInherited, self).__setstate__(state)


class Bar(TypedBaseModel):
    def __init__(self):

        self.field1 = "field1"
        self.field2 = 2

    def __getstate__(self):

        return {
            "field1": self.field1,
            "field2": self.field2,
        }

    def __setstate__(self, state):

        self.field1 = state["field1"]
        self.field2 = state["field2"]


class BarInherited(Bar):

    field3: str = 1000


class Baz(Enum):

    Option1 = "Option1"
    Option2 = "Option2"


class Qux(IntEnum):

    Option1 = 1
    Option2 = 2


class NestedParent:
    class NestedChild(Enum):

        Option1 = "Option1"
        Option2 = "Option2"


class ComplexObject:
    class NestedClass1:
        def __init__(self):

            self.field1 = 5 * unit.kelvin

        def __getstate__(self):
            return {
                "field1": self.field1,
            }

        def __setstate__(self, state):
            self.field1 = state["field1"]

    class NestedClass2:
        def __init__(self):
            self.field1 = Qux.Option1

        def __getstate__(self):
            return {
                "field1": self.field1,
            }

        def __setstate__(self, state):
            self.field1 = state["field1"]

    def __init__(self):

        self.field1 = ComplexObject.NestedClass1()
        self.field2 = ComplexObject.NestedClass2()

    def __getstate__(self):

        return {"field1": self.field1, "field2": self.field2}

    def __setstate__(self, state):

        self.field1 = state["field1"]
        self.field2 = state["field2"]


class TestClass(TypedBaseModel):
    def __init__(self, inputs=None):
        self.inputs = inputs

        self.foo = Foo()
        self.bar = Bar()

        self.foo_inherited = FooInherited()
        self.bar_inherited = BarInherited()

        self.complex = ComplexObject()

    def __getstate__(self):

        return {
            "inputs": self.inputs,
            "foo": self.foo,
            "bar": self.bar,
            "foo_inherited": self.foo_inherited,
            "bar_inherited": self.bar_inherited,
            "complex": self.complex,
        }

    def __setstate__(self, state):

        self.inputs = state["inputs"]

        self.foo = state["foo"]
        self.bar = state["bar"]

        self.foo_inherited = state["foo_inherited"]
        self.bar_inherited = state["bar_inherited"]

        self.complex = state["complex"]


def test_polymorphic_dictionary():
    """Test the polymorphic dictionary helper class."""

    test_dictionary = {
        "test_str": "test1",
        "test_int": 1,
        "test_bool": True,
        "test_None": None,
        "test_Foo": Foo(),
        "test_FooInherited": FooInherited(),
        "test_Bar": Bar(),
        "test_BarInherited": BarInherited(),
        "test_Baz": Baz.Option1,
        "test_Qux": Qux.Option1,
        "test_Nested": NestedParent.NestedChild.Option1,
        "test_List": [Foo(), Bar(), 1, "Hello World"],
        "test_Complex": ComplexObject(),
    }

    test_object = TestClass(inputs=test_dictionary)
    test_json = test_object.json()

    test_recreated = TestClass.parse_json(test_json)
    test_recreated_json = test_recreated.json()

    assert test_json == test_recreated_json


def test_dimensionless_quantity_serialization():

    test_value = 1.0 * unit.dimensionless

    serialized_value = serialize_quantity(test_value)
    deserialized_value = deserialize_quantity(serialized_value)

    assert test_value == deserialized_value

    test_value = 1.0 * unit.dimensionless

    serialized_value = serialize_quantity(test_value)
    deserialized_value = deserialize_quantity(serialized_value)

    assert test_value == deserialized_value


@pytest.mark.parametrize("float_type", [np.float16, np.float32, np.float64])
def test_numpy_float_serialization(float_type):

    original_value = float_type(0.987654321)

    serialized_value = json.dumps(original_value, cls=TypedJSONEncoder)
    deserialized_value = json.loads(serialized_value, cls=TypedJSONDecoder)

    assert original_value == deserialized_value


@pytest.mark.parametrize("int_type", [np.int32, np.int64])
def test_numpy_int_serialization(int_type):

    original_value = int_type(987654321)

    serialized_value = json.dumps(original_value, cls=TypedJSONEncoder)
    deserialized_value = json.loads(serialized_value, cls=TypedJSONDecoder)

    assert original_value == deserialized_value


def test_numpy_array_serialization():

    one_dimensional_array = np.array([1, 2, 3, 4, 5])

    serialized_value = json.dumps(one_dimensional_array, cls=TypedJSONEncoder)
    deserialized_value = json.loads(serialized_value, cls=TypedJSONDecoder)

    assert np.allclose(one_dimensional_array, deserialized_value)

    two_dimensional_array = np.array([[1, 9], [2, 8], [3, 7], [4, 6], [5, 5]])

    serialized_value = json.dumps(two_dimensional_array, cls=TypedJSONEncoder)
    deserialized_value = json.loads(serialized_value, cls=TypedJSONDecoder)

    assert np.allclose(two_dimensional_array, deserialized_value)

    one_dimensional_quantity_array = one_dimensional_array * unit.kelvin

    serialized_value = json.dumps(one_dimensional_quantity_array, cls=TypedJSONEncoder)
    deserialized_value = json.loads(serialized_value, cls=TypedJSONDecoder)

    assert np.allclose(
        one_dimensional_quantity_array.magnitude, deserialized_value.magnitude
    )

    two_dimensional_quantity_array = two_dimensional_array * unit.kelvin

    serialized_value = json.dumps(two_dimensional_quantity_array, cls=TypedJSONEncoder)
    deserialized_value = json.loads(serialized_value, cls=TypedJSONDecoder)

    assert np.allclose(
        two_dimensional_quantity_array.magnitude, deserialized_value.magnitude
    )


def test_pint_serialization():

    test_value = 1.0 * unit.kelvin

    serialized_value = json.dumps(test_value, cls=TypedJSONEncoder)
    deserialized_value = json.loads(serialized_value, cls=TypedJSONDecoder)

    assert test_value == deserialized_value

    test_value = test_value.plus_minus(1.0 * unit.kelvin)

    serialized_value = json.dumps(test_value, cls=TypedJSONEncoder)
    deserialized_value = json.loads(serialized_value, cls=TypedJSONDecoder)

    assert test_value.value == deserialized_value.value
    assert test_value.error == deserialized_value.error


def test_datetime_serialization():

    test_value = datetime.now()

    serialized_value = json.dumps(test_value, cls=TypedJSONEncoder)
    deserialized_value = json.loads(serialized_value, cls=TypedJSONDecoder)

    assert test_value == deserialized_value


def test_type_string_to_object():

    client_class = _type_string_to_object("openff.evaluator.client.EvaluatorClient")
    assert client_class == EvaluatorClient

    nested_class = _type_string_to_object(
        "openff.evaluator.client.EvaluatorClient._Submission"
    )
    assert nested_class == EvaluatorClient._Submission


def test_type_to_type_string():

    client_string = _type_to_type_string(EvaluatorClient)
    assert client_string == "openff.evaluator.client.client.EvaluatorClient"

    nested_string = _type_to_type_string(EvaluatorClient._Submission)
    assert nested_string == "openff.evaluator.client.client.EvaluatorClient._Submission"


def test_backwards_compatibility():

    old_json = '{"value": 298.15, "unit": "K", "@type": "evaluator.unit.Quantity"}'

    value = json.loads(old_json, cls=TypedJSONDecoder)
    assert isinstance(value, unit.Quantity)
