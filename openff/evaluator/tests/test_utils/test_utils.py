"""
Units tests for openff.evaluator.utils.exceptions
"""
import os
import tempfile

from openff.evaluator.utils.utils import get_nested_attribute, is_file_and_not_empty


class DummyNestedClass:
    def __init__(self):
        self.object_a = None
        self.object_b = None


class DummyDescriptor1:
    def __init__(self, class_attribute):
        self.attribute = class_attribute.__name__

    def __get__(self, instance, owner=None):
        pass

    def __set__(self, instance, value):
        pass


class DummyDescriptor2:
    def __init__(self, class_attribute):
        self.attribute = class_attribute.__name__

    def __get__(self, instance, owner=None):
        pass

    def __set__(self, instance, value):
        pass


def test_get_nested_attribute():

    dummy_object = DummyNestedClass()
    dummy_object.object_a = "a"

    dummy_nested_object_a = DummyNestedClass()
    dummy_nested_object_a.object_a = 1
    dummy_nested_object_a.object_b = [0]

    dummy_nested_list_object_0 = DummyNestedClass()
    dummy_nested_list_object_0.object_a = "a"
    dummy_nested_list_object_0.object_b = "b"

    dummy_nested_object_b = DummyNestedClass()
    dummy_nested_object_b.object_a = 2
    dummy_nested_object_b.object_b = [dummy_nested_list_object_0]

    dummy_object.object_b = {"a": dummy_nested_object_a, "b": dummy_nested_object_b}

    assert get_nested_attribute(dummy_object, "object_a") == "a"

    assert get_nested_attribute(dummy_object, "object_b[a].object_a") == 1
    assert get_nested_attribute(dummy_object, "object_b[a].object_b[0]") == 0

    assert get_nested_attribute(dummy_object, "object_b[b].object_a") == 2
    assert get_nested_attribute(dummy_object, "object_b[b].object_b[0].object_a") == "a"
    assert get_nested_attribute(dummy_object, "object_b[b].object_b[0].object_b") == "b"


def test_is_file_and_not_empty(tmpdir):
    path = "testfile"
    with tempfile.TemporaryDirectory() as directory:
        with open(path, "w") as f:
            pass

    assert os.path.isfile(path)
    assert os.path.getsize(path) == 0
    assert not is_file_and_not_empty(path)

    path = "testfile2"
    with tempfile.TemporaryDirectory() as directory:
        with open(path, "w") as f:
            f.write("ubiquitous mendacious polyglottal")

    assert os.path.isfile(path)
    assert os.path.getsize(path) != 0
    assert is_file_and_not_empty(path)
