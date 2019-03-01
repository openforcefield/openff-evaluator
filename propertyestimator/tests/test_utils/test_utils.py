"""
Units tests for propertyestimator.utils.exceptions
"""
import abc

from propertyestimator.utils import utils
from propertyestimator.utils.utils import SubhookedABCMeta


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


class DummyInterface(SubhookedABCMeta):

    @abc.abstractmethod
    def dummy_function_1(self):
        pass


class DummyDecoratedClass:

    @DummyDescriptor1
    def dummy_function_1(self):
        pass

    @DummyDescriptor1
    def dummy_function_2(self):
        pass

    @DummyDescriptor2
    def dummy_function_3(self):
        pass


def test_find_decorator():
    """Test that decorated functions can be identified."""

    types = utils.find_types_with_decorator(DummyDecoratedClass, 'DummyDescriptor1')
    assert len(types) == 2 and set(types) == {'dummy_function_1', 'dummy_function_2'}

    types = utils.find_types_with_decorator(DummyDecoratedClass, 'DummyDescriptor2')
    assert len(types) == 1 and types[0] == 'dummy_function_3'


def test_interfaces():
    """Test that interface checking is working."""
    dummy_class = DummyDecoratedClass()
    assert isinstance(dummy_class, DummyInterface)
