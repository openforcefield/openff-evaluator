"""
A collection of descriptors used to mark-up elements in a workflow, such
as the inputs or outputs of workflow protocols.
"""
import abc
import copy
import inspect
from enum import Enum

from propertyestimator import unit
from propertyestimator.attributes.typing import is_instance_of_type, is_supported_type
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.serialization import TypedBaseModel


class UndefinedAttribute:
    """A custom type used to differentiate between ``None`` values,
    and an undeclared optional value."""

    def __eq__(self, other):
        return type(other) == UndefinedAttribute

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        return


class PlaceholderInput:
    """A class to act as a place holder for an input value
    which is not known a priori. This may include a value
    which will be set by a workflow as the output of an
    executed protocol.
    """

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


UNDEFINED = UndefinedAttribute()


class BaseMergeBehaviour(Enum):
    """A base class for enums which will describes how attributes should
    be handled when attempting to merge similar protocols.
    """

    pass


class MergeBehaviour(BaseMergeBehaviour):
    """A enum which describes how attributes should be handled when
    attempting to merge similar protocols.

    This enum may take values of

    * ExactlyEqual: This attribute must be exactly equal between two protocols for
      them to be able to merge.
    """

    ExactlyEqual = "ExactlyEqual"


class InequalityMergeBehaviour(BaseMergeBehaviour):
    """A enum which describes how attributes which can be compared
    with inequalities should be merged.

    This enum may take values of

    * SmallestValue: When two protocols are merged, the smallest value of this
      attribute from either protocol is retained.
    * LargestValue: When two protocols are merged, the largest value of this
      attribute from either protocol is retained.
    """

    SmallestValue = "SmallestValue"
    LargestValue = "LargestValue"


class BaseAttributeClass(TypedBaseModel):
    """A base class for objects which require will defined
    attributes which, dependant on the object, may be flagged
    as either being a required input to the object or a
    provided output.
    """

    @classmethod
    def _get_attributes(cls, attribute_type):
        """Returns all attributes of a specific `attribute_type`.

        Parameters
        ----------
        attribute_type: type of BaseAttribute
            The type of attribute to search for.

        Returns
        -------
        list of str
            The names of the attributes of the specified
            type.
        """

        all_bases = [base_class for base_class in reversed(inspect.getmro(cls))]

        attribute_names = []

        for base_class in all_bases:

            attribute_names.extend(
                [
                    attribute_name
                    for attribute_name in base_class.__dict__
                    if type(base_class.__dict__[attribute_name]) == attribute_type
                ]
            )

        return attribute_names

    @classmethod
    def get_input_attributes(cls):
        """Returns all of the input attributes of the class.

        Returns
        -------
        list of str
            The names of the attributes.
        """
        return cls._get_attributes(InputAttribute)

    @classmethod
    def get_output_attributes(cls):
        """Returns all of the input attributes of the class.

        Returns
        -------
        list of str
            The names of the attributes.
        """
        return cls._get_attributes(OutputAttribute)

    @classmethod
    def get_all_attributes(cls):
        """Returns all of the attributes of the class.

        Returns
        -------
        list of str
            The names of the attributes.
        """
        return cls.get_input_attributes() + cls.get_output_attributes()

    def __getstate__(self):

        input_attributes = {
            name: getattr(self, name) for name in self.get_input_attributes()
        }
        output_attributes = {
            name: getattr(self, name) for name in self.get_output_attributes()
        }

        state = {}

        if len(input_attributes) > 0:
            state["input_attributes"] = input_attributes
        if len(output_attributes) > 0:
            state["output_attributes"] = output_attributes

        return state

    def __setstate__(self, state):

        input_attributes = {
            name: getattr(self, name) for name in self.get_input_attributes()
        }
        output_attributes = {
            name: getattr(self, name) for name in self.get_output_attributes()
        }

        state_input = state.get("input_attributes", {})
        state_output = state.get("output_attributes", {})

        for name in input_attributes:

            attribute = getattr(self.__class__, name)

            if not attribute.optional and name not in state_input:

                raise IndexError(
                    f"The required {name} input was not present in "
                    f"the state dictionary."
                )

            # This should handle type checking.
            setattr(self, name, state_input[name])

        for name in output_attributes:

            if name not in state_output:
                continue

            setattr(self, name, state_output[name])


class BaseAttribute(abc.ABC):
    """A custom descriptor used to mark class attributes as being either
    a required input, or provided output of a protocol.

    This decorator expects the object to have a matching private field
    in addition to the public attribute. For example if an object has
    an attribute `substance`, the object must also have a `_substance`
    field.

    Notes
    -----
    The attribute class will automatically create this private
    attribute on the object and populate it with the default value.
    """

    def __init__(self, docstring, type_hint):
        """Initializes a new BaseAttribute object.

        Parameters
        ----------
        docstring: str
            A docstring describing the attributes purpose. This will automatically
            be decorated with additional information such as type hints, default
            values, etc.
        type_hint: type, typing.Union
            The expected type of this attribute. This will be used to help the
            workflow engine ensure that expected input types match corresponding
            output values.
        """

        if not is_supported_type(type_hint):

            raise ValueError(
                f"The {type_hint} type is not supported by the "
                f"workflow type hinting system."
            )

        if hasattr(type_hint, "__qualname__"):

            if type_hint.__qualname__ == "build_quantity_class.<locals>.Quantity":
                typed_docstring = f"Quantity: {docstring}"
            elif type_hint.__qualname__ == "build_quantity_class.<locals>.Unit":
                typed_docstring = f"Unit: {docstring}"
            else:
                typed_docstring = f"{type_hint.__qualname__}: {docstring}"

        elif hasattr(type_hint, "__name__"):
            typed_docstring = f"{type_hint.__name__}: {docstring}"
        else:
            typed_docstring = f"{str(type_hint)}: {docstring}"

        self.__doc__ = typed_docstring
        self.type_hint = type_hint

    def __set_name__(self, owner, name):
        self._private_attribute_name = "_" + name

    def __get__(self, instance, owner=None):

        if instance is None:
            # Handle the case where this is called on the class directly,
            # rather than an instance.
            return self

        try:
            return getattr(instance, self._private_attribute_name)
        except AttributeError:
            return UNDEFINED

    def __set__(self, instance, value):

        if (
            not is_instance_of_type(value, self.type_hint)
            and not isinstance(value, PlaceholderInput)
            and not value == UNDEFINED
        ):

            raise ValueError(
                f"The {self._private_attribute_name[1:]} attribute can only accept "
                f"values of type {self.type_hint}"
            )

        setattr(instance, self._private_attribute_name, value)


class InputAttribute(BaseAttribute):
    """A descriptor used to mark an attribute of an object as
    an input to that object.

    An attribute can either be set with a value directly, or it
    can also be set to a `ProtocolPath` to be set be the workflow
    manager.

    Examples
    ----------
    To mark an attribute as an input:

    >>> from propertyestimator.attributes import BaseAttributeClass, InputAttribute
    >>>
    >>> class MyObject(BaseAttributeClass):
    >>>
    >>>     my_input = InputAttribute(
    >>>         docstring='An input will be used.',
    >>>         type_hint=float,
    >>>         default_value=0.1
    >>>     )
    """

    def __init__(
        self,
        docstring,
        type_hint,
        default_value,
        optional=False,
        merge_behavior=MergeBehaviour.ExactlyEqual,
    ):

        """Initializes a new InputAttribute object.

        Parameters
        ----------
        default_value: Any
            The default value for this attribute.
        optional: bool
            Defines whether this is an optional input of a class. If true,
            the `default_value` must be set to `UNDEFINED`.
        merge_behavior: BaseMergeBehaviour
            An enum describing how this input should be handled when considering
            whether to, and actually merging two different objects.
        """

        docstring = f"**Input** - {docstring}"

        if not isinstance(merge_behavior, BaseMergeBehaviour):
            raise ValueError(
                "The merge behaviour must inherit from `BaseMergeBehaviour`"
            )

        # Automatically extend the docstrings.
        if isinstance(
            default_value, (int, float, str, unit.Quantity, EstimatedQuantity, Enum)
        ) or (
            isinstance(default_value, (list, tuple, set, frozenset))
            and len(default_value) <= 4
        ):

            docstring = (
                f"{docstring} The default value of this attribute "
                f"is ``{str(default_value)}``."
            )

        elif default_value == UNDEFINED:

            optional_string = "" if optional else " and must be set by the user."

            docstring = (
                f"{docstring} The default value of this attribute "
                f"is not set{optional_string}."
            )

        if (
            merge_behavior == InequalityMergeBehaviour.SmallestValue
            or merge_behavior == InequalityMergeBehaviour.LargestValue
        ):

            merge_docstring = ""

            if merge_behavior == InequalityMergeBehaviour.SmallestValue:
                merge_docstring = (
                    "When two protocols are merged, the smallest value of "
                    "this attribute from either protocol is retained."
                )

            if merge_behavior == InequalityMergeBehaviour.SmallestValue:
                merge_docstring = (
                    "When two protocols are merged, the largest value of "
                    "this attribute from either protocol is retained."
                )

            docstring = f"{docstring} {merge_docstring}"

        if optional is True:
            docstring = f"{docstring} This input is *optional*."

        super().__init__(docstring, type_hint)

        self.optional = optional
        self.merge_behavior = merge_behavior

        self._default_value = default_value

    def __get__(self, instance, owner=None):

        if instance is None:
            # Handle the case where this is called on the class directly,
            # rather than an instance.
            return self

        if not hasattr(instance, self._private_attribute_name):
            # Make sure to only ever pass a copy of the default value to ensure
            # mutable values such as lists don't get set by reference.
            setattr(
                instance,
                self._private_attribute_name,
                copy.deepcopy(self._default_value),
            )

        return getattr(instance, self._private_attribute_name)


class OutputAttribute(BaseAttribute):
    """A descriptor used to mark an attribute of an as
    an output of that object. This attribute is
    expected to be populated by the class itself.

    Examples
    ----------
    To mark an attribute as an output:

    >>> from propertyestimator.attributes import BaseAttributeClass, OutputAttribute,
    >>>
    >>> class MyObject(BaseAttributeClass):
    >>>
    >>>     my_output = OutputAttribute(
    >>>         docstring='An output that will be filled.',
    >>>         type_hint=float
    >>>     )
    """

    def __init__(self, docstring, type_hint):
        """Initializes a new OutputAttribute object.
        """
        docstring = f"**Output** - {docstring}"
        super().__init__(docstring, type_hint)
