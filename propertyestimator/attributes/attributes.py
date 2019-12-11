"""
A collection of descriptors used to add meta data to
object attributes.
"""
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


class PlaceholderValue:
    """A class to act as a place holder for an attribute whose value is
    not known a priori, but will be set later by some specialised code.
    This may include the input to a protocol which will be set by a
    workflow as the output of an executed protocol.
    """

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


UNDEFINED = UndefinedAttribute()


class AttributeClass(TypedBaseModel):
    """A base class for objects which require well defined
    attributes with additional metadata.
    """

    @classmethod
    def _get_attributes(cls, attribute_type=None):
        """Returns all attributes of a specific `attribute_type`.

        Parameters
        ----------
        attribute_type: type of Attribute, optional
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

            found_attributes = [
                attribute_name
                for attribute_name in base_class.__dict__
                if isinstance(base_class.__dict__[attribute_name], Attribute)
            ]

            if attribute_type is not None:

                found_attributes = [
                    name
                    for name in found_attributes
                    if type(base_class.__dict__[name]) == attribute_type
                ]

            attribute_names.extend(found_attributes)

        return attribute_names

    def __getstate__(self):

        attributes = {name: getattr(self, name) for name in self._get_attributes()}

        return attributes

    def __setstate__(self, state):

        attribute_names = self._get_attributes()

        for name in attribute_names:

            attribute = getattr(self.__class__, name)

            if not attribute.optional and name not in state:

                raise IndexError(
                    f"The {name} attribute was not present in " f"the state dictionary."
                )

            # This should handle type checking.
            setattr(self, name, state[name])


class Attribute:
    """A custom descriptor used to add useful metadata to class
    attributes.

    This decorator expects the object to have a matching private field
    in addition to the public attribute. For example if an object has
    an attribute `substance`, the object must also have a `_substance`
    field.

    Notes
    -----
    The attribute class will automatically create this private
    attribute on the object and populate it with the default value.
    """

    def __init__(self, docstring, type_hint, default_value=UNDEFINED, optional=False):
        """Initializes a new Attribute object.

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
        default_value: Any
            The default value for this attribute.
        optional: bool
            Defines whether this is an optional input of a class. If true,
            the `default_value` should be set to `UNDEFINED`.
        """

        if not is_supported_type(type_hint):

            raise ValueError(
                f"The {type_hint} type is not supported by the "
                f"workflow type hinting system."
            )

        if hasattr(type_hint, "__qualname__"):

            if type_hint.__qualname__ == "build_quantity_class.<locals>.Quantity":
                docstring = f"Quantity: {docstring}"
            elif type_hint.__qualname__ == "build_quantity_class.<locals>.Unit":
                docstring = f"Unit: {docstring}"
            else:
                docstring = f"{type_hint.__qualname__}: {docstring}"

        elif hasattr(type_hint, "__name__"):
            docstring = f"{type_hint.__name__}: {docstring}"
        else:
            docstring = f"{str(type_hint)}: {docstring}"

        # Handle the default value.
        self._default_value = default_value

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

        self.optional = optional

        if optional is True:
            docstring = f"{docstring} This attribute is *optional*."

        self.__doc__ = docstring
        self.type_hint = type_hint

    def __set_name__(self, owner, name):
        self._private_attribute_name = "_" + name

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

    def __set__(self, instance, value):

        if (
            not is_instance_of_type(value, self.type_hint)
            and not isinstance(value, PlaceholderValue)
            and not value == UNDEFINED
        ):

            raise ValueError(
                f"The {self._private_attribute_name[1:]} attribute can only accept "
                f"values of type {self.type_hint}"
            )

        setattr(instance, self._private_attribute_name, value)
