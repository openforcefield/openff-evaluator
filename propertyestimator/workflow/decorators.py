"""
A collection of descriptors used to mark-up elements in a workflow, such
as the inputs or outputs of workflow protocols.
"""
import abc

from enum import Enum

from propertyestimator import unit
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow.typing import is_instance_of_type, is_supported_type
from propertyestimator.workflow.utils import PlaceholderInput


class MergeBehaviour(Enum):
    """A enum which describes how attributes should be handled when
    attempting to merge similar protocols.

    Attributes
    ----------
    ExactlyEqual: str
        This attribute must be exactly equal between two protocols for
        them to be able to merge.
    """

    ExactlyEqual = 'ExactlyEqual'


class InequalityMergeBehaviour(MergeBehaviour):
    """A enum which describes how attributes which can be compared
    with inequalities should be merged.

    Attributes
    ----------
    SmallestValue: str
        When two protocols are merged, the smallest value of this
        attribute from either protocol is retained.
    LargestValue: str
        When two protocols are merged, the largest value of this
        attribute from either protocol is retained.
    """

    SmallestValue = 'SmallestValue'
    LargestValue = 'LargestValue'


class BaseProtocolAttribute(abc.ABC):
    """A custom descriptor used to mark class attributes as being either
    a required input, or provided output of a protocol.

    Notes
    -----
    This decorator expects the protocol to have a matching private field
    in addition to the public attribute. For example if a protocol has
    an attribute `substance`, by default the protocol must also have a
    `_substance` field.
    """

    _attribute_missing_string = 'The protocol does not have a {private_attribute_name} ' \
                                'attribute to match the {attribute_name} descriptor. ' \
                                'This should be defined in the `__init__` method.'

    def __init__(self, docstring, type_hint):
        """Initializes a new BaseProtocolAttribute object.

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

            raise ValueError(f'The {type_hint} type is not supported by the '
                             f'workflow type hinting system.')

        typed_docstring = f'{type_hint}: {docstring}'
        self.__doc__ = typed_docstring
        self._type_hint = type_hint

    def __set_name__(self, owner, name):
        self._private_attribute_name = '_' + name

    def __get__(self, instance, owner=None):

        if instance is None:
            # Handle the case where this is called on the class directly,
            # rather than an instance.
            return self

        try:
            return getattr(instance, self._private_attribute_name)

        except AttributeError:

            raise AttributeError(BaseProtocolAttribute._attribute_missing_string.format(
                private_attribute_name=self._private_attribute_name, attribute_name=self._private_attribute_name[1:]
            ))

    def __set__(self, instance, value):

        if not hasattr(instance, self._private_attribute_name):

            raise AttributeError(BaseProtocolAttribute._attribute_missing_string.format(
                private_attribute_name=self._private_attribute_name, attribute_name=self._private_attribute_name[1:]
            ))

        if (not is_instance_of_type(value, self._type_hint) and
            not isinstance(value, PlaceholderInput)):

            raise ValueError(f'The {self._private_attribute_name[1:]} attribute can only accept '
                             f'values of type {self._type_hint}')

        setattr(instance, self._private_attribute_name, value)


class _InputAttribute(BaseProtocolAttribute):
    """A descriptor used to mark an attribute of a protocol as
    an input to that protocol.

    Examples
    ----------
    To mark an attribute as an input:

    >>> from propertyestimator.workflow.protocols import BaseProtocol
    >>> from propertyestimator.workflow.decorators import protocol_input
    >>>
    >>> class MyProtocol(BaseProtocol):
    >>>
    >>>     my_input = protocol_input(docstring='An input will be used.', type_hint=float, default_value=0.1)
    >>>
    >>>     def __init__(self, protocol_id):
    >>>         super().__init__(protocol_input)
    >>>         self._my_input = 0.1
    """

    class UNDEFINED:
        """A custom type used to differentiate between ``None`` values,
        and an undeclared optional value."""
        pass

    def __init__(self, docstring, type_hint, default_value, optional=False,
                 merge_behavior=MergeBehaviour.ExactlyEqual):

        """Initializes a new protocol_input object.

        Parameters
        ----------
        default_value: Any
            The default value for this attribute.
        optional: bool
            Defines whether this is an optional input of a class. If true,
            the `default_value` must be set to `UNDEFINED`.
        merge_behavior: MergeBehaviour
            An enum describing how this input should be handled when considering
            whether to, and actually merging two different protocols.
        """

        # Automatically extend the docstrings.
        if (isinstance(default_value, (int, float, str, unit.Quantity, EstimatedQuantity, Enum)) or
            (isinstance(default_value, (list, tuple, set, frozenset)) and len(default_value) <= 4)):

            docstring = f'{docstring} The default value of this attribute ' \
                        f'is {str(default_value)}.'

        elif default_value == self.UNDEFINED:

            optional_string = '' if optional else ' and must be set by the user.'

            docstring = f'{docstring} The default value of this attribute ' \
                         f'is undefined{optional_string}.'

        if optional is True:

            docstring = f'{docstring} This input is optional.'

        if type(merge_behavior) in [MergeBehaviour, InequalityMergeBehaviour]:
            docstring = f'{docstring} {merge_behavior.__doc__}'

        super().__init__(docstring, type_hint)

        self._optional = optional
        self._merge_behavior = merge_behavior


class _OutputAttribute(BaseProtocolAttribute):
    """A descriptor used to mark an attribute of a protocol as
    an output of that protocol.

    Examples
    ----------
    To mark an attribute as an output:

    >>> from propertyestimator.workflow.protocols import BaseProtocol
    >>> from propertyestimator.workflow.decorators import protocol_output
    >>>
    >>> class MyProtocol(BaseProtocol):
    >>>
    >>>     my_output = protocol_output(docstring='An output that will be filled.', type_hint=float)
    >>>
    >>>     def __init__(self, protocol_id):
    >>>         super().__init__(protocol_input)
    >>>         self._my_output = 0.1
    """

    def __init__(self, docstring, type_hint):
        """Initializes a new protocol_output object.
        """
        super().__init__(docstring, type_hint)


protocol_input = _InputAttribute
protocol_output = _OutputAttribute
