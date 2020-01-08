"""
A collection of descriptors used to mark-up class fields which
hold importance to the workflow engine, such as the inputs or
outputs of workflow protocols.
"""
import copy
from enum import Enum

from propertyestimator.attributes import UNDEFINED, Attribute


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
    * Custom: This attribute will be ignored by the built-in merging code such that
     user specified behavior can be implemented.
    """

    ExactlyEqual = "ExactlyEqual"
    Custom = "Custom"


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


class InputAttribute(Attribute):
    """A descriptor used to mark an attribute of an object as
    an input to that object.

    An attribute can either be set with a value directly, or it
    can also be set to a `ProtocolPath` to be set be the workflow
    manager.

    Examples
    ----------
    To mark an attribute as an input:

    >>> from propertyestimator.attributes import AttributeClass
    >>> from propertyestimator.workflow.attributes import InputAttribute
    >>>
    >>> class MyObject(AttributeClass):
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
        merge_behavior: BaseMergeBehaviour
            An enum describing how this input should be handled when considering
            whether to, and actually merging two different objects.
        """

        docstring = f"**Input** - {docstring}"

        if not isinstance(merge_behavior, BaseMergeBehaviour):
            raise ValueError(
                "The merge behaviour must inherit from `BaseMergeBehaviour`"
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

        super().__init__(docstring, type_hint, default_value, optional)

        self.merge_behavior = merge_behavior

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


class OutputAttribute(Attribute):
    """A descriptor used to mark an attribute of an as
    an output of that object. This attribute is expected
    to be populated by the object itself, rather than be
    set externally.

    Examples
    ----------
    To mark an attribute as an output:

    >>> from propertyestimator.attributes import AttributeClass
    >>> from propertyestimator.workflow.attributes import OutputAttribute
    >>>
    >>> class MyObject(AttributeClass):
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
        super().__init__(docstring, type_hint, UNDEFINED, optional=False)
