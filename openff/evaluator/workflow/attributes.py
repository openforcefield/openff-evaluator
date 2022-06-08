"""
A collection of descriptors used to mark-up class fields which
hold importance to the workflow engine, such as the inputs or
outputs of workflow protocols.
"""
import warnings
from enum import Enum

from openff.evaluator.attributes import UNDEFINED, Attribute


def __getattr__(clsname):
    if "Behaviour" in clsname:
        us_clsname = clsname.replace("Behaviour", "Behavior")
        us_cls = globals().get(us_clsname)
        if us_cls is not None:
            warnings.filterwarnings("default", category=DeprecationWarning)
            warnings.warn(
                f"{clsname} is a DEPRECATED spelling and will be removed "
                f"in a future release. Please use {us_clsname} instead.",
                DeprecationWarning,
            )
            return us_cls
    raise AttributeError(f"module {__name__} has no attribute {clsname}")


class BaseMergeBehavior(Enum):
    """A base class for enums which will describes how attributes should
    be handled when attempting to merge similar protocols.
    """

    pass


class MergeBehavior(BaseMergeBehavior):
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


class InequalityMergeBehavior(BaseMergeBehavior):
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

    >>> from openff.evaluator.attributes import AttributeClass
    >>> from openff.evaluator.workflow.attributes import InputAttribute
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
        merge_behavior=MergeBehavior.ExactlyEqual,
    ):
        """Initializes a new InputAttribute object.

        Parameters
        ----------
        merge_behavior: BaseMergeBehavior
            An enum describing how this input should be handled when considering
            whether to, and actually merging two different objects.
        """

        docstring = f"**Input** - {docstring}"

        if not isinstance(merge_behavior, BaseMergeBehavior):
            raise ValueError(
                "The merge behaviour must inherit from `BaseMergeBehavior`"
            )

        if (
            merge_behavior == InequalityMergeBehavior.SmallestValue
            or merge_behavior == InequalityMergeBehavior.LargestValue
        ):

            merge_docstring = ""

            if merge_behavior == InequalityMergeBehavior.SmallestValue:
                merge_docstring = (
                    "When two protocols are merged, the smallest value of "
                    "this attribute from either protocol is retained."
                )

            if merge_behavior == InequalityMergeBehavior.SmallestValue:
                merge_docstring = (
                    "When two protocols are merged, the largest value of "
                    "this attribute from either protocol is retained."
                )

            docstring = f"{docstring} {merge_docstring}"

        super().__init__(docstring, type_hint, default_value, optional)

        self.merge_behavior = merge_behavior


class OutputAttribute(Attribute):
    """A descriptor used to mark an attribute of an as
    an output of that object. This attribute is expected
    to be populated by the object itself, rather than be
    set externally.

    Examples
    ----------
    To mark an attribute as an output:

    >>> from openff.evaluator.attributes import AttributeClass
    >>> from openff.evaluator.workflow.attributes import OutputAttribute
    >>>
    >>> class MyObject(AttributeClass):
    >>>
    >>>     my_output = OutputAttribute(
    >>>         docstring='An output that will be filled.',
    >>>         type_hint=float
    >>>     )
    """

    def __init__(self, docstring, type_hint):
        """Initializes a new OutputAttribute object."""
        docstring = f"**Output** - {docstring}"
        super().__init__(docstring, type_hint, UNDEFINED, optional=False)
