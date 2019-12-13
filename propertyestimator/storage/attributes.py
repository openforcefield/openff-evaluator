"""
A collection of descriptors used to mark-up class fields which
hold importance to the workflow engine, such as the inputs or
outputs of workflow protocols.
"""
from enum import Enum

from propertyestimator.attributes import UNDEFINED, Attribute


class FilePath(str):
    """Represents a string file path.
    """


class ComparisonBehaviour(Enum):
    """A enum which describes how attributes should be handled when
    comparing whether two pieces of cached data contain the same
    information.

    This enum may take values of

    * Ignore: This attribute will be ignored when comparing whether
              two pieces of data are redundant.
    * Compare: This attribute will be considered when comparing whether
               two pieces of data are redundant.
    """

    Ignore = "Ignore"
    Compare = "Compare"


class StorageAttribute(Attribute):
    """A descriptor used to mark attributes of a class as those
    which store information about a cached piece of data.
    """

    def __init__(
        self,
        docstring,
        type_hint,
        optional=False,
        comparison_behavior=ComparisonBehaviour.Compare,
    ):

        """Initializes a new InputAttribute object.

        Parameters
        ----------
        comparison_behavior: ComparisonBehaviour
            An enum describing whether this attribute should be considered
            when deciding whether to, and actually merging two different
            pieces of cached data.
        """

        docstring = f"**Input** - {docstring}"

        if not isinstance(comparison_behavior, ComparisonBehaviour):
            raise ValueError(
                "The comparison behaviour must be a `ComparisonBehaviour` value"
            )

        compare_docstring = ""

        if comparison_behavior == ComparisonBehaviour.Ignore:
            compare_docstring = (
                "This attribute will not be considered when deciding whether"
                "to merge to pieces of cached data."
            )
        if comparison_behavior == ComparisonBehaviour.Compare:
            compare_docstring = (
                "This attribute will be considered when deciding whether"
                "to merge to pieces of cached data."
            )

        docstring = f"{docstring} {compare_docstring}"

        super().__init__(docstring, type_hint, UNDEFINED, optional)

        self.comparison_behavior = comparison_behavior

    def __set__(self, instance, value):

        # Handle the special case of turning strings
        # into file path objects for convenience.
        if (
            isinstance(value, str)
            and isinstance(self.type_hint, type)
            and issubclass(self.type_hint, FilePath)
        ):
            # This is necessary as the json library currently doesn't
            # support custom serialization of IntFlag or IntEnum.
            value = FilePath(value)

        super(StorageAttribute, self).__set__(instance, value)
