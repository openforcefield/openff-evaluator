"""
A collection of descriptors used to add extra metadata
to storage class attributes.
"""
from propertyestimator.attributes import UNDEFINED, Attribute


class FilePath(str):
    """Represents a string file path.
    """


class StorageAttribute(Attribute):
    """A descriptor used to mark attributes of a class as those
    which store information about a cached piece of data.
    """

    def __init__(
        self, docstring, type_hint, optional=False,
    ):
        super().__init__(docstring, type_hint, UNDEFINED, optional)


class QueryAttribute(Attribute):
    """A descriptor used to add additional metadata to
    attributes of a storage query.
    """

    def __init__(self, docstring, type_hint, optional=False, custom_match=False):

        """Initializes self.

        Parameters
        ----------
        custom_match: bool
            Whether a custom behaviour will be implemented when
            matching this attribute against the matching data object
            attribute.
        """
        super().__init__(docstring, type_hint, UNDEFINED, optional)
        self.custom_match = custom_match
