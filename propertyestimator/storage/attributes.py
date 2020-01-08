"""
A collection of descriptors used to add extra metadata
to storage class attributes.
"""
from propertyestimator.attributes import UNDEFINED, Attribute


class FilePath(str):
    """Represents a string file path.
    """

    pass


class StorageAttribute(Attribute):
    """A descriptor used to mark attributes of a class as those
    which store information about a cached piece of data.
    """

    def __init__(
        self, docstring, type_hint, optional=False,
    ):
        super().__init__(docstring, type_hint, UNDEFINED, optional)

    def _set_value(self, instance, value):

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

        super(StorageAttribute, self)._set_value(instance, value)


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
