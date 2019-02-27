"""
A collection of commonly raised python exceptions.
"""
from propertyestimator.utils.serialization import TypedBaseModel


class XmlNodeMissingException(Exception):

    def __init__(self, node_name):

        message = 'The calculation template does not contain a <' + str(node_name) + '> node.'
        super().__init__(message)


class PropertyEstimatorException(TypedBaseModel):
    """A json serializable object wrapper containing information about
    a failed property calculation.

    .. todo:: Flesh out more fully.
    """
    directory: str
    message: str
