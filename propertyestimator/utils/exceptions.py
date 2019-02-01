"""
A collection of commonly raised python exceptions.
"""

from pydantic import BaseModel


class XmlNodeMissingException(Exception):

    def __init__(self, node_name):

        message = 'The calculation template does not contain a <' + str(node_name) + '> node.'
        super().__init__(message)


class PropertyEstimatorException(BaseModel):
    """A json serializable object wrapper containing information about
    a failed property calculation.

    .. todo:: Flesh out more fully.
    """
    directory: str
    message: str
