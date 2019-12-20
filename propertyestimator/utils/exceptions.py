"""
A collection of commonly raised python exceptions.
"""
from propertyestimator.utils.serialization import TypedBaseModel


class XmlNodeMissingException(Exception):
    def __init__(self, node_name):

        message = (
            "The calculation template does not contain a <" + str(node_name) + "> node."
        )
        super().__init__(message)


class EvaluatorException(TypedBaseModel):
    """A json serializable object wrapper containing information about
    a failed property calculation.

    .. todo:: Flesh out more fully.
    """

    def __init__(self, directory="", message=""):
        """Constructs a new EvaluatorException object.

        Parameters
        ----------
        directory: str
            The directory in which this exception was raised.
        message:
            Information about the raised exception.
        """

        self.directory = directory
        self.message = message

    def __getstate__(self):

        return {"directory": self.directory, "message": self.message}

    def __setstate__(self, state):

        self.directory = state["directory"]
        self.message = state["message"]
