"""
A collection of commonly raised python exceptions.
"""
import traceback

from openff.evaluator.utils.serialization import TypedBaseModel


class EvaluatorException(TypedBaseModel):
    """A serializable wrapper around an `Exception`.
    """

    @classmethod
    def from_exception(cls, exception):
        """Initialize this class from an existing exception.

        Parameters
        ----------
        exception: Exception
            The existing exception

        Returns
        -------
        cls
            The initialized exception object.
        """

        message = traceback.format_exception(None, exception, exception.__traceback__)
        return cls(message)

    def __init__(self, message=None):
        """Constructs a new EvaluatorException object.

        Parameters
        ----------
        message: str or list of str
            Information about the raised exception.
        """
        self.message = message

    def __getstate__(self):
        return {"message": self.message}

    def __setstate__(self, state):
        self.message = state["message"]

    def __str__(self):

        message = self.message

        if isinstance(message, list):
            message = "".join(message)

        return str(message)
