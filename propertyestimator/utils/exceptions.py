"""
A collection of commonly raised python exceptions.
"""
import traceback

from propertyestimator.utils.serialization import TypedBaseModel


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


class WorkflowException(EvaluatorException):
    """An exception which was raised while executing a workflow
    protocol.
    """

    def __init__(self, message, protocol_id):
        """Constructs a new EvaluatorException object.

        Parameters
        ----------
        message: str or list of str
            Information about the raised exception.
        protocol_id: str
            The id of the protocol which was the exception.
        """
        super(WorkflowException, self).__init__(message)

        self.message = message
        self.protocol_id = protocol_id

    def __getstate__(self):

        state = super(WorkflowException, self).__getstate__()

        if self.protocol_id is not None:
            state["protocol_id"] = self.protocol_id

        return state

    def __setstate__(self, state):

        super(WorkflowException, self).__setstate__(state)

        if "protocol_id" in state:
            self.protocol_id = state["protocol_id"]
