from propertyestimator.utils.exceptions import EvaluatorException


class WorkflowException(EvaluatorException):
    """An exception which was raised while executing a workflow
    protocol.
    """

    def __init__(self, message=None, protocol_id=None):
        """Constructs a new EvaluatorException object.

        Parameters
        ----------
        message: str or list of str
            Information about the raised exception.
        protocol_id: str
            The id of the protocol which was the exception.
        """
        super(WorkflowException, self).__init__(message)
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

    def __str__(self):

        base_str = super(WorkflowException, self).__str__()
        return f"{self.protocol_id} failed to execute.\n\n{base_str}"
