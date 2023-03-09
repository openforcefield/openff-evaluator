"""
A collection of commonly raised python exceptions.
"""
import traceback
from typing import Optional

from openff.evaluator.utils.serialization import TypedBaseModel


class EvaluatorException(TypedBaseModel, BaseException):
    """A serializable wrapper around an `Exception`."""

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
        super(EvaluatorException, self).__init__(message)
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


class MissingOptionalDependency(EvaluatorException):
    """An exception raised when an optional dependency is required
    but cannot be found.

    Attributes
    ----------
    library_name
        The name of the missing library.
    license_issue
        Whether the library was importable but was unusable due
        to a missing license.
    """

    def __init__(
        self,
        library_name: str,
        license_issue: bool = False,
        extra: Optional[str] = None,
    ):
        """

        Parameters
        ----------
        library_name
            The name of the missing library.
        license_issue
            Whether the library was importable but was unusable due
            to a missing license.
        extra
            An extra string to append to the error message.
        """

        message = f"The optional {library_name} module could not be imported."

        if license_issue:
            message = f"{message} This is due to a missing license."

        if extra:
            message = f"{message} {extra}"

        super(MissingOptionalDependency, self).__init__(message)

        self.library_name = library_name
        self.license_issue = license_issue
