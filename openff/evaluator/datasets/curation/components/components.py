import abc
import logging

import pandas
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class _MetaCurationComponent(type):

    components = {}

    def __init__(cls, name, bases, attrs):

        type.__init__(cls, name, bases, attrs)

        if name in _MetaCurationComponent.components:

            raise ValueError(
                "Cannot have more than one curation component with the same name"
            )

        _MetaCurationComponent.components[name] = cls


class CurationComponentSchema(BaseModel, abc.ABC):
    """A base class for curation component schemas."""


class CurationComponent(metaclass=_MetaCurationComponent):
    @classmethod
    @abc.abstractmethod
    def _apply(
        cls, data_frame: pandas.DataFrame, schema, n_processes
    ) -> pandas.DataFrame:
        raise NotImplementedError()

    @classmethod
    def apply(
        cls, data_frame: pandas.DataFrame, schema, n_processes=1
    ) -> pandas.DataFrame:
        """Apply this component to a data frame.

        Parameters
        ----------
        data_frame: pandas.DataFrame
            The data frame to apply the component to.
        schema: T
            The schema which defines how this component should be applied.
        n_processes: int
            The number of processes that this component is allowed to
            parallelize across.

        Returns
        -------
        pandas.DataFrame
            The data frame which has had the component applied to it.
        """

        modified_data_frame = cls._apply(data_frame, schema, n_processes)

        n_data_points = len(data_frame)
        n_filtered = len(modified_data_frame)

        if n_filtered != n_data_points:

            direction = "removed" if n_filtered < n_data_points else "added"

            logger.info(
                f"{abs(n_filtered - n_data_points)} data points were {direction} after "
                f"applying the {cls.__name__} component."
            )

        return modified_data_frame
