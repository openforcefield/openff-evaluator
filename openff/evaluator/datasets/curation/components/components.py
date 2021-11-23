import abc
import logging
from typing import overload

import pandas
from pydantic import BaseModel

from openff.evaluator.datasets import PhysicalPropertyDataSet

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
    """A base class for schemas which specify how particular curation
    components should be applied to a data set."""


class CurationComponent(metaclass=_MetaCurationComponent):
    """A base component for curation components which apply a particular operation
    (such as filtering or data conversion) to a data set."""

    @classmethod
    @abc.abstractmethod
    def _apply(
        cls, data_frame: pandas.DataFrame, schema, n_processes
    ) -> pandas.DataFrame:
        raise NotImplementedError()

    @classmethod
    @overload
    def apply(
        cls,
        data_set: PhysicalPropertyDataSet,
        schema: CurationComponentSchema,
        n_processes: int = 1,
    ) -> PhysicalPropertyDataSet:
        ...

    @classmethod
    @overload
    def apply(
        cls,
        data_set: pandas.DataFrame,
        schema: CurationComponentSchema,
        n_processes: int = 1,
    ) -> pandas.DataFrame:
        ...

    @classmethod
    def apply(cls, data_set, schema, n_processes=1):
        """Apply this curation component to a data set.

        Parameters
        ----------
        data_set
            The data frame to apply the component to.
        schema
            The schema which defines how this component should be applied.
        n_processes
            The number of processes that this component is allowed to
            parallelize across.

        Returns
        -------
            The data set which has had the component applied to it.
        """

        data_frame = data_set

        if isinstance(data_frame, PhysicalPropertyDataSet):
            data_frame = data_frame.to_pandas()

        modified_data_frame = cls._apply(data_frame, schema, n_processes)

        n_data_points = len(data_frame)
        n_filtered = len(modified_data_frame)

        if n_filtered != n_data_points:
            direction = "removed" if n_filtered < n_data_points else "added"

            logger.info(
                f"{abs(n_filtered - n_data_points)} data points were {direction} after "
                f"applying the {cls.__name__} component."
            )

        if isinstance(data_set, PhysicalPropertyDataSet):
            modified_data_frame = PhysicalPropertyDataSet.from_pandas(
                modified_data_frame
            )

        return modified_data_frame
