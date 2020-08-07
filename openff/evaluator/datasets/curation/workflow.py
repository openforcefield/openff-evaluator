import logging
from typing import List, Union, overload

import numpy
import pandas
from pydantic import BaseModel, Field

from openff.evaluator.datasets import PhysicalPropertyDataSet
from openff.evaluator.datasets.curation.components import CurationComponent
from openff.evaluator.datasets.curation.components.conversion import (
    ConversionComponentSchema,
)
from openff.evaluator.datasets.curation.components.filtering import (
    FilterComponentSchema,
)
from openff.evaluator.datasets.curation.components.freesolv import (
    FreeSolvComponentSchema,
)
from openff.evaluator.datasets.curation.components.selection import (
    SelectionComponentSchema,
)
from openff.evaluator.datasets.curation.components.thermoml import (
    ThermoMLComponentSchema,
)

logger = logging.getLogger(__name__)


class CurationWorkflowSchema(BaseModel):
    """A schemas which encodes how a set of curation components should be applied
    sequentially to a data set."""

    component_schemas: List[
        Union[
            ConversionComponentSchema,
            FilterComponentSchema,
            FreeSolvComponentSchema,
            SelectionComponentSchema,
            ThermoMLComponentSchema,
        ]
    ] = Field(
        default_factory=list,
        description="The schemas of the components to apply as part of this workflow. "
        "The components will be applied in the order they appear in this list.",
    )


class CurationWorkflow:
    """A convenience class for applying a set of curation components
    sequentially to a data set."""

    @classmethod
    @overload
    def apply(
        cls,
        data_set: PhysicalPropertyDataSet,
        schema: CurationWorkflowSchema,
        n_processes: int = 1,
    ) -> PhysicalPropertyDataSet:
        ...

    @classmethod
    @overload
    def apply(
        cls,
        data_set: pandas.DataFrame,
        schema: CurationWorkflowSchema,
        n_processes: int = 1,
    ) -> pandas.DataFrame:
        ...

    @classmethod
    def apply(cls, data_set, schema, n_processes=1):
        """Apply each component of this curation workflow to an initial data set in
        sequence.

        Parameters
        ----------
        data_set
            The data set to apply the workflow to. This may either be a
            data set object or it's pandas representation.
        schema
            The schema which defines the components to apply.
        n_processes
            The number of processes that each component is allowed to
            parallelize across.

        Returns
        -------
            The data set which has had the curation workflow applied to it.
        """

        component_classes = CurationComponent.components

        data_frame = data_set

        if isinstance(data_frame, PhysicalPropertyDataSet):
            data_frame = data_frame.to_pandas()

        data_frame = data_frame.copy()
        data_frame = data_frame.fillna(value=numpy.nan)

        for component_schema in schema.component_schemas:

            component_class_name = component_schema.__class__.__name__.replace(
                "Schema", ""
            )
            component_class = component_classes[component_class_name]

            logger.info(f"Applying {component_class_name}")

            data_frame = component_class.apply(
                data_frame, component_schema, n_processes
            )

            logger.info(f"{component_class_name} applied")

            data_frame = data_frame.fillna(value=numpy.nan)

        if isinstance(data_set, PhysicalPropertyDataSet):
            data_frame = PhysicalPropertyDataSet.from_pandas(data_frame)

        return data_frame
