import logging
from typing import List, Union

import numpy
import pandas
from pydantic import BaseModel, Field

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
    @classmethod
    def apply(
        cls, data_frame: pandas.DataFrame, schema: CurationWorkflowSchema, n_processes=1
    ) -> pandas.DataFrame:
        """Apply each component of this workflow to an initial data frame in
        sequence.

        Parameters
        ----------
        data_frame: pandas.DataFrame
            The data frame to apply the workflow to.
        schema: WorkflowSchema
            The schema which defines the components to apply.
        n_processes: int
            The number of processes that each component is allowed to
            parallelize across.

        Returns
        -------
        pandas.DataFrame
            The data frame which has had the components applied to it.
        """

        component_classes = CurationComponent.components

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

        return data_frame
