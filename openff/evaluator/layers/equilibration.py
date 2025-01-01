"""A calculation layer for equilibration.
"""
import logging
from openff.evaluator.layers import calculation_layer
from openff.evaluator.layers.workflow import (
    WorkflowCalculationLayer,
    WorkflowCalculationSchema,
)

logger = logging.getLogger(__name__)

class EquilibrationSchema(WorkflowCalculationSchema):
    """A schema which encodes the options and the workflow schema
    that the `EquilibrationLayer` should use when estimating a given class
    of physical properties using the built-in workflow framework.
    """

    pass


@calculation_layer()
class EquilibrationLayer(WorkflowCalculationLayer):
    """A calculation layer which employs molecular simulation
    to estimate sets of physical properties.
    """

    @classmethod
    def required_schema_type(cls):
        return EquilibrationSchema
    
    @classmethod
    def _move_property_from_queued_to_estimated(cls, returned_output, batch):
        logger.info("EquilibrationLayers do not estimate properties.")
