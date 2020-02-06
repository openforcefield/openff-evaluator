"""A calculation layer which employs molecular simulation
to estimate sets of physical properties.
"""
from evaluator.layers import calculation_layer
from evaluator.layers.workflow import (
    WorkflowCalculationLayer,
    WorkflowCalculationSchema,
)


class SimulationSchema(WorkflowCalculationSchema):
    """A schema which encodes the options and the workflow schema
    that the `SimulationLayer` should use when estimating a given class
    of physical properties using the built-in workflow framework.
    """

    pass


@calculation_layer()
class SimulationLayer(WorkflowCalculationLayer):
    """A calculation layer which employs molecular simulation
to estimate sets of physical properties.
    """

    @classmethod
    def required_schema_type(cls):
        return SimulationSchema
