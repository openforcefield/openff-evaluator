"""A calculation layer for equilibration.
"""

import logging
from math import sqrt

from openff.evaluator.attributes import (
    UNDEFINED,
    Attribute,
    AttributeClass,
    PlaceholderValue,
)
from openff.evaluator.layers import calculation_layer
from openff.evaluator.layers.workflow import (
    WorkflowCalculationLayer,
    WorkflowCalculationSchema,
)
from openff.evaluator.workflow import Workflow

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

    @staticmethod
    def _get_workflow_metadata(
        working_directory,
        physical_property,
        force_field_path,
        parameter_gradient_keys,
        storage_backend,
        calculation_schema,
    ):
        """Returns the global metadata to pass to the workflow.

        Parameters
        ----------
        working_directory: str
            The local directory in which to store all local,
            temporary calculation data from this workflow.
        physical_property : PhysicalProperty
            The property that the workflow will estimate.
        force_field_path : str
            The path to the force field parameters to use in the workflow.
        parameter_gradient_keys: list of ParameterGradientKey
            A list of references to all of the parameters which all observables
            should be differentiated with respect to.
        storage_backend: StorageBackend
            The backend used to store / retrieve data from previous calculations.
        calculation_schema: WorkflowCalculationSchema
            The schema containing all of this layers options.

        Returns
        -------
        dict of str and Any, optional
            The global metadata to make available to a workflow.
            Returns `None` if the required metadata could not be
            found / assembled.
        """
        target_uncertainty = None

        if calculation_schema.absolute_tolerance != UNDEFINED:
            target_uncertainty = calculation_schema.absolute_tolerance
        elif calculation_schema.relative_tolerance != UNDEFINED:
            target_uncertainty = (
                physical_property.uncertainty * calculation_schema.relative_tolerance
            )

        global_metadata = Workflow.generate_default_metadata(
            physical_property,
            force_field_path,
            parameter_gradient_keys,
            None,  # set target to None for now
        )
        global_metadata["target_uncertainty"] = target_uncertainty
        per_component_uncertainty = target_uncertainty / sqrt(
            physical_property.substance.number_of_components + 1
        )
        global_metadata["per_component_uncertainty"] = per_component_uncertainty

        return global_metadata

    @classmethod
    def required_schema_type(cls):
        return EquilibrationSchema

    # @classmethod
    # def _move_property_from_queued_to_estimated(cls, returned_output, batch):
    #     logger.info("EquilibrationLayers do not estimate properties.")
