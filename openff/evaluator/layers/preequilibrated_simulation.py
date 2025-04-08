"""A calculation layer which employs molecular simulation
from pre-equilibrated systems to estimate sets of physical properties.
"""

import copy
import logging

from openff.evaluator.attributes import UNDEFINED, Attribute
from openff.evaluator.layers import calculation_layer
from openff.evaluator.layers.equilibration import (
    ConditionAggregationBehavior,
    EquilibrationLayer,
    EquilibrationProperty,
    default_storage_query,
)
from openff.evaluator.layers.workflow import (
    WorkflowCalculationLayer,
    WorkflowCalculationSchema,
)
from openff.evaluator.storage.query import EquilibrationDataQuery

logger = logging.getLogger(__name__)


class PreequilibratedSimulationSchema(WorkflowCalculationSchema):
    """A schema which encodes the options and the workflow schema
    that the `PreequilibratedSimulationLayer` should use when estimating a given class
    of physical properties using the built-in workflow framework.
    """

    n_uncorrelated_samples: int = Attribute(
        docstring="The minimum number of uncorrelated samples to use in evaluating property.",
        type_hint=int,
        optional=True,
        default_value=UNDEFINED,
    )

    equilibration_error_tolerances = Attribute(
        docstring="The error tolerances to use when equilibrating the box.",
        type_hint=list,
        default_value=[],
    )
    equilibration_error_aggregration = Attribute(
        docstring="How to aggregate errors -- any vs all.",
        type_hint=ConditionAggregationBehavior,
        default_value=ConditionAggregationBehavior.All,
    )

    equilibration_error_on_failure = Attribute(
        docstring="Whether to raise an error if the convergence conditions are not met.",
        type_hint=bool,
        default_value=False,
    )
    equilibration_max_iterations = Attribute(
        docstring="The maximum number of iterations to run the equilibration for.",
        type_hint=int,
        default_value=100,
    )

    storage_queries = Attribute(
        docstring="The queries to perform when retrieving data for each "
        "of the components in the system from the storage backend. The "
        "keys of this dictionary will correspond to the metadata keys made "
        "available to the workflow system.",
        type_hint=dict,
        default_value=default_storage_query(),
    )

    number_of_molecules = Attribute(
        docstring="The number of molecules in the system.",
        type_hint=int,
    )

    max_iterations = Attribute(
        docstring="The maximum number of iterations to run the simulation for.",
        type_hint=int,
        default_value=100,
    )

    def validate(self, attribute_type=None):
        super().validate(attribute_type)

        assert len(self.storage_queries) > 0
        assert all(
            isinstance(x, EquilibrationDataQuery) for x in self.storage_queries.values()
        )
        if self.equilibration_error_tolerances:
            for error_tolerance in self.equilibration_error_tolerances:
                assert isinstance(error_tolerance, EquilibrationProperty)


@calculation_layer()
class PreequilibratedSimulationLayer(WorkflowCalculationLayer):
    """A calculation layer which employs molecular simulation
    to estimate sets of physical properties.
    """

    @classmethod
    def required_schema_type(cls):
        return PreequilibratedSimulationSchema

    @staticmethod
    def _get_workflow_metadata(
        working_directory,
        physical_property,
        force_field_path,
        parameter_gradient_keys,
        storage_backend,
        calculation_schema,
    ):
        """
        Get the metadata required to run a workflow calculation.
        This method injects a preequilibrated_box_file into the metadata.
        """
        global_metadata = WorkflowCalculationLayer._get_workflow_metadata(
            working_directory,
            physical_property,
            force_field_path,
            parameter_gradient_keys,
            storage_backend,
            calculation_schema,
        )

        global_metadata["equilibration_error_tolerances"] = copy.deepcopy(
            calculation_schema.equilibration_error_tolerances
        )
        global_metadata["equilibration_error_aggregration"] = (
            calculation_schema.equilibration_error_aggregration
        )

        EquilibrationLayer._update_metadata_with_template_queries(
            global_metadata,
            working_directory,
            physical_property,
            force_field_path,
            storage_backend,
            calculation_schema,
        )

        return global_metadata
