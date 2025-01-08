"""A calculation layer which employs molecular simulation
from pre-equilibrated systems to estimate sets of physical properties.
"""

import collections
import copy
import logging
import os

from openff.evaluator.attributes import (
    UNDEFINED,
    Attribute,
    AttributeClass,
    PlaceholderValue,
)
from openff.evaluator.datasets import PropertyPhase
from openff.evaluator.layers import calculation_layer
from openff.evaluator.layers.layers import CalculationLayerResult
from openff.evaluator.layers.workflow import (
    WorkflowCalculationLayer,
    WorkflowCalculationSchema,
)
from openff.evaluator.storage.query import SimulationDataQuery

logger = logging.getLogger(__name__)


def default_storage_query():
    """Return the default query to use when retrieving cached simulation
     data from the storage backend.

    Currently this query will search for data for the full substance of
    interest in the liquid phase.

    Returns
    -------
    dict of str and SimulationDataQuery
        A single query with a key of `"preequilibrated_box"`.
    """

    query = SimulationDataQuery()
    query.substance = PlaceholderValue()
    query.thermodynamic_state = PlaceholderValue()
    query.max_number_of_molecules = PlaceholderValue()

    query.property_phase = PropertyPhase.Liquid
    query.calculation_layer = "EquilibrationLayer"

    return {"preequilibrated_box": query}


class PreequilibratedSimulationSchema(WorkflowCalculationSchema):
    """A schema which encodes the options and the workflow schema
    that the `PreequilibratedSimulationLayer` should use when estimating a given class
    of physical properties using the built-in workflow framework.
    """

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

    def validate(self, attribute_type=None):
        super().validate(attribute_type)

        assert len(self.storage_queries) > 0
        assert all(
            isinstance(x, SimulationDataQuery) for x in self.storage_queries.values()
        )


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

        # input data from equilibration
        template_queries = calculation_schema.storage_queries

        # Apply the storage queries
        required_force_field_keys = set()

        for key in template_queries:
            query = copy.deepcopy(template_queries[key])

            # Fill in any place holder values.
            if isinstance(query.substance, PlaceholderValue):
                query.substance = physical_property.substance
            if isinstance(query.thermodynamic_state, PlaceholderValue):
                query.thermodynamic_state = physical_property.thermodynamic_state
            if isinstance(query.max_number_of_molecules, PlaceholderValue):
                query.max_number_of_molecules = calculation_schema.number_of_molecules

            # Apply the query.
            query_results = storage_backend.query(query)

            if len(query_results) == 0:
                # We haven't found and cached data which is compatible
                # with this property.
                return None

            # TODO: would there be more than one box returned?
            assert len(query_results) == 1
            query_list = list(query_results.values())[0]
            assert len(query_list) == 1

            storage_key, data_object, data_directory = query_list[0]
            coordinate_file = os.path.join(
                data_directory, data_object.coordinate_file_name
            )
            assert os.path.exists(coordinate_file)

        return global_metadata
