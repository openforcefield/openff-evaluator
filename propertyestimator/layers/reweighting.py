"""This module implements a `CalculationLayer` which attempts to
'reweight' cached simulation data to evalulate the values of properties
at states which have not previously been simulated directly, but where
simulations at similar states have been run.
"""
import copy
import os

from propertyestimator.attributes import Attribute, PlaceholderValue
from propertyestimator.datasets import PropertyPhase
from propertyestimator.layers import calculation_layer
from propertyestimator.layers.workflow import (
    WorkflowCalculationLayer,
    WorkflowCalculationSchema,
)
from propertyestimator.storage.query import SimulationDataQuery


def default_storage_query():
    """Return the default query to use when retrieving cached simulation
     data from the storage backend.

    Currently this query will search for data for the full substance of
    interest in the liquid phase.

    Returns
    -------
    dict of str and SimulationDataQuery
        A single query with a key of `"full_system_data"`.
    """

    query = SimulationDataQuery()
    query.substance = PlaceholderValue()
    query.property_phase = PropertyPhase.Liquid

    return {"full_system_data": query}


class ReweightingSchema(WorkflowCalculationSchema):
    """A schema which encodes the options and the workflow schema
    that the `SimulationLayer` should use when estimating a given class
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

    def validate(self, attribute_type=None):
        super(ReweightingSchema, self).validate(attribute_type)

        assert len(self.storage_queries) > 0

        assert all(
            isinstance(x, SimulationDataQuery) for x in self.storage_queries.values()
        )


@calculation_layer()
class ReweightingLayer(WorkflowCalculationLayer):
    """A `CalculationLayer` which attempts to 'reweight' cached simulation
    data to evaluate the values of properties at states which have not previously
    been simulated directly, but where simulations at similar states have been run
    previously.
    """

    @classmethod
    def required_schema_type(cls):
        return ReweightingSchema

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

        Parameters
        ----------
        calculation_schema: ReweightingSchema
        """

        global_metadata = WorkflowCalculationLayer._get_workflow_metadata(
            working_directory,
            physical_property,
            force_field_path,
            parameter_gradient_keys,
            storage_backend,
            calculation_schema,
        )

        template_queries = calculation_schema.storage_queries

        # Apply the storage queries
        required_force_field_keys = set()

        for key in template_queries:

            query = copy.deepcopy(template_queries[key])

            # Fill in any place holder values.
            if isinstance(query.substance, PlaceholderValue):
                query.substance = physical_property.substance

            # Apply the query.
            query_results = storage_backend.query(query)

            if len(query_results) == 0:
                # We haven't found and cached data which is compatible
                # with this property.
                return None

            # Save a local copy of the data object file.
            stored_data_tuples = []

            for query_list in query_results.values():

                query_data_tuples = []

                for storage_key, data_object, data_directory in query_list:

                    object_path = os.path.join(working_directory, f"{storage_key}")
                    force_field_path = os.path.join(
                        working_directory, f"{data_object.force_field_id}"
                    )

                    # Save a local copy of the data object file.
                    if not os.path.isfile(object_path):
                        data_object.json(object_path)

                    required_force_field_keys.add(data_object.force_field_id)
                    query_data_tuples.append(
                        (object_path, data_directory, force_field_path)
                    )

                stored_data_tuples.append(query_data_tuples)

            # Add the results to the metadata.
            if len(stored_data_tuples) == 1:
                stored_data_tuples = stored_data_tuples[0]

            global_metadata[key] = stored_data_tuples

        # Make a local copy of the required force fields
        for force_field_id in required_force_field_keys:

            force_field_path = os.path.join(working_directory, force_field_id)

            if not os.path.isfile(force_field_path):

                existing_force_field = storage_backend.retrieve_force_field(
                    force_field_id
                )
                existing_force_field.json(force_field_path)

        return global_metadata
