"""This module implements a `CalculationLayer` which attempts to
'reweight' cached simulation data to evalulate the values of properties
at states which have not previously been simulated directly, but where
simulations at similar states have been run.
"""
from propertyestimator.attributes import Attribute
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

        for query in self.storage_queries.values():
            query.validate()


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
        physical_property,
        force_field_path,
        parameter_gradient_keys,
        storage_backend,
        calculation_schema,
    ):

        global_metadata = WorkflowCalculationLayer._get_workflow_metadata(
            physical_property,
            force_field_path,
            parameter_gradient_keys,
            storage_backend,
            calculation_schema,
        )

        # TODO: Implement this using the soon to be built storage queries.
        # substance_id = physical_property.substance.identifier
        # data_class_type = physical_property.required_data_class
        #
        # if (
        #     substance_id not in stored_data_paths
        #     or data_class_type not in stored_data_paths[substance_id]
        # ):
        #     # We haven't found and cached data which is compatible with this property.
        #     continue
        #
        # global_metadata["full_system_data"] = stored_data_paths[substance_id][
        #     data_class_type
        # ]
        # global_metadata["component_data"] = []
        #
        # if property_to_calculate.multi_component_property:
        #
        #     has_data_for_property = True
        #
        #     for component in property_to_calculate.substance.components:
        #
        #         temporary_substance = Substance()
        #         temporary_substance.add_component(
        #             component, amount=Substance.MoleFraction()
        #         )
        #
        #         if (
        #             temporary_substance.identifier not in stored_data_paths
        #             or data_class_type
        #             not in stored_data_paths[temporary_substance.identifier]
        #         ):
        #             has_data_for_property = False
        #             break
        #
        #         global_metadata["component_data"].append(
        #             stored_data_paths[temporary_substance.identifier][data_class_type]
        #         )
        #
        #     if not has_data_for_property:
        #         continue

        return global_metadata
