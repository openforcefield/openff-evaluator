"""This module implements a `CalculationLayer` which attempts to
'reweight' cached simulation data to evalulate the values of properties
at states which have not previously been simulated directly, but where
simulations at similar states have been run.
"""
from propertyestimator.attributes import UNDEFINED, Attribute
from propertyestimator.layers import calculation_layer
from propertyestimator.layers.workflow import (
    WorkflowCalculationLayer,
    WorkflowCalculationSchema,
)
from propertyestimator.storage.query import SimulationDataQuery


class ReweightingSchema(WorkflowCalculationSchema):
    """A schema which encodes the options and the workflow schema
    that the `SimulationLayer` should use when estimating a given class
    of physical properties using the built-in workflow framework.
    """

    storage_query = Attribute(
        docstring="The query to use when retrieving cached simulation data"
        "from the storage backend.",
        type_hint=SimulationDataQuery,
        default_value=UNDEFINED,
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
