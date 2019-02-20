"""
The surrogate modelling estimation layer.
"""

from propertyestimator.layers import register_calculation_layer, PropertyCalculationLayer
from propertyestimator.utils.serialization import serialize_force_field


@register_calculation_layer()
class SurrogateLayer(PropertyCalculationLayer):
    """A calculation layer which aims to calculate physical properties from
    a surrogate model, such as a Gaussian mixture model.

    .. warning :: This class has not yet been implemented.
    """

    @staticmethod
    def schedule_calculation(calculation_backend, storage_backend, layer_directory,
                             data_model, callback, synchronous=False):

        force_field = storage_backend.retrieve_force_field(data_model.force_field_id)

        surrogate_futures = []

        for physical_property in data_model.queued_properties:

            surrogate_future = calculation_backend.submit_task(SurrogateLayer.perform_surrogate_extrapolation,
                                                               physical_property,
                                                               serialize_force_field(force_field))

            surrogate_futures.append(surrogate_future)

        PropertyCalculationLayer._await_results(calculation_backend,
                                                storage_backend,
                                                layer_directory,
                                                data_model,
                                                callback,
                                                surrogate_futures,
                                                synchronous)

    @staticmethod
    def perform_surrogate_extrapolation(physical_property, force_field_dict, **kwargs):
        """A placeholder method that would be used to spawn the surrogate
        model backend.

        .. warning :: This method has not yet been implemented.
        """

        # A return value indicates that the surrogate layer did not
        # have access to enough information to accurately estimate the property.
        return None
