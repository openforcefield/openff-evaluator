"""
An API for allowing new properties to be estimated.
"""
from collections import defaultdict

from propertyestimator.datasets import PhysicalProperty
from propertyestimator.layers import PropertyCalculationLayer

available_layers = {}
default_workflow_schemas = defaultdict(dict)


def register_calculation_layer(layer_class):
    """Registers a class as being a calculation layer
    which may be used in property calculations.
    """
    assert isinstance(layer_class, PropertyCalculationLayer)

    if layer_class.__name__ in available_layers:
        raise ValueError(f"The {layer_class} layer is already registered.")

    available_layers[layer_class.__name__] = layer_class


def calculation_layer():
    """A decorator which registers a class as being a calculation layer
    which may be used in property calculations.
    """

    def decorator(cls):
        register_calculation_layer(cls)
        return cls

    return decorator


def register_default_schema(property_class, layer_class, default_schema=None):
    """Registers a default workflow schema to use when estimating
    a class of properties (e.g. `Density`) using  specified calculation
    layer (e.g. `SimulationLayer`).

    Parameters
    ----------
    property_class: type of PhysicalProperty
        The class of properties to associate with the
        specified `calculation_layer` and `property_class`.
    layer_class: type of PropertyCalculationLayer
        The calculation layer to associate the schema with.
    default_schema: function
        A function which returns the workflow schema for a
        given set of workflow options.
    """

    assert issubclass(property_class, PhysicalProperty)
    assert issubclass(layer_class, PropertyCalculationLayer)

    default_workflow_schemas[layer_class.__name__][
        property_class.__name__
    ] = default_schema
