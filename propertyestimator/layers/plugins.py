"""
An API for registering new calculation layers.

Attributes
----------
registered_calculation_layers: dict of str and type of CalculationLayer
    The calculation layers which have been registered as being
    available to use in property estimations.
registered_calculation_schemas: dict of str and dict of str and type of CalculationLayerSchema
    The default calculation schemas to use when estimating a class of properties (e.g. `Density`)
    with a specific calculation layer (e.g. `SimulationLayer`).

    The dictionary is of the form `registered_calculation_schemas['LayerType']['PropertyType']`
"""
from collections import defaultdict
from typing import Dict, Type

from propertyestimator.datasets import PhysicalProperty
from propertyestimator.layers import CalculationLayer
from propertyestimator.layers.layers import CalculationLayerSchema

registered_calculation_layers: Dict[str, Type[CalculationLayer]] = {}
registered_calculation_schemas: Dict[
    str, Dict[str, CalculationLayerSchema]
] = defaultdict(dict)


def register_calculation_layer(layer_class):
    """Registers a class as being a calculation layer
    which may be used in property calculations.

    Parameters
    ----------
    layer_class: type of CalculationLayer
        The calculation layer to register.
    """
    assert issubclass(layer_class, CalculationLayer)
    assert issubclass(layer_class.required_schema_type(), CalculationLayerSchema)

    if layer_class.__name__ in registered_calculation_layers:
        raise ValueError(f"The {layer_class} layer is already registered.")

    registered_calculation_layers[layer_class.__name__] = layer_class


def register_calculation_schema(property_class, layer_class, schema):
    """Registers the default calculation schema to use when estimating a
    class of properties (e.g. `Density`) with a specific calculation layer
    (e.g. the `SimulationLayer`).

    Parameters
    ----------
    property_class: type of PhysicalProperty
        The class of properties to associate with the
        specified `calculation_layer` and `property_class`.
    layer_class: type of CalculationLayer
        The calculation layer to associate the schema with.
    schema: CalculationLayerSchema or Callable[[CalculationLayerSchema], CalculationLayerSchema]
        Either the calculation schema to use, or a function which
        will create the schema from an existing CalculationLayerSchema.
    """

    assert issubclass(property_class, PhysicalProperty)
    assert issubclass(layer_class, CalculationLayer)
    assert isinstance(schema, CalculationLayerSchema) or callable(schema)

    assert property_class != PhysicalProperty
    assert layer_class != CalculationLayer

    registered_calculation_schemas[layer_class.__name__][
        property_class.__name__
    ] = schema


def calculation_layer():
    """A decorator which registers a class as being a calculation layer
    which may be used in property calculations.
    """

    def decorator(cls):
        register_calculation_layer(cls)
        return cls

    return decorator
