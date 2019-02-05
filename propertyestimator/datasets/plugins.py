"""
An API for allowing properties to register themselves as being extractable
from a given data set.
"""


registered_thermoml_properties = {}


def register_thermoml_property(thermoml_string):
    """A decorator which registers information on how to parse a given
    ThermoML property

    For now this only takes input of a thermoML string, but in future
    will give greater control over exactly how ThermoML XML gets parsed
    to an actual property."""

    def decorator(cls):
        registered_thermoml_properties[thermoml_string] = cls
        return cls

    return decorator
