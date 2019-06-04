"""
An API for allowing properties to register themselves as being extractable
from a given data set.
"""

registered_thermoml_properties = {}


class ThermoMLPlugin:
    """Represents a property which may be extracted from a ThermoML archive.
    """

    def __init__(self, class_type, string_identifier, supported_phases):
        """Constructs a new ThermoMLPlugin object.

        Parameters
        ----------
        class_type: type
            The type of property which this plugin represents.
        string_identifier: str
            The ThermoML string identifier (ePropName) for this property.
        supported_phases: PropertyPhase:
            An enum which encodes all of the phases for which this
            property supports being estimated in.
        """

        self.class_type = class_type
        self.string_identifier = string_identifier

        self.supported_phases = supported_phases


def register_thermoml_property(thermoml_string, supported_phases):
    """A decorator which registers information on how to parse a given
    ThermoML property

    For now this only takes input of a thermoML string, but in future
    will give greater control over exactly how ThermoML XML gets parsed
    to an actual property.

    Parameters
    ----------
    thermoml_string: str
        The ThermoML string identifier (ePropName) for this property.
    supported_phases: PropertyPhase:
        An enum which encodes all of the phases for which this
        property supports being estimated in.
    """

    def decorator(cls):
        registered_thermoml_properties[thermoml_string] = ThermoMLPlugin(cls,
                                                                         thermoml_string,
                                                                         supported_phases)

        return cls

    return decorator
