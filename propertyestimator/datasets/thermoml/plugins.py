import functools

from propertyestimator.datasets.thermoml import ThermoMLDataSet


class _ThermoMLPlugin:
    """Represents a property which may be extracted from a ThermoML archive.
    """

    def __init__(self, string_identifier, conversion_function, supported_phases):
        """Constructs a new ThermoMLPlugin object.

        Parameters
        ----------
        string_identifier: str
            The ThermoML string identifier (ePropName) for this property.
        conversion_function: function
            A function which maps a `ThermoMLProperty` into a
            `PhysicalProperty`.
        supported_phases: PropertyPhase:
            An enum which encodes all of the phases for which this
            property supports being estimated in.
        """

        self.string_identifier = string_identifier
        self.conversion_function = conversion_function

        self.supported_phases = supported_phases


def _default_mapping(property_class, property_to_map):
    """

    Parameters
    ----------
    property_class: type of PhysicalProperty
        The class to map this property into.
    property_to_map: ThermoMLProperty
        The ThermoML property to map.
    """

    mapped_property = property_class()

    mapped_property.value = property_to_map.value
    mapped_property.uncertainty = property_to_map.uncertainty

    mapped_property.phase = property_to_map.phase

    mapped_property.thermodynamic_state = property_to_map.thermodynamic_state
    mapped_property.substance = property_to_map.substance

    mapped_property.source = property_to_map
    return mapped_property


def register_thermoml_property(
    thermoml_string, supported_phases, property_class=None, conversion_function=None
):
    """A function used to map a property from the ThermoML archive
    to an internal `PhysicalProperty` object of the correct type.

    This function takes either a specific class (e.g. `Density`)
    which maps directly to the specified `thermoml_string`, or a
    a function which maps a `ThermoMLProperty` into a `PhysicalProperty`
    allowing fuller control.

    Parameters
    ----------
    thermoml_string: str
        The ThermoML string identifier (ePropName) for this property.
    supported_phases: PropertyPhase:
        An enum which encodes all of the phases for which this property
        supports being estimated in.
    property_class: type of PhysicalProperty, optional
        The class associated with this physical property. This argument
        is mutually exclusive with the `conversion_function` argument.
    conversion_function: function
        A function which maps a `ThermoMLProperty` into a `PhysicalProperty`.
        This argument is mutually exclusive with the `property_class` argument.
    """

    if (property_class is None and conversion_function is None) or (
        property_class is not None and conversion_function is not None
    ):
        raise ValueError(
            "Only one of the `property_class` and `conversion_function` must be set."
        )

    if conversion_function is None:
        conversion_function = functools.partial(_default_mapping, property_class)

    ThermoMLDataSet.registered_properties[thermoml_string] = _ThermoMLPlugin(
        thermoml_string, conversion_function, supported_phases
    )


def thermoml_property(thermoml_string, supported_phases):
    """A decorator which wraps around the `register_thermoml_property`
    method.

    Parameters
    ----------
    thermoml_string: str
        The ThermoML string identifier (ePropName) for this property.
    supported_phases: PropertyPhase:
        An enum which encodes all of the phases for which this
        property supports being estimated in.
    """

    def decorator(cls):
        register_thermoml_property(thermoml_string, supported_phases, cls)
        return cls

    return decorator
