"""
Defines an API for defining thermodynamic states.
"""
import math
from enum import Enum

from propertyestimator import unit
from propertyestimator.utils.serialization import TypedBaseModel


class Ensemble(Enum):
    """An enum describing the available thermodynamic ensembles.
    """
    NVT = "NVT"
    NPT = "NPT"


class ThermodynamicState(TypedBaseModel):
    """
    Data specifying a physical thermodynamic state obeying Boltzmann statistics.

    Attributes
    ----------
    temperature : propertyestimator.unit.Quantity with units compatible with kelvin
        The external temperature
    pressure : propertyestimator.unit.Quantity with units compatible with atmospheres
        The external pressure

    Examples
    --------
    Specify an NPT state at 298 K and 1 atm pressure.

    >>> state = ThermodynamicState(temperature=298.0*unit.kelvin, pressure=1.0*unit.atmospheres)

    Note that the pressure is only relevant for periodic systems.

    """

    @property
    def molar_temperature(self):
        """Returns the temperature multiplied by the molar gas constant"""
        return (self.temperature * unit.molar_gas_constant).to(unit.kilojoule / unit.mole)

    @property
    def beta(self):
        """Returns one divided by the temperature multiplied by the molar gas constant"""
        return 1.0 / self.molar_temperature

    def __init__(self, temperature=None, pressure=None):
        """Constructs a new ThermodynamicState object.

        Parameters
        ----------
        temperature : propertyestimator.unit.Quantity with units compatible with kelvin
            The external temperature
        pressure : propertyestimator.unit.Quantity with units compatible with atmospheres
            The external pressure
        """

        self.temperature = temperature
        self.pressure = pressure

    def __getstate__(self):

        return {
            'temperature': self.temperature,
            'pressure': self.pressure,
        }

    def __setstate__(self, state):

        self.temperature = state['temperature']
        self.pressure = state['pressure']

    def __repr__(self):
        """
        Returns a string representation of a state.
        """
        return_value = "ThermodynamicState("

        if self.temperature is not None:
            return_value += "temperature={0:s}, ".format(repr(self.temperature))
        if self.pressure is not None:
            return_value += "pressure = {0:s}".format(repr(self.pressure))

        return_value += ")"

        return return_value

    def __str__(self):

        return_value = "<ThermodynamicState object"

        if self.temperature is not None:
            return_value += ", temperature = {0:s}".format(str(self.temperature))
        if self.pressure is not None:
            return_value += ", pressure = {0:s}".format(str(self.pressure))

        return_value += ">"

        return return_value

    def __eq__(self, other):

        if not isinstance(other, ThermodynamicState):
            return False

        return (math.isclose(self.temperature.to(unit.kelvin).magnitude,
                             other.temperature.to(unit.kelvin).magnitude) and
                math.isclose(self.pressure.to(unit.atmosphere).magnitude,
                             other.pressure.to(unit.atmosphere).magnitude))

    def __ne__(self, other):
        return not (self == other)
