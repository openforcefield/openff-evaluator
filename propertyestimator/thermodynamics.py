"""
Defines an API for defining thermodynamic states.
"""
import math
from enum import Enum
from typing import Optional

from pydantic import BaseModel, validator
from simtk import unit

from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.serialization import deserialize_quantity, serialize_quantity


class Ensemble(Enum):
    """An enum describing the available thermodynamic ensembles.
    """
    NVT = "NVT"
    NPT = "NPT"


class ThermodynamicState(BaseModel):
    """
    Data specifying a physical thermodynamic state obeying Boltzmann statistics.

    Attributes
    ----------
    temperature : simtk.unit.Quantity with units compatible with kelvin
        The external temperature
    pressure : simtk.unit.Quantity with units compatible with atmospheres
        The external pressure

    Examples
    --------
    Specify an NPT state at 298 K and 1 atm pressure.

    >>> state = ThermodynamicState(temperature=298.0*unit.kelvin, pressure=1.0*unit.atmospheres)

    Note that the pressure is only relevant for periodic systems.

    """

    temperature: Optional[unit.Quantity] = None
    pressure: Optional[unit.Quantity] = None

    class Config:

        arbitrary_types_allowed = True

        json_encoders = {
            EstimatedQuantity: lambda value: value.__getstate__(),
            unit.Quantity: lambda v: serialize_quantity(v),
        }

    @validator('temperature', pre=True, whole=True)
    def validate_temperature(cls, v):

        if isinstance(v, dict):
            v = deserialize_quantity(v)

        if isinstance(v, unit.Quantity):
            v = v.in_units_of(unit.kelvin)

        return v
    
    @validator('pressure', pre=True, whole=True)
    def validate_pressure(cls, v):

        if isinstance(v, dict):
            v = deserialize_quantity(v)

        if isinstance(v, unit.Quantity):
            v = v.in_units_of(unit.atmospheres)

        return v

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

        return (math.isclose(self.temperature / unit.kelvin, other.temperature / unit.kelvin) and
                math.isclose(self.pressure / unit.atmosphere, other.pressure / unit.atmosphere))

    def __ne__(self, other):
        return not (self == other)
