from openmm.unit import Quantity as OpenMMQuantity
from pint import Measurement as _Measurement
from pint import Quantity as _Quantity
from pint import Unit as _Unit
from pint import UnitRegistry as _UnitRegistry

class Unit(_Unit): ...

class Quantity(_Quantity):
    def __dask_tokenize__(self): ...
    def to_openmm(self) -> OpenMMQuantity: ...

class Measurement(_Measurement):
    def __dask_tokenize__(self): ...

class UnitRegistry(_UnitRegistry): ...

DEFAULT_UNIT_REGISTRY: UnitRegistry
