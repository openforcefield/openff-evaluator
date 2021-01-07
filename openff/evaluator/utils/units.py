"""A module to strip ``pint`` of its dynamic classes."""
import os
import uuid
import warnings

import pint
from pint.measurement import _Measurement
from pint.quantity import _Quantity
from pint.unit import _Unit

from openff.evaluator.utils import get_data_filename

DEFAULT_UNIT_REGISTRY = pint.UnitRegistry(
    get_data_filename(os.path.join("units", "defaults.txt"))
)


def _unpickle_quantity(cls, *args):
    """Rebuild quantity upon unpickling using the application registry."""
    return pint._unpickle(DEFAULT_UNIT_REGISTRY.Quantity, *args)


def _unpickle_unit(cls, *args):
    """Rebuild unit upon unpickling using the application registry."""
    return pint._unpickle(DEFAULT_UNIT_REGISTRY.Unit, *args)


def _unpickle_measurement(cls, *args):
    """Rebuild measurement upon unpickling using the application registry."""
    return pint._unpickle(DEFAULT_UNIT_REGISTRY.Measurement, *args)


class Unit(_Unit):
    _REGISTRY = DEFAULT_UNIT_REGISTRY

    def __reduce__(self):
        return _unpickle_unit, (Unit, self._units)


class Quantity(_Quantity):
    _REGISTRY = DEFAULT_UNIT_REGISTRY

    def __reduce__(self):
        return _unpickle_quantity, (Quantity, self.magnitude, self._units)

    def __dask_tokenize__(self):
        return uuid.uuid4().hex

    @staticmethod
    def _dask_finalize(results, func, args, units):
        values = func(results, *args)
        return Quantity(values, units)


class Measurement(_Measurement):
    _REGISTRY = DEFAULT_UNIT_REGISTRY

    def __reduce__(self):
        return _unpickle_measurement, (Measurement, self.magnitude, self._units)

    def __dask_tokenize__(self):
        return uuid.uuid4().hex

    @staticmethod
    def _dask_finalize(results, func, args, units):
        values = func(results, *args)
        return Measurement(values, units)


DEFAULT_UNIT_REGISTRY.Unit = Unit
DEFAULT_UNIT_REGISTRY.Quantity = Quantity
DEFAULT_UNIT_REGISTRY.Measurement = Measurement

pint.set_application_registry(DEFAULT_UNIT_REGISTRY)
DEFAULT_UNIT_REGISTRY.default_format = "~"

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Quantity([])
