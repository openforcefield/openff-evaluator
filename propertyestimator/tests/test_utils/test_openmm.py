import inspect
from random import randint, random

import numpy as np
import pytest
from simtk import unit as simtk_unit

from propertyestimator import unit
from propertyestimator.utils.openmm import (
    openmm_quantity_to_pint,
    openmm_unit_to_pint,
    pint_quantity_to_openmm,
    pint_unit_to_openmm,
    unsupported_openmm_units,
)


def _get_all_openmm_units():
    """Returns a list of all of the defined OpenMM units.

    Returns
    -------
    list of simtk.unit.Unit
    """

    all_openmm_units = set(
        [
            unit_type
            for _, unit_type in inspect.getmembers(simtk_unit)
            if isinstance(unit_type, simtk_unit.Unit)
        ]
    )

    all_openmm_units = all_openmm_units.difference(unsupported_openmm_units)

    return all_openmm_units


def _get_all_pint_units():
    """Returns a list of all of the pint units which
    could be converted from all OpenMM units.

    Returns
    -------
    list of pint.Unit
    """

    all_openmm_units = _get_all_openmm_units()

    all_pint_units = [
        openmm_unit_to_pint(openmm_unit) for openmm_unit in all_openmm_units
    ]

    return all_pint_units


def test_daltons():

    openmm_quantity = random() * simtk_unit.dalton
    openmm_raw_value = openmm_quantity.value_in_unit(simtk_unit.gram / simtk_unit.mole)

    pint_quantity = openmm_quantity_to_pint(openmm_quantity)
    pint_raw_value = pint_quantity.to(unit.gram / unit.mole).magnitude

    assert np.allclose(openmm_raw_value, pint_raw_value)


@pytest.mark.parametrize("openmm_unit", _get_all_openmm_units())
@pytest.mark.parametrize(
    "value",
    [random(), randint(1, 10), [random(), random()], np.array([random(), random()])],
)
def test_openmm_unit_to_pint(openmm_unit, value):

    openmm_quantity = value * openmm_unit
    openmm_raw_value = openmm_quantity.value_in_unit(openmm_unit)

    pint_quantity = openmm_quantity_to_pint(openmm_quantity)
    pint_raw_value = pint_quantity.magnitude

    assert np.allclose(openmm_raw_value, pint_raw_value)


@pytest.mark.parametrize("pint_unit", _get_all_pint_units())
@pytest.mark.parametrize(
    "value",
    [random(), randint(1, 10), [random(), random()], np.array([random(), random()])],
)
def test_pint_to_openmm(pint_unit, value):

    pint_quantity = value * pint_unit
    pint_raw_value = pint_quantity.magnitude

    openmm_quantity = pint_quantity_to_openmm(pint_quantity)
    openmm_raw_value = openmm_quantity.value_in_unit(openmm_quantity.unit)

    assert np.allclose(openmm_raw_value, pint_raw_value)


def test_combinatorial_pint_to_openmm():

    all_pint_units = _get_all_pint_units()

    for i in range(len(all_pint_units)):

        for j in range(i, len(all_pint_units)):

            pint_unit = all_pint_units[i] * all_pint_units[j]

            pint_quantity = random() * pint_unit
            pint_raw_value = pint_quantity.magnitude

            openmm_quantity = pint_quantity_to_openmm(pint_quantity)
            openmm_raw_value = openmm_quantity.value_in_unit(openmm_quantity.unit)

            assert np.isclose(openmm_raw_value, pint_raw_value)


def test_combinatorial_openmm_to_pint():

    all_openmm_units = list(_get_all_openmm_units())

    for i in range(len(all_openmm_units)):

        for j in range(i, len(all_openmm_units)):

            openmm_unit = all_openmm_units[i] * all_openmm_units[j]

            openmm_quantity = random() * openmm_unit
            openmm_raw_value = openmm_quantity.value_in_unit(openmm_unit)

            pint_quantity = openmm_quantity_to_pint(openmm_quantity)
            pint_raw_value = pint_quantity.magnitude

            assert np.isclose(openmm_raw_value, pint_raw_value)


@pytest.mark.parametrize(
    "openmm_unit",
    {*_get_all_openmm_units(), simtk_unit.dalton ** 2, simtk_unit.dalton ** 3},
)
@pytest.mark.parametrize(
    "value",
    [random(), randint(1, 10), [random(), random()], np.array([random(), random()])],
)
def test_openmm_unit_conversions(openmm_unit, value):

    openmm_quantity = value * openmm_unit

    openmm_base_quantity = openmm_quantity.in_unit_system(simtk_unit.md_unit_system)

    if not isinstance(openmm_base_quantity, simtk_unit.Quantity):
        openmm_base_quantity *= simtk_unit.dimensionless

    pint_base_quantity = openmm_quantity_to_pint(openmm_base_quantity)

    pint_unit = openmm_unit_to_pint(openmm_unit)
    pint_quantity = pint_base_quantity.to(pint_unit)

    pint_raw_value = pint_quantity.magnitude
    openmm_raw_value = openmm_quantity.value_in_unit(openmm_unit)

    assert np.allclose(openmm_raw_value, pint_raw_value)


@pytest.mark.parametrize("pint_unit", _get_all_pint_units())
@pytest.mark.parametrize(
    "value",
    [random(), randint(1, 10), [random(), random()], np.array([random(), random()])],
)
def test_pint_unit_conversions(pint_unit, value):

    pint_quantity = value * pint_unit

    pint_base_quantity = pint_quantity.to_base_units()
    openmm_base_quantity = pint_quantity_to_openmm(pint_base_quantity)

    openmm_unit = pint_unit_to_openmm(pint_unit)
    openmm_quantity = openmm_base_quantity.in_units_of(openmm_unit)

    pint_raw_value = pint_quantity.magnitude

    if pint_unit == unit.dimensionless and (
        isinstance(openmm_quantity, float)
        or isinstance(openmm_quantity, int)
        or isinstance(openmm_quantity, list)
        or isinstance(openmm_quantity, np.ndarray)
    ):
        openmm_raw_value = openmm_quantity
    else:
        openmm_raw_value = openmm_quantity.value_in_unit(openmm_unit)

    assert np.allclose(openmm_raw_value, pint_raw_value)


def test_constants():

    assert np.isclose(
        simtk_unit.AVOGADRO_CONSTANT_NA.value_in_unit((1.0 / simtk_unit.mole).unit),
        (1.0 * unit.avogadro_constant).to((1.0 / unit.mole).units).magnitude,
    )

    assert np.isclose(
        simtk_unit.BOLTZMANN_CONSTANT_kB.value_in_unit(
            simtk_unit.joule / simtk_unit.kelvin
        ),
        (1.0 * unit.boltzmann_constant).to(unit.joule / unit.kelvin).magnitude,
    )

    assert np.isclose(
        simtk_unit.MOLAR_GAS_CONSTANT_R.value_in_unit(
            simtk_unit.joule / simtk_unit.kelvin / simtk_unit.mole
        ),
        (1.0 * unit.molar_gas_constant)
        .to(unit.joule / unit.kelvin / unit.mole)
        .magnitude,
    )

    assert np.isclose(
        simtk_unit.GRAVITATIONAL_CONSTANT_G.value_in_unit(
            simtk_unit.meter ** 2 * simtk_unit.newton / simtk_unit.kilogram ** 2
        ),
        (1.0 * unit.newtonian_constant_of_gravitation)
        .to(unit.meter ** 2 * unit.newton / unit.kilogram ** 2)
        .magnitude,
    )

    assert np.isclose(
        simtk_unit.SPEED_OF_LIGHT_C.value_in_unit(
            simtk_unit.meter / simtk_unit.seconds
        ),
        (1.0 * unit.speed_of_light).to(unit.meter / unit.seconds).magnitude,
    )
