"""
Units tests for evaluator.datasets
"""
import pint
import pytest

from evaluator import unit
from evaluator.datasets import PhysicalProperty, PropertyPhase
from evaluator.datasets.thermoml.plugins import thermoml_property
from evaluator.datasets.thermoml.thermoml import (
    ThermoMLDataSet,
    _unit_from_thermoml_string,
)
from evaluator.plugins import register_default_plugins
from evaluator.utils import get_data_filename

register_default_plugins()


@thermoml_property("Osmotic coefficient", supported_phases=PropertyPhase.Liquid)
class OsmoticCoefficient(PhysicalProperty):

    def default_unit(cls):
        return unit.dimensionless


@thermoml_property(
    "Vapor or sublimation pressure, kPa",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class VaporPressure(PhysicalProperty):

    def default_unit(cls):
        return unit.kilopascal


@thermoml_property("Activity coefficient", supported_phases=PropertyPhase.Liquid)
class ActivityCoefficient(PhysicalProperty):

    def default_unit(cls):
        return unit.dimensionless


supported_units = [
    "K",
    "kPa",
    "kg/m3",
    "mol/kg",
    "mol/dm3",
    "kJ/mol",
    "m3/kg",
    "mol/m3",
    "m3/mol",
    "J/K/mol",
    "J/K/kg",
    "J/K/m3",
    "1/kPa",
    "m/s",
    "MHz",
]


@pytest.mark.parametrize("unit_string", supported_units)
def test_thermoml_unit_from_string(unit_string):
    """A test to ensure all unit conversions are valid."""

    dummy_string = f"Property, {unit_string}"

    returned_unit = _unit_from_thermoml_string(dummy_string)
    assert returned_unit is not None and isinstance(returned_unit, pint.Unit)


def test_thermoml_from_url():
    """A test to ensure that ThermoML archive files can be loaded from a url."""

    data_set = ThermoMLDataSet.from_url(
        "https://trc.nist.gov/ThermoML/10.1021/acs.jced.6b00916.xml"
    )
    assert data_set is not None

    assert len(data_set) > 0

    data_set = ThermoMLDataSet.from_url(
        "https://trc.nist.gov/ThermoML/10.1021/acs.jced.6b00916.xmld"
    )
    assert data_set is None


def test_thermoml_from_doi():
    """A test to ensure that ThermoML archive files can be loaded from a doi."""

    data_set = ThermoMLDataSet.from_doi("10.1016/j.jct.2016.10.001")

    assert data_set is not None
    assert len(data_set) > 0

    data_set = ThermoMLDataSet.from_doi("10.1016/j.jct.2016.12.009x")
    assert data_set is None


def test_thermoml_from_files():
    """A test to ensure that ThermoML archive files can be loaded from local sources."""

    data_set = ThermoMLDataSet.from_file(
        get_data_filename("properties/single_density.xml"),
        get_data_filename("properties/single_dielectric.xml"),
        get_data_filename("properties/single_enthalpy_mixing.xml"),
    )

    assert data_set is not None
    assert len(data_set) == 3

    data_set = ThermoMLDataSet.from_file("dummy_filename")
    assert data_set is None


def test_thermoml_mass_constraints():
    """A collection of tests to ensure that the Mass fraction constraint is
    implemented correctly alongside solvent constraints."""

    # Mass fraction
    data_set = ThermoMLDataSet.from_file(get_data_filename("test/properties/mass.xml"))

    assert data_set is not None
    assert len(data_set) > 0

    # Mass fraction + Solvent: Mass fraction
    data_set = ThermoMLDataSet.from_file(
        get_data_filename("test/properties/mass_mass.xml")
    )

    assert data_set is not None
    assert len(data_set) > 0

    # Mass fraction + Solvent: Mole fraction
    data_set = ThermoMLDataSet.from_file(
        get_data_filename("test/properties/mass_mole.xml")
    )

    assert data_set is not None
    assert len(data_set) > 0


def test_thermoml_molality_constraints():
    """A collection of tests to ensure that the Molality constraint is
    implemented correctly alongside solvent constraints."""

    # Molality
    data_set = ThermoMLDataSet.from_file(
        get_data_filename("test/properties/molality.xml")
    )

    assert data_set is not None
    assert len(data_set) > 0

    # Molality + Solvent: Mass fraction
    data_set = ThermoMLDataSet.from_file(
        get_data_filename("test/properties/molality_mass.xml")
    )

    assert data_set is not None
    assert len(data_set) > 0

    # Molality + Solvent: Mole fraction
    data_set = ThermoMLDataSet.from_file(
        get_data_filename("test/properties/molality_mole.xml")
    )

    assert data_set is not None
    assert len(data_set) > 0

    # Molality + Solvent: Molality
    data_set = ThermoMLDataSet.from_file(
        get_data_filename("test/properties/molality_molality.xml")
    )

    assert data_set is not None
    assert len(data_set) > 0


def test_thermoml_mole_constraints():
    """A collection of tests to ensure that the Mole fraction constraint is
    implemented correctly alongside solvent constraints."""

    # Mole fraction
    data_set = ThermoMLDataSet.from_file(get_data_filename("test/properties/mole.xml"))

    assert data_set is not None
    assert len(data_set) > 0

    # Mole fraction + Solvent: Mass fraction
    data_set = ThermoMLDataSet.from_file(
        get_data_filename("test/properties/mole_mass.xml")
    )

    assert data_set is not None
    assert len(data_set) > 0

    # Mole fraction + Solvent: Mole fraction
    data_set = ThermoMLDataSet.from_file(
        get_data_filename("test/properties/mole_mole.xml")
    )

    assert data_set is not None
    assert len(data_set) > 0

    # Mole fraction + Solvent: Molality
    data_set = ThermoMLDataSet.from_file(
        get_data_filename("test/properties/mole_molality.xml")
    )

    assert data_set is not None
    assert len(data_set) > 0
