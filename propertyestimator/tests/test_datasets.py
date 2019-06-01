"""
Units tests for propertyestimator.datasets
"""

import pytest
from simtk import unit

from propertyestimator.datasets.plugins import register_thermoml_property
from propertyestimator.datasets.thermoml import unit_from_thermoml_string
from propertyestimator.utils import get_data_filename

from propertyestimator.properties import PhysicalProperty, PropertyPhase
from propertyestimator.datasets import ThermoMLDataSet


@register_thermoml_property('Osmotic coefficient', supported_phases=PropertyPhase.Liquid)
class OsmoticCoefficient(PhysicalProperty):
    pass


@register_thermoml_property("Vapor or sublimation pressure, kPa",
                            supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas)
class VaporPressure(PhysicalProperty):
    pass


@register_thermoml_property('Activity coefficient', supported_phases=PropertyPhase.Liquid)
class ActivityCoefficient(PhysicalProperty):
    pass


supported_units = ['K', 'kPa', 'kg/m3', 'mol/kg', 'mol/dm3', 'kJ/mol', 'm3/kg', 'mol/m3',
                   'm3/mol', 'J/K/mol', 'J/K/kg', 'J/K/m3', '1/kPa', 'm/s', 'MHz']


@pytest.mark.skip(reason="Uncertainties have been unexpectedly removed from ThermoML "
                         "so these tests will fail until they have been re-added")
def test_from_url():

    data_set = ThermoMLDataSet.from_url('https://trc.nist.gov/journals/jct/2005v37/i04/j.jct.2004.09.022.xml')
    assert data_set is not None

    assert len(data_set.properties) > 0

    data_set = ThermoMLDataSet.from_url('https://trc.nist.gov/journals/jct/2005v37/i04/j.jct.2004.09.022.xmld')
    assert data_set is None


@pytest.mark.skip(reason="Uncertainties have been unexpectedly removed from ThermoML "
                         "so these tests will fail until they have been re-added")
def test_serialization():

    data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
    assert data_set is not None

    assert len(data_set.properties) > 0

    for mixture_tag in data_set.properties:

        for physical_property in data_set.properties[mixture_tag]:
            physical_property_json = physical_property.json()
            print(physical_property_json)

            physical_property_recreated = PhysicalProperty.parse_json(physical_property_json)
            print(physical_property_recreated)


@pytest.mark.skip(reason="Uncertainties have been unexpectedly removed from ThermoML "
                         "so these tests will fail until they have been re-added")
def test_from_doi():

    data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.10.001')
    assert data_set is not None

    assert len(data_set.properties) > 0

    for mixture_tag in data_set.properties:

        for physical_property in data_set.properties[mixture_tag]:

            physical_property_json = physical_property.json()
            print(physical_property_json)

            physical_property_recreated = PhysicalProperty.parse_json(physical_property_json)
            print(physical_property_recreated)

    data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.12.009')
    assert data_set is None

    data_set = ThermoMLDataSet.from_doi('10.1016/j.jct.2016.12.009x')
    assert data_set is None


def test_from_files():

    data_set = ThermoMLDataSet.from_file(get_data_filename('properties/j.jct.2004.09.014.xml'),
                                         get_data_filename('properties/j.jct.2004.09.022.xml'),
                                         get_data_filename('properties/j.jct.2007.09.004.xml'))
    assert data_set is not None

    assert len(data_set.properties) > 0

    data_set = ThermoMLDataSet.from_file('properties/j.jct.2004.09.014.xmld')
    assert data_set is None


@pytest.mark.parametrize("unit_string", supported_units)
def test_unit_from_string(unit_string):

    dummy_string = f'Property, {unit_string}'

    returned_unit = unit_from_thermoml_string(dummy_string)
    assert returned_unit is not None and isinstance(returned_unit, unit.Unit)


def test_mass_constraints():
    """A collection of tests to ensure that the Mass fraction constraint is
    implemented correctly alongside solvent constraints."""

    # Mass fraction
    data_set = ThermoMLDataSet.from_file(get_data_filename('test/properties/mass.xml'))

    assert data_set is not None
    assert len(data_set.properties) > 0

    # Mass fraction + Solvent: Mass fraction
    data_set = ThermoMLDataSet.from_file(get_data_filename('test/properties/mass_mass.xml'))

    assert data_set is not None
    assert len(data_set.properties) > 0

    # Mass fraction + Solvent: Mole fraction
    data_set = ThermoMLDataSet.from_file(get_data_filename('test/properties/mass_mole.xml'))

    assert data_set is not None
    assert len(data_set.properties) > 0


def test_molality_constraints():
    """A collection of tests to ensure that the Molality constraint is
    implemented correctly alongside solvent constraints."""

    # Molality
    data_set = ThermoMLDataSet.from_file(get_data_filename('test/properties/molality.xml'))

    assert data_set is not None
    assert len(data_set.properties) > 0

    # Molality + Solvent: Mass fraction
    data_set = ThermoMLDataSet.from_file(get_data_filename('test/properties/molality_mass.xml'))

    assert data_set is not None
    assert len(data_set.properties) > 0

    # Molality + Solvent: Mole fraction
    data_set = ThermoMLDataSet.from_file(get_data_filename('test/properties/molality_mole.xml'))

    assert data_set is not None
    assert len(data_set.properties) > 0

    # Molality + Solvent: Molality
    data_set = ThermoMLDataSet.from_file(get_data_filename('test/properties/molality_molality.xml'))

    assert data_set is not None
    assert len(data_set.properties) > 0


def test_mole_constraints():
    """A collection of tests to ensure that the Mole fraction constraint is
    implemented correctly alongside solvent constraints."""

    # Mole fraction
    data_set = ThermoMLDataSet.from_file(get_data_filename('test/properties/mole.xml'))

    assert data_set is not None
    assert len(data_set.properties) > 0

    # Mole fraction + Solvent: Mass fraction
    data_set = ThermoMLDataSet.from_file(get_data_filename('test/properties/mole_mass.xml'))

    assert data_set is not None
    assert len(data_set.properties) > 0

    # Mole fraction + Solvent: Mole fraction
    data_set = ThermoMLDataSet.from_file(get_data_filename('test/properties/mole_mole.xml'))

    assert data_set is not None
    assert len(data_set.properties) > 0

    # Mole fraction + Solvent: Molality
    data_set = ThermoMLDataSet.from_file(get_data_filename('test/properties/mole_molality.xml'))

    assert data_set is not None
    assert len(data_set.properties) > 0


