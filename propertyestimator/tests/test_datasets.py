"""
Units tests for propertyestimator.datasets
"""

import pytest

from propertyestimator import unit
from propertyestimator.datasets import ThermoMLDataSet, PhysicalPropertyDataSet
from propertyestimator.datasets.plugins import register_thermoml_property
from propertyestimator.datasets.thermoml import unit_from_thermoml_string
from propertyestimator.properties import PhysicalProperty, PropertyPhase, Density
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import create_filterable_data_set, create_dummy_property
from propertyestimator.utils import get_data_filename


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


@pytest.mark.parametrize("unit_string", supported_units)
def test_thermoml_unit_from_string(unit_string):
    """A test to ensure all unit conversions are valid."""

    dummy_string = f'Property, {unit_string}'

    returned_unit = unit_from_thermoml_string(dummy_string)
    assert returned_unit is not None and isinstance(returned_unit, unit.Unit)


@pytest.mark.skip(reason="Uncertainties have been unexpectedly removed from ThermoML "
                         "so these tests will fail until they have been re-added")
def test_thermoml_from_url():
    """A test to ensure that ThermoML archive files can be loaded from a url."""

    data_set = ThermoMLDataSet.from_url('https://trc.nist.gov/journals/jct/2005v37/i04/j.jct.2004.09.022.xml')
    assert data_set is not None

    assert len(data_set.properties) > 0

    data_set = ThermoMLDataSet.from_url('https://trc.nist.gov/journals/jct/2005v37/i04/j.jct.2004.09.022.xmld')
    assert data_set is None


@pytest.mark.skip(reason="Uncertainties have been unexpectedly removed from ThermoML "
                         "so these tests will fail until they have been re-added")
def test_thermoml_from_doi():
    """A test to ensure that ThermoML archive files can be loaded from a doi."""

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


def test_thermoml_from_files():
    """A test to ensure that ThermoML archive files can be loaded from local sources."""

    data_set = ThermoMLDataSet.from_file(get_data_filename('properties/single_density.xml'),
                                         get_data_filename('properties/single_dielectric.xml'),
                                         get_data_filename('properties/single_enthalpy_mixing.xml'))

    assert data_set is not None
    assert len(data_set.properties) == 3

    data_set = ThermoMLDataSet.from_file('dummy_filename')
    assert data_set is None


def test_thermoml_mass_constraints():
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


def test_thermoml_molality_constraints():
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


def test_thermoml_mole_constraints():
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


def test_serialization():
    """A test to ensure that data sets are JSON serializable."""

    data_set = ThermoMLDataSet.from_file(get_data_filename('properties/single_density.xml'))
    data_set_json = data_set.json()

    parsed_data_set = ThermoMLDataSet.parse_json(data_set_json)
    assert data_set.number_of_properties == parsed_data_set.number_of_properties

    parsed_data_set_json = parsed_data_set.json()
    assert parsed_data_set_json == data_set_json


def test_filter_by_property_types():
    """A test to ensure that data sets may be filtered by property type."""

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_property_types('Density')

    assert dummy_data_set.number_of_properties == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_property_types('Density', 'DielectricConstant')

    assert dummy_data_set.number_of_properties == 2


def test_filter_by_phases():
    """A test to ensure that data sets may be filtered by phases."""

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_phases(phases=PropertyPhase.Liquid)

    assert dummy_data_set.number_of_properties == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_phases(phases=PropertyPhase.Liquid |
                                           PropertyPhase.Solid)

    assert dummy_data_set.number_of_properties == 2

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_phases(phases=PropertyPhase.Liquid |
                                           PropertyPhase.Solid |
                                           PropertyPhase.Gas)

    assert dummy_data_set.number_of_properties == 3


def test_filter_by_temperature():
    """A test to ensure that data sets may be filtered by temperature."""

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_temperature(min_temperature=287 * unit.kelvin,
                                         max_temperature=289 * unit.kelvin)

    assert dummy_data_set.number_of_properties == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_temperature(min_temperature=287 * unit.kelvin,
                                         max_temperature=299 * unit.kelvin)

    assert dummy_data_set.number_of_properties == 2

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_temperature(min_temperature=287 * unit.kelvin,
                                         max_temperature=309 * unit.kelvin)

    assert dummy_data_set.number_of_properties == 3


def test_filter_by_pressure():
    """A test to ensure that data sets may be filtered by pressure."""

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_pressure(min_pressure=0.4 * unit.atmosphere,
                                      max_pressure=0.6 * unit.atmosphere)

    assert dummy_data_set.number_of_properties == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_pressure(min_pressure=0.4 * unit.atmosphere,
                                      max_pressure=1.1 * unit.atmosphere)

    assert dummy_data_set.number_of_properties == 2

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_pressure(min_pressure=0.4 * unit.atmosphere,
                                      max_pressure=1.6 * unit.atmosphere)

    assert dummy_data_set.number_of_properties == 3


def test_filter_by_components():
    """A test to ensure that data sets may be filtered by the number of components."""

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_components(number_of_components=1)

    assert dummy_data_set.number_of_properties == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_components(number_of_components=2)

    assert dummy_data_set.number_of_properties == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_components(number_of_components=3)

    assert dummy_data_set.number_of_properties == 1


def test_filter_by_elements():
    """A test to ensure that data sets may be filtered by which elements their
    measured properties contain."""

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_elements('H', 'C')

    assert dummy_data_set.number_of_properties == 1

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_elements('H', 'C', 'N')

    assert dummy_data_set.number_of_properties == 2

    dummy_data_set = create_filterable_data_set()
    dummy_data_set.filter_by_elements('H', 'C', 'N', 'O')

    assert dummy_data_set.number_of_properties == 3


def test_filter_by_smiles():
    """A test to ensure that data sets may be filtered by which smiles their
    measured properties contain."""

    methanol_substance = Substance()
    methanol_substance.add_component(Substance.Component('CO'), Substance.MoleFraction(1.0))

    ethanol_substance = Substance()
    ethanol_substance.add_component(Substance.Component('CCO'), Substance.MoleFraction(1.0))

    property_a = create_dummy_property(Density)
    property_a.substance = methanol_substance

    property_b = create_dummy_property(Density)
    property_b.substance = ethanol_substance

    data_set = PhysicalPropertyDataSet()
    data_set.properties[methanol_substance.identifier] = [property_a]
    data_set.properties[ethanol_substance.identifier] = [property_b]

    data_set.filter_by_smiles('CO')

    assert data_set.number_of_properties == 1
    assert methanol_substance.identifier in data_set.properties
    assert ethanol_substance.identifier not in data_set.properties
