"""
Units tests for propertyestimator.datasets
"""

import pytest
from simtk import unit

from propertyestimator.datasets.thermoml import unit_from_thermoml_string
from propertyestimator.utils import get_data_filename

from propertyestimator.properties import PhysicalProperty
from propertyestimator.datasets import ThermoMLDataSet


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
