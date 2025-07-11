"""
Units tests for openff.evaluator.datasets
"""

import numpy as np
import pytest
from openff.units import unit
from openff.utilities.utilities import get_data_file_path

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.datasets import PhysicalProperty, PropertyPhase
from openff.evaluator.datasets.thermoml import thermoml_property
from openff.evaluator.datasets.thermoml.thermoml import (
    ThermoMLDataSet,
    _Compound,
    _Constraint,
    _ConstraintType,
    _PureOrMixtureData,
    _unit_from_thermoml_string,
)
from openff.evaluator.plugins import register_default_plugins
from openff.evaluator.properties import EnthalpyOfMixing
from openff.evaluator.utils import get_data_filename

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
    assert returned_unit is not None and isinstance(returned_unit, unit.Unit)


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

    # Make sure the DOI was found from the enthalpy file
    for physical_property in data_set:
        if isinstance(physical_property, EnthalpyOfMixing):
            assert physical_property.source.doi != UNDEFINED
            assert physical_property.source.doi == "10.1016/j.jct.2008.12.004"

        else:
            assert physical_property.source.doi == ""
            assert physical_property.source.reference != UNDEFINED

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
    assert len(data_set.properties[0].substance) > 1


def test_thermoml_mole_constraints(caplog):
    """A collection of tests to ensure that the Mole fraction constraint is
    implemented correctly alongside solvent constraints."""

    # Mole fraction
    # This file contains a bad smiles to test Issue #620
    # Test that the file a) gets parsed
    # and b) logs a warning about parsing radicals

    with caplog.at_level("WARNING"):
        data_set = ThermoMLDataSet.from_file(
            get_data_filename("test/properties/mole.xml")
        )
    assert "An error occurred while parsing a compound" in caplog.text
    assert "radical" in caplog.text

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


def test_trim_missing_from_pandas():
    """
    Trim physical properties when some thermophysical data missing.

    See #653 for more context.
    """
    import pandas

    ThermoMLDataSet.from_pandas(
        pandas.read_csv(
            get_data_file_path(
                "data/test/properties/osmotic_subset.csv",
                "openff.evaluator",
            ),
            index_col=0,
        )
    )


class TestPureOrMixtureData:

    @staticmethod
    def _generate_dummy_compounds():
        solute = _Compound()
        solute.smiles = "C"
        solute.compound_index = 0

        solvent = _Compound()
        solvent.smiles = "O"
        solvent.compound_index = 1

        return solute, solvent

    @pytest.mark.parametrize(
        "molality, expected_mole_fraction",
        [(1.0, 0.0176965), (0.0, 0.0), (55.508, 0.5)],
    )
    def test_convert_molality_no_solvent_provided(
        self, molality: float, expected_mole_fraction: float
    ):
        constraint = _Constraint()
        constraint.type = _ConstraintType.ComponentMolality
        constraint.value = molality * unit.mole / unit.kilogram
        constraint.compound_index = 0

        solute, solvent = self._generate_dummy_compounds()

        mole_fractions = _PureOrMixtureData._convert_molality(
            [constraint],
            compounds={0: solute, 1: solvent},
        )
        assert np.isclose(
            mole_fractions[0].m_as(unit.dimensionless),
            expected_mole_fraction,
            atol=1e-5,
        )
        assert np.isclose(
            mole_fractions[1].m_as(unit.dimensionless),
            1 - expected_mole_fraction,
            atol=1e-5,
        )

    @pytest.mark.parametrize(
        "mass_fraction, expected_mole_fraction",
        [
            (1.0, 1.0),
            (0.0, 0.0),
            (0.5, 0.528962),
        ],
    )
    def test_convert_mass_fractions(
        self,
        mass_fraction,
        expected_mole_fraction,
    ):
        constraint = _Constraint()
        constraint.type = _ConstraintType.ComponentMassFraction
        constraint.value = mass_fraction
        constraint.compound_index = 0

        solute, solvent = self._generate_dummy_compounds()

        mole_fractions = _PureOrMixtureData._convert_mass_fractions(
            [constraint],
            compounds={0: solute, 1: solvent},
        )

        assert np.isclose(
            mole_fractions[0].m_as(unit.dimensionless),
            expected_mole_fraction,
            atol=1e-5,
        )
        assert np.isclose(
            mole_fractions[1].m_as(unit.dimensionless),
            1 - expected_mole_fraction,
            atol=1e-5,
        )

    @pytest.mark.parametrize(
        "mole_fraction, expected_solute_moles, expected_solvent_moles",
        [
            (1.0, 62.33416, 0.0),
            (0.0, 0.0, 55.5084),
            (0.5, 29.36177, 29.36177),
        ],
    )
    def test_solvent_mole_fractions_to_moles(
        self,
        mole_fraction: float,
        expected_solute_moles: float,
        expected_solvent_moles: float,
    ):
        solvent_mass = 1 * unit.kilogram
        solvent_mole_fractions = {0: mole_fraction, 1: 1 - mole_fraction}
        solvent_compounds = dict(enumerate(self._generate_dummy_compounds()))
        moles = _PureOrMixtureData._solvent_mole_fractions_to_moles(
            solvent_mass,
            solvent_mole_fractions,
            solvent_compounds,
        )
        assert np.isclose(
            moles[0].m_as(unit.moles),
            expected_solute_moles,
            atol=1e-5,
        )
        assert np.isclose(
            moles[1].m_as(unit.moles),
            expected_solvent_moles,
            atol=1e-5,
        )
