from simtk import unit

from propertyestimator.datasets import PhysicalPropertyDataSet
from propertyestimator.properties import PropertyPhase, CalculationSource, Density, DielectricConstant, EnthalpyOfMixing
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import get_data_filename


def create_dummy_substance(number_of_components, elements=None):
    """Creates a substance with a given number of components,
    each containing the specified elements.

    Parameters
    ----------
    number_of_components : int
        The number of components to add to the substance.
    elements : list of str
        The elements that each component should containt.

    Returns
    -------
    Substance
        The created substance.
    """
    if elements is None:
        elements = ['C']

    substance = Substance()

    mole_fraction = 1.0 / number_of_components

    for index in range(number_of_components):

        smiles_pattern = ''.join(elements * (index + 1))

        substance.add_component(Substance.Component(smiles_pattern),
                                Substance.MoleFraction(mole_fraction))

    return substance


def create_dummy_property(property_class):
    """Create a dummy liquid property of specified type measured at
    298 K and 1 atm, with 2 components of methane and ethane.

    The property also contains a dummy receptor metadata entry.

    Parameters
    ----------
    property_class : type
        The type of property, e.g. Density, DielectricConstant...

    Returns
    -------
    PhysicalProperty
        The created property.
    """
    substance = create_dummy_substance(number_of_components=2)

    dummy_property = property_class(thermodynamic_state=ThermodynamicState(temperature=298 * unit.kelvin,
                                                                           pressure=1 * unit.atmosphere),
                                    phase=PropertyPhase.Liquid,
                                    substance=substance,
                                    value=10 * unit.gram,
                                    uncertainty=1 * unit.gram)
    
    dummy_property.source = CalculationSource(fidelity='dummy', provenance={})

    # Make sure the property has the meta data required for more
    # involved properties.
    dummy_property.metadata = {
        'receptor_mol2': 'unknown_path.mol2'
    }

    return dummy_property


def create_filterable_data_set():
    """Creates a dummy data with a diverse set of properties to
    be filtered, namely:

        - a liquid density measured at 298 K and 0.5 atm with 1 component containing only carbon.
        - a gaseous dielectric measured at 288 K and 1 atm with 2 components containing only nitrogen.
        - a solid EoM measured at 308 K and 1.5 atm with 3 components containing only oxygen.

    Returns
    -------
    PhysicalPropertyDataSet
        The created data set.
    """

    source = CalculationSource('Dummy', {})
    carbon_substance = create_dummy_substance(number_of_components=1, elements=['C'])

    density_property = Density(thermodynamic_state=ThermodynamicState(temperature=298 * unit.kelvin,
                                                                      pressure=0.5 * unit.atmosphere),
                               phase=PropertyPhase.Liquid,
                               substance=carbon_substance,
                               value=1 * unit.gram / unit.milliliter,
                               uncertainty=0.11 * unit.gram / unit.milliliter,
                               source=source)

    nitrogen_substance = create_dummy_substance(number_of_components=2, elements=['N'])

    dielectric_property = DielectricConstant(thermodynamic_state=ThermodynamicState(temperature=288 * unit.kelvin,
                                                                                    pressure=1 * unit.atmosphere),
                                             phase=PropertyPhase.Gas,
                                             substance=nitrogen_substance,
                                             value=1 * unit.dimensionless,
                                             uncertainty=0.11 * unit.dimensionless,
                                             source=source)

    oxygen_substance = create_dummy_substance(number_of_components=3, elements=['O'])

    enthalpy_property = EnthalpyOfMixing(thermodynamic_state=ThermodynamicState(temperature=308 * unit.kelvin,
                                                                                pressure=1.5 * unit.atmosphere),
                                         phase=PropertyPhase.Solid,
                                         substance=oxygen_substance,
                                         value=1 * unit.kilojoules_per_mole,
                                         uncertainty=0.11 * unit.kilojoules_per_mole,
                                         source=source)

    data_set = PhysicalPropertyDataSet()
    data_set.properties[carbon_substance.identifier] = [density_property]
    data_set.properties[nitrogen_substance.identifier] = [dielectric_property]
    data_set.properties[oxygen_substance.identifier] = [enthalpy_property]

    return data_set


def build_tip3p_smirnoff_force_field(file_path=None):
    """Combines the smirnoff99Frosst and tip3p offxml files
    into a single one which can be consumed by the property
    estimator.

    Parameters
    ----------
    file_path: str, optional
        The path to save the force field to.

    Returns
    -------
    str
        The file path to the combined force field file.
    """
    from openforcefield.typing.engines.smirnoff import ForceField
    
    smirnoff_force_field_path = get_data_filename('forcefield/smirnoff99Frosst.offxml')
    tip3p_force_field_path = get_data_filename('forcefield/tip3p.offxml')

    smirnoff_force_field_with_tip3p = ForceField(smirnoff_force_field_path,
                                                 tip3p_force_field_path)

    force_field_path = 'smirnoff99Frosst_tip3p.offxml' if file_path is None else file_path
    smirnoff_force_field_with_tip3p.to_file(force_field_path)

    return force_field_path
