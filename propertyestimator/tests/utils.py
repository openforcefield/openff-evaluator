from simtk import unit

from propertyestimator.properties import PropertyPhase, CalculationSource
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import get_data_filename


def create_dummy_property(property_class):

    substance = Substance()

    substance.add_component(Substance.Component(smiles='C'), Substance.MoleFraction(0.5))
    substance.add_component(Substance.Component(smiles='CO'), Substance.MoleFraction(0.5))

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


def build_tip3p_smirnoff_force_field():
    """Combines the smirnoff99Frosst and tip3p offxml files
    into a single one which can be consumed by the property
    estimator.

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

    force_field_path = 'smirnoff99Frosst_tip3p.offxml'
    smirnoff_force_field_with_tip3p.to_file(force_field_path)

    return force_field_path
