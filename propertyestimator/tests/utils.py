from simtk import unit

from propertyestimator.properties import PropertyPhase, CalculationSource
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState


def create_dummy_property(property_class):

    substance = Substance()

    substance.add_component(Substance.Component(smiles='C'), mole_fraction=0.5)
    substance.add_component(Substance.Component(smiles='CO'), mole_fraction=0.5)

    dummy_property = property_class(thermodynamic_state=ThermodynamicState(temperature=298 * unit.kelvin,
                                                                           pressure=1 * unit.atmosphere),
                                    phase=PropertyPhase.Liquid,
                                    substance=substance,
                                    value=10 * unit.gram,
                                    uncertainty=1 * unit.gram)
    
    dummy_property.source = CalculationSource(fidelity='dummy', provenance={})

    return dummy_property
