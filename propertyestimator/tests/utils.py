from simtk import unit

from propertyestimator.properties import PropertyPhase, CalculationSource
from propertyestimator.substances import Mixture
from propertyestimator.thermodynamics import ThermodynamicState


def create_dummy_property(property_class):

    substance = Mixture()
    substance.add_component('C', 0.5)
    substance.add_component('CO', 0.5)

    dummy_property = property_class(thermodynamic_state=ThermodynamicState(temperature=298 * unit.kelvin,
                                                                           pressure=1 * unit.atmosphere),
                                    phase=PropertyPhase.Liquid,
                                    substance=substance,
                                    value=10 * unit.gram,
                                    uncertainty=1 * unit.gram)
    
    dummy_property.source = CalculationSource(fidelity='dummy', provenance={})

    return dummy_property
