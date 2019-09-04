"""
Units tests for propertyestimator.substances
"""
import numpy as np

from propertyestimator.substances import Substance


def test_add_mole_fractions():

    substance = Substance()

    substance.add_component(Substance.Component('C'), Substance.MoleFraction(0.5))
    substance.add_component(Substance.Component('C'), Substance.MoleFraction(0.5))

    assert substance.number_of_components == 1

    amounts = substance.get_amounts('C')

    assert len(amounts) == 1
    assert isinstance(amounts[0], Substance.MoleFraction)

    assert np.isclose(amounts[0].value, 1.0)


def test_multiple_amounts():

    substance = Substance()

    substance.add_component(Substance.Component('[Na+]'), Substance.MoleFraction(0.75))
    substance.add_component(Substance.Component('[Na+]'), Substance.ExactAmount(1))

    substance.add_component(Substance.Component('[Cl-]'), Substance.MoleFraction(0.25))
    substance.add_component(Substance.Component('[Cl-]'), Substance.ExactAmount(1))

    assert substance.number_of_components == 2

    sodium_amounts = substance.get_amounts('[Na+]')
    chlorine_amounts = substance.get_amounts('[Cl-]')

    assert len(sodium_amounts) == 2
    assert len(chlorine_amounts) == 2

    molecule_counts = substance.get_molecules_per_component(6)

    assert len(molecule_counts) == 2

    assert molecule_counts['[Na+]'] == 4
    assert molecule_counts['[Cl-]'] == 2
