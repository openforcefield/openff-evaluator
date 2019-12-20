"""
Units tests for propertyestimator.substances
"""
import numpy as np

from propertyestimator.substances import Component, ExactAmount, MoleFraction, Substance


def test_add_mole_fractions():

    substance = Substance()

    substance.add_component(Component("C"), MoleFraction(0.5))
    substance.add_component(Component("C"), MoleFraction(0.5))

    assert substance.number_of_components == 1

    amounts = substance.get_amounts(substance.components[0])

    assert len(amounts) == 1

    amount = next(iter(amounts))

    assert isinstance(amount, MoleFraction)
    assert np.isclose(amount.value, 1.0)


def test_multiple_amounts():

    substance = Substance()

    sodium = Component("[Na+]")
    chloride = Component("[Cl-]")

    substance.add_component(sodium, MoleFraction(0.75))
    substance.add_component(sodium, ExactAmount(1))

    substance.add_component(chloride, MoleFraction(0.25))
    substance.add_component(chloride, ExactAmount(1))

    assert substance.number_of_components == 2

    sodium_amounts = substance.get_amounts(sodium)
    chlorine_amounts = substance.get_amounts(chloride)

    assert len(sodium_amounts) == 2
    assert len(chlorine_amounts) == 2

    molecule_counts = substance.get_molecules_per_component(6)

    assert len(molecule_counts) == 2

    assert molecule_counts[sodium.identifier] == 4
    assert molecule_counts[chloride.identifier] == 2
