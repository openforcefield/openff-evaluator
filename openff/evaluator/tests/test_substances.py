"""
Units tests for openff.evaluator.substances
"""
import numpy as np
import pytest

from openff.evaluator.substances import Component, ExactAmount, MoleFraction, Substance


@pytest.mark.parametrize(
    "smiles,expected",
    [
        ("C1=CC=CC=C1", "c1ccccc1"),
        ("c1ccccc1", "c1ccccc1"),
        ("[C@H](F)(Cl)Br", "F[C@@H](Cl)Br"),
        ("C(F)(Cl)Br", "FC(Cl)Br"),
    ],
)
def test_component_standardization(smiles, expected):

    component = Component(smiles=smiles)
    assert component.smiles == expected


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


def test_truncate_n_molecules():

    substance = Substance()

    substance.add_component(
        component=Component(smiles="[Na+]"), amount=MoleFraction(0.00267),
    )
    substance.add_component(
        component=Component(smiles="[Cl-]"), amount=MoleFraction(0.00267),
    )
    substance.add_component(
        component=Component(smiles="O"), amount=MoleFraction(1.0 - 2.0 * 0.00267)
    )

    # Attempt to get the number of molecules without truncating.
    with pytest.raises(ValueError):
        substance.get_molecules_per_component(1000, truncate_n_molecules=False)

    # Attempt to get the number of molecules with truncating.
    molecule_counts = substance.get_molecules_per_component(
        1000, truncate_n_molecules=True
    )

    assert molecule_counts == {"[Na+]{solv}": 3, "[Cl-]{solv}": 3, "O{solv}": 994}


def test_substance_len():

    substance = Substance.from_components("C", "CC", "CCC", "CCC")
    assert len(substance) == 3
