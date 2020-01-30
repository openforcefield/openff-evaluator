"""
Units tests for propertyestimator.substances
"""
import numpy as np

from propertyestimator.substances import Component, ExactAmount, MoleFraction, Substance


def test_substance_from_components_no_amounts():

    water = Component(smiles="O")
    methanol = "CO"

    substance = Substance.from_components(water, methanol)
    substance.validate()

    assert substance.number_of_components == 2

    amounts = substance.get_amounts(substance.components[0])

    assert len(amounts) == 1

    amount = next(iter(amounts))

    assert isinstance(amount, MoleFraction)
    assert np.isclose(amount.value, 0.5)


def test_substance_from_components():

    sodium = Component(smiles="[Na+]")
    chloride = Component(smiles="[Cl-]")

    amounts = {
        sodium: [MoleFraction(0.75), ExactAmount(1)],
        chloride: [MoleFraction(0.25), ExactAmount(1)]
    }

    substance = Substance.from_components(sodium, chloride, amounts=amounts)

    assert len(substance) == 2

    sodium_amounts = substance.get_amounts(sodium)
    chlorine_amounts = substance.get_amounts(chloride)

    assert len(sodium_amounts) == 2
    assert len(chlorine_amounts) == 2

    molecule_counts = substance.get_molecules_per_component(6)

    assert len(molecule_counts) == 2

    assert molecule_counts[sodium.identifier] == 4
    assert molecule_counts[chloride.identifier] == 2


def test_substance_from_components_with_roles():

    sodium_solvent = Component(smiles="[Na+]", role=Component.Role.Solvent)
    chloride_solvent = Component(smiles="[Cl-]", role=Component.Role.Solvent)

    sodium_solute = Component(smiles="[Na+]", role=Component.Role.Solute)
    chloride_solute = Component(smiles="[Cl-]", role=Component.Role.Solute)

    amounts = {
        sodium_solute: MoleFraction(0.75),
        sodium_solvent: ExactAmount(1),
        chloride_solvent: MoleFraction(0.25),
        chloride_solute: ExactAmount(1),
    }

    substance = Substance.from_components(sodium_solvent, sodium_solute, chloride_solvent, chloride_solute, amounts=amounts)

    assert len(substance) == 4

    for component in substance:
        assert len(substance.get_amounts(component)) == 1


def test_len():

    substance = Substance.from_components("O", "CO")
    assert len(substance) == 2


def test_iter():

    water = Component("O")
    methanol = Component("CO")

    substance = Substance.from_components(water, methanol)

    components = []

    for component in substance:
        components.append(component)

    assert components == [water, methanol]
