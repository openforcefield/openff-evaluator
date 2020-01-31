"""
Units tests for propertyestimator.substances
"""
import numpy as np

from propertyestimator.substances import Component, ExactAmount, MoleFraction, Substance


def test_substance_from_smiles_no_amounts():

    substance = Substance.from_smiles("O", "CO")
    substance.validate()

    assert substance.number_of_components == 2

    water_amounts = substance.get_amounts(substance.components[0])
    assert len(water_amounts) == 1

    assert isinstance(water_amounts[0], MoleFraction)
    assert np.isclose(water_amounts[0].value, 0.5)

    methanol_amounts = substance.get_amounts(substance.components[1])
    assert len(methanol_amounts) == 1

    assert isinstance(methanol_amounts[0], MoleFraction)
    assert np.isclose(methanol_amounts[0].value, 0.5)


def test_substance_from_smiles():

    amounts = {
        "O": MoleFraction(0.25),
        "CO": [MoleFraction(0.75), ExactAmount(1)]
    }

    substance = Substance.from_smiles("O", "CO", amounts=amounts)
    substance.validate()

    assert substance.number_of_components == 2

    water_amounts = substance.get_amounts(substance.components[0])
    assert len(water_amounts) == 1

    assert isinstance(water_amounts[0], MoleFraction)
    assert np.isclose(water_amounts[0].value, 0.25)

    methanol_amounts = substance.get_amounts(substance.components[1])
    assert len(methanol_amounts) == 2

    assert isinstance(methanol_amounts[0], MoleFraction)
    assert np.isclose(methanol_amounts[0].value, 0.75)

    assert isinstance(methanol_amounts[1], ExactAmount)
    assert np.isclose(methanol_amounts[1].value, 1)


def test_substance_components_no_amounts():

    substance = Substance.from_smiles("O", "CO")
    substance.validate()

    assert substance.number_of_components == 2

    water_amounts = substance.get_amounts(substance.components[0])
    assert len(water_amounts) == 1

    assert isinstance(water_amounts[0], MoleFraction)
    assert np.isclose(water_amounts[0].value, 0.5)

    methanol_amounts = substance.get_amounts(substance.components[1])
    assert len(methanol_amounts) == 1

    assert isinstance(methanol_amounts[0], MoleFraction)
    assert np.isclose(methanol_amounts[0].value, 0.5)


def test_substance_from_smiles():

    amounts = {
        "O": MoleFraction(0.25),
        "CO": [MoleFraction(0.75), ExactAmount(1)]
    }

    substance = Substance.from_smiles("O", "CO", amounts=amounts)
    substance.validate()

    assert substance.number_of_components == 2

    water_amounts = substance.get_amounts(substance.components[0])
    assert len(water_amounts) == 1

    assert isinstance(water_amounts[0], MoleFraction)
    assert np.isclose(water_amounts[0].value, 0.25)

    methanol_amounts = substance.get_amounts(substance.components[1])
    assert len(methanol_amounts) == 2

    assert isinstance(methanol_amounts[0], MoleFraction)
    assert np.isclose(methanol_amounts[0].value, 0.75)

    assert isinstance(methanol_amounts[1], ExactAmount)
    assert np.isclose(methanol_amounts[1].value, 1)


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
