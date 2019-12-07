"""
Units tests for propertyestimator.utils.packmol
"""
import numpy as np
import pytest

from propertyestimator import unit
from propertyestimator.utils import create_molecule_from_smiles, packmol


def _validate_water_results(topology, positions):
    """Performs a simple check that the positions and topology
    is consistent with a box of 10 water molecules"""
    assert topology is not None and positions is not None

    assert len(positions) == 30
    assert (
        unit.get_base_units(unit.angstrom)[-1]
        == unit.get_base_units(positions.units)[-1]
    )

    assert topology.getNumChains() == 1
    assert topology.getNumResidues() == 10
    assert topology.getNumAtoms() == 30
    assert topology.getNumBonds() == 20


def _validate_paracetamol_results(topology, positions):
    """Performs a simple check that the positions and topology
    is consistent with a box of 10 water molecules"""
    assert topology is not None and positions is not None

    assert len(positions) == 20
    assert (
        unit.get_base_units(unit.angstrom)[-1]
        == unit.get_base_units(positions.units)[-1]
    )

    assert topology.getNumChains() == 1
    assert topology.getNumResidues() == 1
    assert topology.getNumAtoms() == 20
    assert topology.getNumBonds() == 20


def test_packmol_packbox():
    """Test transitive graph reduction utility."""
    from simtk import unit as simtk_unit

    molecules = [create_molecule_from_smiles("O")]

    topology, positions = packmol.pack_box(
        molecules, [10], mass_density=1.0 * unit.grams / unit.milliliters
    )
    _validate_water_results(topology, positions)

    topology, positions = packmol.pack_box(
        molecules, [10], box_size=([20] * 3) * unit.angstrom
    )
    _validate_water_results(topology, positions)

    assert np.allclose(
        topology.getPeriodicBoxVectors()[0].value_in_unit(simtk_unit.angstrom),
        (22, 0, 0),
    )
    assert np.allclose(
        topology.getPeriodicBoxVectors()[1].value_in_unit(simtk_unit.angstrom),
        (0, 22, 0),
    )
    assert np.allclose(
        topology.getPeriodicBoxVectors()[2].value_in_unit(simtk_unit.angstrom),
        (0, 0, 22),
    )

    with pytest.raises(ValueError):
        packmol.pack_box(molecules, [10, 20], box_size=([20] * 3) * unit.angstrom)

    # Test something a bit more tricky than water
    molecules = [create_molecule_from_smiles("CC(=O)NC1=CC=C(C=C1)O")]

    topology, positions = packmol.pack_box(
        molecules, [1], box_size=([20] * 3) * unit.angstrom
    )
    _validate_paracetamol_results(topology, positions)
