"""
Units tests for openff.evaluator.utils.packmol
"""

import numpy as np
import pytest
from openff.toolkit import ForceField, Molecule
from openff.units import unit

from openff.evaluator.utils import packmol
from openff.evaluator.utils.packmol import PackmolRuntimeException


def test_packmol_box_size():

    molecules = [Molecule.from_smiles("O")]

    topology, _ = packmol.pack_box(molecules, [10], box_size=([20] * 3) * unit.angstrom)

    assert topology is not None

    assert len(topology.chains) == 10  # should be 1 ...
    assert len(topology.residues) == 10
    assert topology.n_atoms == 30
    assert topology.n_bonds == 20

    assert all(x.residue_name == "HOH" for x in topology.residues)

    assert np.allclose(topology.box_vectors.m_as("nanometer").diagonal(), 2.2)


def test_packmol_bad_input():

    molecules = [Molecule.from_smiles("O")]

    with pytest.raises(ValueError):
        packmol.pack_box(molecules, [10, 20], box_size=([20] * 3) * unit.angstrom)


def test_packmol_failed():

    molecules = [Molecule.from_smiles("O")]

    with pytest.raises(PackmolRuntimeException):
        packmol.pack_box(molecules, [10], box_size=([0.1] * 3) * unit.angstrom)


def test_packmol_water():

    molecules = [Molecule.from_smiles("O")]

    topology, _ = packmol.pack_box(
        molecules,
        [10],
        mass_density=1.0 * unit.grams / unit.milliliters,
    )

    assert topology is not None

    assert len(topology.chains) == 10  # should be 1 ...
    assert len(topology.residues) == 10
    assert topology.n_atoms == 30
    assert topology.n_bonds == 20

    assert all(residue.residue_name == "HOH" for residue in topology.residues)


def test_packmol_ions():

    molecules = [
        Molecule.from_smiles("[Na+]"),
        Molecule.from_smiles("[Cl-]"),
        Molecule.from_smiles("[K+]"),
    ]

    topology, _ = packmol.pack_box(
        molecules, [1, 1, 1], box_size=([20] * 3) * unit.angstrom
    )

    assert topology is not None

    assert len(topology.chains) == 3
    assert len(topology.residues) == 3
    assert topology.n_atoms == 3
    assert topology.n_bonds == 0

    assert topology.residues[0].residue_name == "Na+"
    assert topology.residues[1].residue_name == "Cl-"
    assert topology.residues[2].residue_name == "K+"

    assert topology.atom(0).name == "Na+"
    assert topology.atom(1).name == "Cl-"
    assert topology.atom(2).name == "K+"


def test_packmol_paracetamol():

    # Test something a bit more tricky than water
    molecules = [Molecule.from_smiles("CC(=O)NC1=CC=C(C=C1)O")]

    topology, _ = packmol.pack_box(molecules, [1], box_size=([20] * 3) * unit.angstrom)

    assert topology is not None

    assert len(topology.chains) == 1
    assert len(topology.residues) == 1
    assert topology.n_atoms == 20
    assert topology.n_bonds == 20


amino_residues = {
    "C[C@H](N)C(=O)O": "ALA",
    # Undefined stereochemistry error.
    # "N=C(N)NCCC[C@H](N)C(=O)O": "ARG",
    "NC(=O)C[C@H](N)C(=O)O": "ASN",
    "N[C@@H](CC(=O)O)C(=O)O": "ASP",
    "N[C@@H](CS)C(=O)O": "CYS",
    "N[C@@H](CCC(=O)O)C(=O)O": "GLU",
    "NC(=O)CC[C@H](N)C(=O)O": "GLN",
    "NCC(=O)O": "GLY",
    "N[C@@H](Cc1c[nH]cn1)C(=O)O": "HIS",
    "CC[C@H](C)[C@H](N)C(=O)O": "ILE",
    "CC(C)C[C@H](N)C(=O)O": "LEU",
    "NCCCC[C@H](N)C(=O)O": "LYS",
    "CSCC[C@H](N)C(=O)O": "MET",
    "N[C@@H](Cc1ccccc1)C(=O)O": "PHE",
    "O=C(O)[C@@H]1CCCN1": "PRO",
    "N[C@@H](CO)C(=O)O": "SER",
    "C[C@@H](O)[C@H](N)C(=O)O": "THR",
    "N[C@@H](Cc1c[nH]c2ccccc12)C(=O)O": "TRP",
    "N[C@@H](Cc1ccc(O)cc1)C(=O)O": "TYR",
    "CC(C)[C@H](N)C(=O)O": "VAL",
}


def test_amino_acid_residue_information():
    smiles = [*amino_residues]

    molecules = [Molecule.from_smiles(x) for x in smiles]
    counts = [1] * len(smiles)

    topology, _ = packmol.pack_box(
        molecules, counts, box_size=([1000] * 3) * unit.angstrom
    )

    for residue, smiles in zip(
        topology.residues,
        smiles,
    ):
        assert residue.residue_name == amino_residues[smiles]

    # dummy check to make sure enough chemical information is present to parametrize,
    # even though this force field is not necessarily designed for amino acids
    ForceField("openff_unconstrained-2.3.0-rc2.offxml").create_openmm_system(topology)
