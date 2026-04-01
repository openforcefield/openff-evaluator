import pytest
from openff.toolkit.topology import Molecule

from openff.evaluator.datasets.curation.components.tautomers import (
    TAUTOMER_RULES,
    TautomerCategory,
    _enumerate_tautomers_cached,
    match_tautomer_categories,
)


def test_tautomer_category_table():
    assert set(TAUTOMER_RULES).issubset(set(TautomerCategory))


@pytest.mark.parametrize(
    "category,dominant_smiles,minor_smiles",
    [
        # Dominant and minor SMILES are known molecules that each SMARTS must
        # directly match (no tautomer enumeration needed).  This proves the
        # SMARTS are chemically meaningful, not merely syntactically valid.
        (TautomerCategory.AMIDE_IMIDIC_ACID, "CC(=O)N", "CC(=N)O"),
        (TautomerCategory.BETA_DIKETONE, "CC(O)=CC(=O)C", "CC(=O)CC(=O)C"),
        (TautomerCategory.KETO_ENOL_ALIPHATIC, "CCC(C)=O", "CCC(O)=C"),
        (TautomerCategory.KETO_ENOL_CYCLIC, "O=C1CCCCC1", "OC1=CCCCC1"),
        (TautomerCategory.THIOKETONE_THIOL_ALIPHATIC, "CC(=S)C", "CC(S)=C"),
        (TautomerCategory.KETO_ENOL_AROMATIC, "Oc1ccccc1", "O=C1C=CC=CC1"),
        (TautomerCategory.IMINE_ENAMINE_SECONDARY, "CCNC=C", "CCN=CC"),
        (TautomerCategory.IMINE_ENAMINE_PRIMARY, "CC=N", "CC(N)=C"),
        (TautomerCategory.ALPHA_AMINO_ACID, "NCC(=O)O", "N=CC(O)O"),
        # 4-methylimidazole: the two N-H tautomers have distinct SMILES, so each
        # SMARTS is tested against a different molecule and a swap bug is detectable.
        (TautomerCategory.ANNULAR_AZOLE, "Cc1c[nH]cn1", "Cc1cnc[nH]1"),
        (TautomerCategory.LACTAM_LACTIM, "O=C1CCCCN1", "OC1=NCCCC1"),
        (TautomerCategory.OXIME_NITROSO, "CC=NO", "CC(N=O)"),
        (TautomerCategory.KETENE_YNOL, "CC=C=O", "CC#CO"),
        (TautomerCategory.CARBOXYLIC_ACID_ENOL, "CC(=O)O", "C=C(O)O"),
        (TautomerCategory.ESTER_ENOL, "CC(=O)OC", "C=C(O)OC"),
        (TautomerCategory.AMIDE_ENOL, "CC(=O)NC", "C=C(O)NC"),
    ],
)
def test_smarts_match_known_forms(category, dominant_smiles, minor_smiles):
    """Each SMARTS must match a known molecule, not merely parse without error."""
    rule = TAUTOMER_RULES[category]
    dom_mol = Molecule.from_smiles(dominant_smiles, allow_undefined_stereo=True)
    min_mol = Molecule.from_smiles(minor_smiles, allow_undefined_stereo=True)
    assert dom_mol.chemical_environment_matches(
        rule.dominant_smarts
    ), f"{category} dominant_smarts did not match {dominant_smiles!r}"
    assert min_mol.chemical_environment_matches(
        rule.minor_smarts
    ), f"{category} minor_smarts did not match {minor_smiles!r}"


@pytest.mark.parametrize(
    "smiles,expected_category",
    [
        ("CC(=O)N", TautomerCategory.AMIDE_IMIDIC_ACID),  # acetamide
        ("CC(=O)CC(=O)C", TautomerCategory.BETA_DIKETONE),  # acetylacetone
        ("O=C1CCCCC1", TautomerCategory.KETO_ENOL_CYCLIC),  # cyclohexanone
        ("CC(C)=O", TautomerCategory.KETO_ENOL_ALIPHATIC),  # acetone
        ("c1ccccc1O", TautomerCategory.KETO_ENOL_AROMATIC),  # phenol
        ("Cc1ccc(O)cc1", TautomerCategory.KETO_ENOL_AROMATIC),  # para-cresol
        ("c1cnc[nH]1", TautomerCategory.ANNULAR_AZOLE),  # imidazole
        ("NCC(=O)O", TautomerCategory.ALPHA_AMINO_ACID),  # glycine
        ("O=C1CCCCN1", TautomerCategory.LACTAM_LACTIM),  # 2-piperidinone
        ("CC=NO", TautomerCategory.OXIME_NITROSO),  # acetaldoxime
        (
            "CC=N",
            TautomerCategory.IMINE_ENAMINE_PRIMARY,
        ),  # ethylideneamine (acetaldehyde primary imine)
        ("CCNC=C", TautomerCategory.IMINE_ENAMINE_SECONDARY),  # secondary enamine
        ("CC(=S)C", TautomerCategory.THIOKETONE_THIOL_ALIPHATIC),  # thioacetone
        ("CC=C=O", TautomerCategory.KETENE_YNOL),  # methylketene
        ("CC(=O)O", TautomerCategory.CARBOXYLIC_ACID_ENOL),  # acetic acid
        ("CC(=O)OC", TautomerCategory.ESTER_ENOL),  # methyl acetate
    ],
)
def test_match_tautomer_categories(smiles, expected_category):
    categories = match_tautomer_categories(smiles)
    assert expected_category in categories


@pytest.mark.parametrize(
    "smiles",
    [
        "C",  # methane
        "CC",  # ethane
        "CCO",  # ethanol
        "c1ccccc1",  # benzene
        "C1CCCCC1",  # cyclohexane
        "CCOCC",  # diethyl ether
    ],
)
def test_match_tautomer_categories_no_match(smiles):
    assert match_tautomer_categories(smiles) == frozenset()


def test_amide_enol_secondary_amide():
    """N-methyl acetamide (CC(=O)NC) triggers AMIDE_ENOL in raw SMARTS but is
    suppressed because AMIDE_IMIDIC_ACID co-fires and takes priority for amides."""
    raw = match_tautomer_categories("CC(=O)NC", suppress=False)
    assert TautomerCategory.AMIDE_ENOL in raw
    suppressed = match_tautomer_categories("CC(=O)NC")
    assert TautomerCategory.AMIDE_IMIDIC_ACID in suppressed
    assert TautomerCategory.AMIDE_ENOL not in suppressed


@pytest.mark.parametrize(
    "category,smiles,expect_dominant,expect_minor",
    [
        # AMIDE_ENOL: both dominant and minor are reachable from a single seed.
        (TautomerCategory.AMIDE_ENOL, "CC(=O)NC", True, True),
        # Plot uses an alternate lactam seed to make the lactam/lactim pair
        # visually distinct while still matching both sides.
        (TautomerCategory.LACTAM_LACTIM, "O=C1NCCC=C1", True, True),
    ],
)
def test_plot_example_tautomer_coverage(
    category,
    smiles,
    expect_dominant,
    expect_minor,
):
    Molecule.from_smiles(smiles, allow_undefined_stereo=True)

    rule = TAUTOMER_RULES[category]
    tautomers = _enumerate_tautomers_cached(smiles)

    has_dominant = any(
        tautomer.chemical_environment_matches(rule.dominant_smarts)
        for tautomer in tautomers
    )
    has_minor = any(
        tautomer.chemical_environment_matches(rule.minor_smarts)
        for tautomer in tautomers
    )

    assert has_dominant is expect_dominant
    assert has_minor is expect_minor
