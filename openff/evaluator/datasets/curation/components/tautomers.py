import functools
from dataclasses import dataclass
from enum import Enum
from typing import Dict, FrozenSet, Iterable, Optional, Tuple

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


class TautomerCategory(str, Enum):
    BETA_DIKETONE = "BETA_DIKETONE"
    KETO_ENOL_ALIPHATIC = "KETO_ENOL_ALIPHATIC"
    KETO_ENOL_CYCLIC = "KETO_ENOL_CYCLIC"
    THIOKETONE_THIOL_ALIPHATIC = "THIOKETONE_THIOL_ALIPHATIC"
    KETO_ENOL_AROMATIC = "KETO_ENOL_AROMATIC"
    IMINE_ENAMINE_SECONDARY = "IMINE_ENAMINE_SECONDARY"
    IMINE_ENAMINE_PRIMARY = "IMINE_ENAMINE_PRIMARY"
    ALPHA_AMINO_ACID = "ALPHA_AMINO_ACID"
    ANNULAR_AZOLE = "ANNULAR_AZOLE"
    LACTAM_LACTIM = "LACTAM_LACTIM"
    AMIDE_IMIDIC_ACID = "AMIDE_IMIDIC_ACID"
    OXIME_NITROSO = "OXIME_NITROSO"
    KETENE_YNOL = "KETENE_YNOL"
    CARBOXYLIC_ACID_ENOL = "CARBOXYLIC_ACID_ENOL"
    ESTER_ENOL = "ESTER_ENOL"
    AMIDE_ENOL = "AMIDE_ENOL"


@dataclass(frozen=True)
class TautomerRule:
    dominant_smarts: str
    minor_smarts: str


TAUTOMER_RULES: Dict[TautomerCategory, TautomerRule] = {
    # Hansen 2023, doi: 10.3390/encyclopedia3010013
    # ratio of keto/enol forms depends on polarity of solvent and character of substituents.
    # recommend not whitelisting this category, but it is included here for completeness
    # and potential future work.
    TautomerCategory.BETA_DIKETONE: TautomerRule(
        dominant_smarts="[#1:5]-[#8X2:1]-[#6X3:2]=[#6X3:3]-[#6X3:4]=[#8X1:6]",
        minor_smarts="[#8X1:1]=[#6X3:2]-[#6AX4:3](-[#1:5])-[#6X3:4]=[#8X1:6]",
    ),
    # classic example of keto/enol tautomerism,
    # keto form dominates; stabilized by carbonyl,
    # see Clayden, Greeves and Warren, Organic Chemistry, 2nd Ed., section 21 (p. 471)
    TautomerCategory.KETO_ENOL_ALIPHATIC: TautomerRule(
        dominant_smarts="[#8;X1;!r:1]=[#6X3;!r;!$([#6X3]-[#8X2]);!$([#6X3]-[#7]):2]-[#6AX4;!r;!$([#6AX4]~[#7,#16,#15]);!$([#6AX4]~[c]):3]-[#1:4]",
        minor_smarts="[#1:4]-[#8;X2H1;!r:1]-[#6X3;!r;!$([#6X3]-[#7]):2]=[#6AX3;!$([#6AX3]~[#7,#16,#15]);!$([#6AX3]-[#6X3]=[#8,#16]);!$([#6AX3]~[c]):3]",
    ),
    # separate out this category to avoid confusion with aromatic enols
    # ketone still dominant
    TautomerCategory.KETO_ENOL_CYCLIC: TautomerRule(
        dominant_smarts="[#8;X1:1]=[#6X3;R;A;!$([#6X3;R;A]~[#6X3;R]=[#8X1]);!$([#6X3;R;A]~[#8;R]):2]-[#6X4;A;R;!$([#6X4;R]~[#7,#16,#15]);!$([#6X4;R]~[#8;R]):3]-[#1:4]",
        minor_smarts="[#1:4]-[#8;X2H1:1]-[#6X3;A;R;!$([#6X3;R]~[#7;R]):2]=[#6X3;A;R;!$([#6X3;R]~[#7,#16,#15]);!$([#6X3;R]~[#6X3]=[#16]):3]",
    ),
    # need this because RDKit finds tautomers for acids
    TautomerCategory.CARBOXYLIC_ACID_ENOL: TautomerRule(
        dominant_smarts="[#1:1]-[#8X2H1:2]-[#6X3:3](-[#6;!$([*](-[#6X3]=[#8X1])~[#6X3]~[#8]):5]-[#1:6])=[#8X1:4]",
        minor_smarts="[#1:1]-[#8X2H1:2]-[#6X3:3](=[#6;!$([*]-[#6X3]=[#8X1]):5])-[#8X2:4]-[#1:6]",
    ),
    # ester enol; ester strongly dominates. analogous to CARBOXYLIC_ACID_ENOL
    # but the alcohol oxygen bears a carbon substituent rather than H.
    TautomerCategory.ESTER_ENOL: TautomerRule(
        dominant_smarts="[#6:1]-[#8X2H0:2]-[#6X3:3](-[#6;!$([*](-[#6X3]=[#8X1])~[#6X3]~[#8]):5]-[#1:6])=[#8X1:4]",
        minor_smarts="[#6:1]-[#8X2H0:2]-[#6X3:3](=[#6;!$([*]-[#6X3]=[#8X1]):5])-[#8X2:4]-[#1:6]",
    ),
    TautomerCategory.AMIDE_ENOL: TautomerRule(
        dominant_smarts="[#1:6]-[#7X3;!r;!$([#7X3]-[#6X3]=[#7]):1]-[#6X3:2](=[#8X1:3])-[#6X4:4]-[#1:5]",
        minor_smarts="[#1:6]-[#7X3;!r;!$([#7X3]-[#6X3]=[#7]):1]-[#6X3:2](-[#8X2:3]-[#1:5])=[#6X3:4]",
    ),
    # aka thione/thiol.
    # likely low-populated, as thioketones and enethiols are inherently unstable (Bruno, Steer and Mezey, 1982).
    # in general tautomers interconvert and both forms can be present at equilibrium
    # (see Bruno, Steer and Mezey, 1982; Selzer and Rappoport 1996) although the thione may be more stable
    # (see Ashry et al 2017, Journal of Molecular Structure).
    # recommend not whitelisting this category, but it is included here for completeness and potential future work.
    TautomerCategory.THIOKETONE_THIOL_ALIPHATIC: TautomerRule(
        dominant_smarts="[#16;X1;!r:1]=[#6X3;!r:2]-[#6AX4;!r:3]-[#1:4]",
        minor_smarts="[#1:4]-[#16;X2H1;!r:1]-[#6X3;!r:2]=[#6AX3;!r:3]",
    ),
    # other classic example of keto/enol tautomerism, but stabilized by aromaticity
    # see Clayden again, same page. Enol is dominant.
    TautomerCategory.KETO_ENOL_AROMATIC: TautomerRule(
        dominant_smarts="[#1:4]-[#8,#16;X2H1:1]-[#6aX3:2]~[#6aX3:3]~[#6a:5]",
        minor_smarts="[#8,#16;X1:1]=[#6X3R;r5,r6:2]-[#6X4R;r5,r6:3](-[#1:4])",
    ),
    # see Clayden 2nd edition p. 456, for secondary, enamine is dominant
    TautomerCategory.IMINE_ENAMINE_SECONDARY: TautomerRule(
        dominant_smarts="[#1,#6:4][#7X3;!r:1]([#1,#6:5])[#6X3:2]=[#6X3:3]",
        minor_smarts="[#1,#6:5][#7X2;H0;!r:1]=[#6X3:2][#6AX4:3][#1:4]",
    ),
    # Clayden 2nd edition p. 455, for primary, imine is dominant
    TautomerCategory.IMINE_ENAMINE_PRIMARY: TautomerRule(
        dominant_smarts="[#7X2;H1:1]=[#6X3:2][#6AX4;!$([#6AX4]~[#6X3]=[#8,#16]);!$([#6AX4]~[#7X3;+1]):3][#1:4]",
        minor_smarts="[#7X3;H2:1][#6X3:2]=[#6X3:3]",
    ),
    # amino acid form (NH2-CαH-C=O) dominates; the imino-diol minor tautomer
    # is chemically exotic and mostly arises as a spurious RDKit enumeration product.
    # recommend whitelist!
    TautomerCategory.ALPHA_AMINO_ACID: TautomerRule(
        dominant_smarts="[#7X3;H2:1]-[#6AX4:2](-[#1:5])-[#6X3:3]=[#8X1:4]",
        minor_smarts="[#1:5]-[#7X2:1]=[#6X3:2]-[#6X4:3](-[#8X2H1:4])-[#8X2H1:6]",
    ),
    # no universally dominant form; ratio depends on substitution and solvent.
    # generally quite complicated and varies by solvent and substituent,
    # only listed here for completeness sake and future work.
    # see Katritzsky 1970: the prototropic tautomerism of heteroaromatic compounds
    # (probably newer sources too)
    TautomerCategory.ANNULAR_AZOLE: TautomerRule(
        dominant_smarts="[#1:3]-[#7X3R:1]@[#6R:2]@[#7X2R:4]",
        minor_smarts="[#7X2R:1]@[#6R:2]@[#7X3R:4]-[#1:3]",
    ),
    # lactam usually dominates in 5-membered rings (Pilgram 1984 4.36.3.1.2)
    # and in saturated lactams (Jose and De 2025, J Comp Chem)
    TautomerCategory.LACTAM_LACTIM: TautomerRule(
        dominant_smarts="[#1:4]-[#7X3R;A;!$([#7;r5]@[#8R,#7R]):1]@[#6R;A:2]=[#8X1!R:3]",
        minor_smarts="[#7RH0;A;!$([#7;r5]@[#8R,#7R]):1]@[#6R;A:2]-[#8X2H1!R:3]-[#1:4]",
    ),
    # amide dominates
    TautomerCategory.AMIDE_IMIDIC_ACID: TautomerRule(
        dominant_smarts="[#1:4]-[#7X3;!r;!$([#7X3]-[#6X3]=[#7]):1]-[#6X3:2]=[#8X1:3]",
        minor_smarts="[#7X2;!r:1]=[#6X3:2]-[#8X2H1:3]-[#1:4]",
    ),
    # oxime is stable -- Clayden
    TautomerCategory.OXIME_NITROSO: TautomerRule(
        dominant_smarts="[#1:4]-[#8X2:1]-[#7X2:2]=[#6X3:3]",
        minor_smarts="[#8X1:1]=[#7X2:2]-[#6X4:3]-[#1:4]",
    ),
    # ketenes more stable
    # Zhdankin 2005, Comprehensive Organic Functional Group Transformations II, 2.21.2.1.1
    TautomerCategory.KETENE_YNOL: TautomerRule(
        dominant_smarts="[#1:4]-[#6X3:1]=[#6X2:2]=[#8X1:3]",
        minor_smarts="[#6X2:1]#[#6X2:2]-[#8X2:3]-[#1:4]",
    ),
}


_TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()

_SUPPRESSED_CATEGORY_MATCHES = {
    TautomerCategory.BETA_DIKETONE: frozenset(
        {
            TautomerCategory.KETO_ENOL_ALIPHATIC,
            # beta-keto acids and beta-keto esters should be classified as
            # BETA_DIKETONE, not as CARBOXYLIC_ACID_ENOL / ESTER_ENOL.
            # LW: I tried for some time and could not generalize exclusive SMARTS,
            # nor could Claude, so this seemed easier
            TautomerCategory.CARBOXYLIC_ACID_ENOL,
            TautomerCategory.ESTER_ENOL,
        }
    ),
    TautomerCategory.AMIDE_IMIDIC_ACID: frozenset(
        {
            TautomerCategory.IMINE_ENAMINE_PRIMARY,
            TautomerCategory.KETO_ENOL_ALIPHATIC,
            TautomerCategory.IMINE_ENAMINE_SECONDARY,
            TautomerCategory.AMIDE_ENOL,
            # quite a few things matching the elementary amide pattern
        }
    ),
}


def _to_category_tuple(
    categories: Optional[Iterable[TautomerCategory]],
) -> Tuple[str, ...]:
    if categories is None:
        return tuple(
            category.value
            for category in TautomerCategory
        )

    return tuple(sorted(category.value for category in categories))


@functools.lru_cache(maxsize=4096)
def _enumerate_tautomers_cached(smiles: str) -> tuple:
    from openff.toolkit.topology import Molecule

    rdmol = Chem.MolFromSmiles(smiles)
    if rdmol is None:
        return tuple()

    tautomers = []
    seen = set()
    # Enumerate always includes the input molecule, so no need to pre-seed `seen`.
    for t in _TAUTOMER_ENUMERATOR.Enumerate(rdmol):
        smi = Chem.MolToSmiles(t, isomericSmiles=True)
        if smi not in seen:
            seen.add(smi)
            try:
                tautomers.append(Molecule.from_smiles(smi, allow_undefined_stereo=True))
            except Exception:
                pass

    return tuple(tautomers)


@functools.lru_cache(maxsize=4096)
def _match_tautomer_categories_cached(
    smiles: str, category_names: Tuple[str, ...]
) -> FrozenSet[TautomerCategory]:
    """Return the raw (unsuppressed) set of matched categories."""
    tautomers = _enumerate_tautomers_cached(smiles)

    if not tautomers:
        return frozenset()

    matched_categories = set()

    for category_name in category_names:
        category = TautomerCategory(category_name)
        rule = TAUTOMER_RULES[category]

        has_dominant = any(
            t.chemical_environment_matches(rule.dominant_smarts) for t in tautomers
        )
        has_minor = any(
            t.chemical_environment_matches(rule.minor_smarts) for t in tautomers
        )

        if has_dominant and has_minor:
            matched_categories.add(category)

    return frozenset(matched_categories)


def _apply_suppression(
    matched: FrozenSet[TautomerCategory],
) -> FrozenSet[TautomerCategory]:
    result = set(matched)
    for category, suppressed in _SUPPRESSED_CATEGORY_MATCHES.items():
        if category in result:
            result.difference_update(suppressed)
    return frozenset(result)


def match_tautomer_categories(
    smiles: str,
    categories: Optional[Iterable[TautomerCategory]] = None,
    suppress: bool = True,
) -> FrozenSet[TautomerCategory]:
    raw = _match_tautomer_categories_cached(smiles, _to_category_tuple(categories))
    return _apply_suppression(raw) if suppress else raw
