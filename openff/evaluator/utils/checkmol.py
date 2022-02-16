import functools
from enum import Enum

from openff.evaluator.utils.exceptions import MissingOptionalDependency


class ChemicalEnvironment(Enum):

    Alkane = "Alkane"
    Cation = "Cation"
    Anion = "Anion"
    Carbonyl = "Carbonyl"
    Aldehyde = "Aldehyde"
    Ketone = "Ketone"
    Thiocarbonyl = "Thiocarbonyl"
    Thioaldehyde = "Thioaldehyde"
    Thioketone = "Thioketone"
    Imine = "Imine"
    Hydrazone = "Hydrazone"
    Semicarbazone = "Semicarbazone"
    Thiosemicarbazone = "Thiosemicarbazone"
    Oxime = "Oxime"
    OximeEther = "Oxime Ether"
    Ketene = "Ketene"
    KeteneAcetalDeriv = "Ketene Acetal Derivative"
    CarbonylHydrate = "Carbonyl Hydrate"
    Hemiacetal = "Hemiacetal"
    Acetal = "Acetal"
    Hemiaminal = "Hemiaminal"
    Aminal = "Aminal"
    Thiohemiaminal = "Thiohemiaminal"
    Thioacetal = "Thioacetal"
    Enamine = "Enamine"
    Enol = "Enol"
    Enolether = "Enolether"
    Hydroxy = "Hydroxy"
    Alcohol = "Alcohol"
    PrimaryAlcohol = "Primary Alcohol"
    SecondaryAlcohol = "Secondary Alcohol"
    TertiaryAlcohol = "Tertiary Alcohol"
    Diol_1_2 = "1,2 Diol"
    Aminoalcohol_1_2 = "1,2 Aminoalcohol"
    Phenol = "Phenol"
    Diphenol_1_2 = "1,2 Diphenol"
    Enediol = "Enediol"
    Ether = "Ether"
    Dialkylether = "Dialkylether"
    Alkylarylether = "Alkylarylether"
    Diarylether = "Diarylether"
    Thioether = "Thioether"
    Disulfide = "Disulfide"
    Peroxide = "Peroxide"
    Hydroperoxide = "Hydroperoxide"
    Hydrazine = "Hydrazine"
    Hydroxylamine = "Hydroxylamine"
    Amine = "Amine"
    PrimaryAmine = "Primary Amine"
    PrimaryAliphAmine = "Primary Aliphatic Amine"
    PrimaryAromAmine = "Primary Aromatic Amine"
    SecondaryAmine = "Secondary Amine"
    SecondaryAliphAmine = "Secondary Aliphatic Amine"
    SecondaryMixedAmine = "Secondary Mixed Amine"
    SecondaryAromAmine = "Secondary Aromatic Amine"
    TertiaryAmine = "Tertiary Amine"
    TertiaryAliphAmine = "Tertiary Aliphatic Amine"
    TertiaryMixedAmine = "Tertiary Mixed Amine"
    TertiaryAromAmine = "Tertiary Aromatic Amine"
    QuartAmmonium = "Quart Ammonium"
    NOxide = "NOxide"
    HalogenDeriv = "Halogen Derivative"
    AlkylHalide = "Alkyl Halide"
    AlkylFluoride = "Alkyl Fluoride"
    AlkylChloride = "Alkyl Chloride"
    AlkylBromide = "Alkyl Bromide"
    AlkylIodide = "Alkyl Iodide"
    ArylHalide = "Aryl Halide"
    ArylFluoride = "Aryl Fluoride"
    ArylChloride = "Aryl Chloride"
    ArylBromide = "Aryl Bromide"
    ArylIodide = "Aryl Iodide"
    Organometallic = "Organometallic"
    Organolithium = "Organolithium"
    Organomagnesium = "Organomagnesium"
    CarboxylicAcidDeriv = "Carboxylic Acid Derivative"
    CarboxylicAcid = "Carboxylic Acid"
    CarboxylicAcidSalt = "Carboxylic Acid Salt"
    CarboxylicAcidEster = "Carboxylic Acid Ester"
    Lactone = "Lactone"
    CarboxylicAcidAmide = "Carboxylic Acid Amide"
    CarboxylicAcidPrimaryAmide = "Carboxylic Acid Primary Amide"
    CarboxylicAcidSecondaryAmide = "Carboxylic Acid Secondary Amide"
    CarboxylicAcidTertiaryAmide = "Carboxylic Acid Tertiary Amide"
    Lactam = "Lactam"
    CarboxylicAcidHydrazide = "Carboxylic Acid Hydrazide"
    CarboxylicAcidAzide = "Carboxylic Acid Azide"
    HydroxamicAcid = "Hydroxamic Acid"
    CarboxylicAcidAmidine = "Carboxylic Acid Amidine"
    CarboxylicAcidAmidrazone = "Carboxylic Acid Amidrazone"
    Nitrile = "Nitrile"
    AcylHalide = "Acyl Halide"
    AcylFluoride = "Acyl Fluoride"
    AcylChloride = "Acyl Chloride"
    AcylBromide = "Acyl Bromide"
    AcylIodide = "Acyl Iodide"
    AcylCyanide = "Acyl Cyanide"
    ImidoEster = "Imido Ester"
    ImidoylHalide = "Imidoyl Halide"
    ThiocarboxylicAcidDeriv = "Thiocarboxylic Acid Derivative"
    ThiocarboxylicAcid = "Thiocarboxylic Acid"
    ThiocarboxylicAcidEster = "Thiocarboxylic Acid Ester"
    Thiolactone = "Thiolactone"
    ThiocarboxylicAcidAmide = "Thiocarboxylic Acid Amide"
    Thiolactam = "Thiolactam"
    ImidoThioester = "Imido Thioester"
    Oxohetarene = "Oxohetarene"
    Thioxohetarene = "Thioxohetarene"
    Iminohetarene = "Iminohetarene"
    OrthocarboxylicAcidDeriv = "Orthocarboxylic Acid Derivative"
    CarboxylicAcidOrthoester = "Carboxylic Acid Orthoester"
    CarboxylicAcidAmideAcetal = "Carboxylic Acid Amide Acetal"
    CarboxylicAcidAnhydride = "Carboxylic Acid Anhydride"
    CarboxylicAcidImide = "Carboxylic Acid Imide"
    CarboxylicAcidUnsubstImide = "Carboxylic Acid Unsubst Imide"
    CarboxylicAcidSubstImide = "Carboxylic Acid Subst Imide"
    Co2Deriv = "CO2 Derivative"
    CarbonicAcidDeriv = "Carbonic Acid Derivative"
    CarbonicAcidMonoester = "Carbonic Acid Monoester"
    CarbonicAcidDiester = "Carbonic Acid Diester"
    CarbonicAcidEsterHalide = "Carbonic Acid Ester Halide"
    ThiocarbonicAcidDeriv = "Thiocarbonic Acid Derivative"
    ThiocarbonicAcidMonoester = "Thiocarbonic Acid Monoester"
    ThiocarbonicAcidDiester = "Thiocarbonic Acid Diester"
    ThiocarbonicAcidEsterHalide = "Thiocarbonic Acid Ester Halide"
    CarbamicAcidDeriv = "Carbamic Acid Derivative"
    CarbamicAcid = "Carbamic Acid"
    CarbamicAcidEster = "Carbamic Acid Ester"
    CarbamicAcidHalide = "Carbamic Acid Halide"
    ThiocarbamicAcidDeriv = "Thiocarbamic Acid Derivative"
    ThiocarbamicAcid = "Thiocarbamic Acid"
    ThiocarbamicAcidEster = "Thiocarbamic Acid Ester"
    ThiocarbamicAcidHalide = "Thiocarbamic Acid Halide"
    Urea = "Urea"
    Isourea = "Isourea"
    Thiourea = "Thiourea"
    Isothiourea = "Isothiourea"
    Guanidine = "Guanidine"
    Semicarbazide = "Semicarbazide"
    Thiosemicarbazide = "Thiosemicarbazide"
    Azide = "Azide"
    AzoCompound = "Azo Compound"
    DiazoniumSalt = "Diazonium Salt"
    Isonitrile = "Isonitrile"
    Cyanate = "Cyanate"
    Isocyanate = "Isocyanate"
    Thiocyanate = "Thiocyanate"
    Isothiocyanate = "Isothiocyanate"
    Carbodiimide = "Carbodiimide"
    NitrosoCompound = "Nitroso Compound"
    NitroCompound = "Nitro Compound"
    Nitrite = "Nitrite"
    Nitrate = "Nitrate"
    SulfuricAcidDeriv = "Sulfuric Acid Derivative"
    SulfuricAcid = "Sulfuric Acid"
    SulfuricAcidMonoester = "Sulfuric Acid Monoester"
    SulfuricAcidDiester = "Sulfuric Acid Diester"
    SulfuricAcidAmideEster = "Sulfuric Acid Amide Ester"
    SulfuricAcidAmide = "Sulfuric Acid Amide"
    SulfuricAcidDiamide = "Sulfuric Acid Diamide"
    SulfurylHalide = "Sulfuryl Halide"
    SulfonicAcidDeriv = "Sulfonic Acid Derivative"
    SulfonicAcid = "Sulfonic Acid"
    SulfonicAcidEster = "Sulfonic Acid Ester"
    Sulfonamide = "Sulfonamide"
    SulfonylHalide = "Sulfonyl Halide"
    Sulfone = "Sulfone"
    Sulfoxide = "Sulfoxide"
    SulfinicAcidDeriv = "Sulfinic Acid Derivative"
    SulfinicAcid = "Sulfinic Acid"
    SulfinicAcidEster = "Sulfinic Acid Ester"
    SulfinicAcidHalide = "Sulfinic Acid Halide"
    SulfinicAcidAmide = "Sulfinic Acid Amide"
    SulfenicAcidDeriv = "Sulfenic Acid Derivative"
    SulfenicAcid = "Sulfenic Acid"
    SulfenicAcidEster = "Sulfenic Acid Ester"
    SulfenicAcidHalide = "Sulfenic Acid Halide"
    SulfenicAcidAmide = "Sulfenic Acid Amide"
    Thiol = "Thiol"
    Alkylthiol = "Alkylthiol"
    Arylthiol = "Arylthiol"
    PhosphoricAcidDeriv = "Phosphoric Acid Derivative"
    PhosphoricAcid = "Phosphoric Acid"
    PhosphoricAcidEster = "Phosphoric Acid Ester"
    PhosphoricAcidHalide = "Phosphoric Acid Halide"
    PhosphoricAcidAmide = "Phosphoric Acid Amide"
    ThiophosphoricAcidDeriv = "Thiophosphoric Acid Derivative"
    ThiophosphoricAcid = "Thiophosphoric Acid"
    ThiophosphoricAcidEster = "Thiophosphoric Acid Ester"
    ThiophosphoricAcidHalide = "Thiophosphoric Acid Halide"
    ThiophosphoricAcidAmide = "Thiophosphoric Acid Amide"
    PhosphonicAcidDeriv = "Phosphonic Acid Derivative"
    PhosphonicAcid = "Phosphonic Acid"
    PhosphonicAcidEster = "Phosphonic Acid Ester"
    Phosphine = "Phosphine"
    Phosphinoxide = "Phosphinoxide"
    BoronicAcidDeriv = "Boronic Acid Derivative"
    BoronicAcid = "Boronic Acid"
    BoronicAcidEster = "Boronic Acid Ester"
    Alkene = "Alkene"
    Alkyne = "Alkyne"
    Aromatic = "Aromaticatic"
    Heterocycle = "Heterocycle"
    AlphaAminoacid = "Alpha Aminoacid"
    AlphaHydroxyacid = "Alpha Hydroxyacid"
    Aqueous = "Aqueous"


def checkmol_code_to_environment(checkmol_code) -> ChemicalEnvironment:

    checkmol_code_map = {
        "000": ChemicalEnvironment.Alkane,
        "001": ChemicalEnvironment.Cation,
        "002": ChemicalEnvironment.Anion,
        "003": ChemicalEnvironment.Carbonyl,
        "004": ChemicalEnvironment.Aldehyde,
        "005": ChemicalEnvironment.Ketone,
        "006": ChemicalEnvironment.Thiocarbonyl,
        "007": ChemicalEnvironment.Thioaldehyde,
        "008": ChemicalEnvironment.Thioketone,
        "009": ChemicalEnvironment.Imine,
        "010": ChemicalEnvironment.Hydrazone,
        "011": ChemicalEnvironment.Semicarbazone,
        "012": ChemicalEnvironment.Thiosemicarbazone,
        "013": ChemicalEnvironment.Oxime,
        "014": ChemicalEnvironment.OximeEther,
        "015": ChemicalEnvironment.Ketene,
        "016": ChemicalEnvironment.KeteneAcetalDeriv,
        "017": ChemicalEnvironment.CarbonylHydrate,
        "018": ChemicalEnvironment.Hemiacetal,
        "019": ChemicalEnvironment.Acetal,
        "020": ChemicalEnvironment.Hemiaminal,
        "021": ChemicalEnvironment.Aminal,
        "022": ChemicalEnvironment.Thiohemiaminal,
        "023": ChemicalEnvironment.Thioacetal,
        "024": ChemicalEnvironment.Enamine,
        "025": ChemicalEnvironment.Enol,
        "026": ChemicalEnvironment.Enolether,
        "027": ChemicalEnvironment.Hydroxy,
        "028": ChemicalEnvironment.Alcohol,
        "029": ChemicalEnvironment.PrimaryAlcohol,
        "030": ChemicalEnvironment.SecondaryAlcohol,
        "031": ChemicalEnvironment.TertiaryAlcohol,
        "032": ChemicalEnvironment.Diol_1_2,
        "033": ChemicalEnvironment.Aminoalcohol_1_2,
        "034": ChemicalEnvironment.Phenol,
        "035": ChemicalEnvironment.Diphenol_1_2,
        "036": ChemicalEnvironment.Enediol,
        "037": ChemicalEnvironment.Ether,
        "038": ChemicalEnvironment.Dialkylether,
        "039": ChemicalEnvironment.Alkylarylether,
        "040": ChemicalEnvironment.Diarylether,
        "041": ChemicalEnvironment.Thioether,
        "042": ChemicalEnvironment.Disulfide,
        "043": ChemicalEnvironment.Peroxide,
        "044": ChemicalEnvironment.Hydroperoxide,
        "045": ChemicalEnvironment.Hydrazine,
        "046": ChemicalEnvironment.Hydroxylamine,
        "047": ChemicalEnvironment.Amine,
        "048": ChemicalEnvironment.PrimaryAmine,
        "049": ChemicalEnvironment.PrimaryAliphAmine,
        "050": ChemicalEnvironment.PrimaryAromAmine,
        "051": ChemicalEnvironment.SecondaryAmine,
        "052": ChemicalEnvironment.SecondaryAliphAmine,
        "053": ChemicalEnvironment.SecondaryMixedAmine,
        "054": ChemicalEnvironment.SecondaryAromAmine,
        "055": ChemicalEnvironment.TertiaryAmine,
        "056": ChemicalEnvironment.TertiaryAliphAmine,
        "057": ChemicalEnvironment.TertiaryMixedAmine,
        "058": ChemicalEnvironment.TertiaryAromAmine,
        "059": ChemicalEnvironment.QuartAmmonium,
        "060": ChemicalEnvironment.NOxide,
        "061": ChemicalEnvironment.HalogenDeriv,
        "062": ChemicalEnvironment.AlkylHalide,
        "063": ChemicalEnvironment.AlkylFluoride,
        "064": ChemicalEnvironment.AlkylChloride,
        "065": ChemicalEnvironment.AlkylBromide,
        "066": ChemicalEnvironment.AlkylIodide,
        "067": ChemicalEnvironment.ArylHalide,
        "068": ChemicalEnvironment.ArylFluoride,
        "069": ChemicalEnvironment.ArylChloride,
        "070": ChemicalEnvironment.ArylBromide,
        "071": ChemicalEnvironment.ArylIodide,
        "072": ChemicalEnvironment.Organometallic,
        "073": ChemicalEnvironment.Organolithium,
        "074": ChemicalEnvironment.Organomagnesium,
        "075": ChemicalEnvironment.CarboxylicAcidDeriv,
        "076": ChemicalEnvironment.CarboxylicAcid,
        "077": ChemicalEnvironment.CarboxylicAcidSalt,
        "078": ChemicalEnvironment.CarboxylicAcidEster,
        "079": ChemicalEnvironment.Lactone,
        "080": ChemicalEnvironment.CarboxylicAcidAmide,
        "081": ChemicalEnvironment.CarboxylicAcidPrimaryAmide,
        "082": ChemicalEnvironment.CarboxylicAcidSecondaryAmide,
        "083": ChemicalEnvironment.CarboxylicAcidTertiaryAmide,
        "084": ChemicalEnvironment.Lactam,
        "085": ChemicalEnvironment.CarboxylicAcidHydrazide,
        "086": ChemicalEnvironment.CarboxylicAcidAzide,
        "087": ChemicalEnvironment.HydroxamicAcid,
        "088": ChemicalEnvironment.CarboxylicAcidAmidine,
        "089": ChemicalEnvironment.CarboxylicAcidAmidrazone,
        "090": ChemicalEnvironment.Nitrile,
        "091": ChemicalEnvironment.AcylHalide,
        "092": ChemicalEnvironment.AcylFluoride,
        "093": ChemicalEnvironment.AcylChloride,
        "094": ChemicalEnvironment.AcylBromide,
        "095": ChemicalEnvironment.AcylIodide,
        "096": ChemicalEnvironment.AcylCyanide,
        "097": ChemicalEnvironment.ImidoEster,
        "098": ChemicalEnvironment.ImidoylHalide,
        "099": ChemicalEnvironment.ThiocarboxylicAcidDeriv,
        "100": ChemicalEnvironment.ThiocarboxylicAcid,
        "101": ChemicalEnvironment.ThiocarboxylicAcidEster,
        "102": ChemicalEnvironment.Thiolactone,
        "103": ChemicalEnvironment.ThiocarboxylicAcidAmide,
        "104": ChemicalEnvironment.Thiolactam,
        "105": ChemicalEnvironment.ImidoThioester,
        "106": ChemicalEnvironment.Oxohetarene,
        "107": ChemicalEnvironment.Thioxohetarene,
        "108": ChemicalEnvironment.Iminohetarene,
        "109": ChemicalEnvironment.OrthocarboxylicAcidDeriv,
        "110": ChemicalEnvironment.CarboxylicAcidOrthoester,
        "111": ChemicalEnvironment.CarboxylicAcidAmideAcetal,
        "112": ChemicalEnvironment.CarboxylicAcidAnhydride,
        "113": ChemicalEnvironment.CarboxylicAcidImide,
        "114": ChemicalEnvironment.CarboxylicAcidUnsubstImide,
        "115": ChemicalEnvironment.CarboxylicAcidSubstImide,
        "116": ChemicalEnvironment.Co2Deriv,
        "117": ChemicalEnvironment.CarbonicAcidDeriv,
        "118": ChemicalEnvironment.CarbonicAcidMonoester,
        "119": ChemicalEnvironment.CarbonicAcidDiester,
        "120": ChemicalEnvironment.CarbonicAcidEsterHalide,
        "121": ChemicalEnvironment.ThiocarbonicAcidDeriv,
        "122": ChemicalEnvironment.ThiocarbonicAcidMonoester,
        "123": ChemicalEnvironment.ThiocarbonicAcidDiester,
        "124": ChemicalEnvironment.ThiocarbonicAcidEsterHalide,
        "125": ChemicalEnvironment.CarbamicAcidDeriv,
        "126": ChemicalEnvironment.CarbamicAcid,
        "127": ChemicalEnvironment.CarbamicAcidEster,
        "128": ChemicalEnvironment.CarbamicAcidHalide,
        "129": ChemicalEnvironment.ThiocarbamicAcidDeriv,
        "130": ChemicalEnvironment.ThiocarbamicAcid,
        "131": ChemicalEnvironment.ThiocarbamicAcidEster,
        "132": ChemicalEnvironment.ThiocarbamicAcidHalide,
        "133": ChemicalEnvironment.Urea,
        "134": ChemicalEnvironment.Isourea,
        "135": ChemicalEnvironment.Thiourea,
        "136": ChemicalEnvironment.Isothiourea,
        "137": ChemicalEnvironment.Guanidine,
        "138": ChemicalEnvironment.Semicarbazide,
        "139": ChemicalEnvironment.Thiosemicarbazide,
        "140": ChemicalEnvironment.Azide,
        "141": ChemicalEnvironment.AzoCompound,
        "142": ChemicalEnvironment.DiazoniumSalt,
        "143": ChemicalEnvironment.Isonitrile,
        "144": ChemicalEnvironment.Cyanate,
        "145": ChemicalEnvironment.Isocyanate,
        "146": ChemicalEnvironment.Thiocyanate,
        "147": ChemicalEnvironment.Isothiocyanate,
        "148": ChemicalEnvironment.Carbodiimide,
        "149": ChemicalEnvironment.NitrosoCompound,
        "150": ChemicalEnvironment.NitroCompound,
        "151": ChemicalEnvironment.Nitrite,
        "152": ChemicalEnvironment.Nitrate,
        "153": ChemicalEnvironment.SulfuricAcidDeriv,
        "154": ChemicalEnvironment.SulfuricAcid,
        "155": ChemicalEnvironment.SulfuricAcidMonoester,
        "156": ChemicalEnvironment.SulfuricAcidDiester,
        "157": ChemicalEnvironment.SulfuricAcidAmideEster,
        "158": ChemicalEnvironment.SulfuricAcidAmide,
        "159": ChemicalEnvironment.SulfuricAcidDiamide,
        "160": ChemicalEnvironment.SulfurylHalide,
        "161": ChemicalEnvironment.SulfonicAcidDeriv,
        "162": ChemicalEnvironment.SulfonicAcid,
        "163": ChemicalEnvironment.SulfonicAcidEster,
        "164": ChemicalEnvironment.Sulfonamide,
        "165": ChemicalEnvironment.SulfonylHalide,
        "166": ChemicalEnvironment.Sulfone,
        "167": ChemicalEnvironment.Sulfoxide,
        "168": ChemicalEnvironment.SulfinicAcidDeriv,
        "169": ChemicalEnvironment.SulfinicAcid,
        "170": ChemicalEnvironment.SulfinicAcidEster,
        "171": ChemicalEnvironment.SulfinicAcidHalide,
        "172": ChemicalEnvironment.SulfinicAcidAmide,
        "173": ChemicalEnvironment.SulfenicAcidDeriv,
        "174": ChemicalEnvironment.SulfenicAcid,
        "175": ChemicalEnvironment.SulfenicAcidEster,
        "176": ChemicalEnvironment.SulfenicAcidHalide,
        "177": ChemicalEnvironment.SulfenicAcidAmide,
        "178": ChemicalEnvironment.Thiol,
        "179": ChemicalEnvironment.Alkylthiol,
        "180": ChemicalEnvironment.Arylthiol,
        "181": ChemicalEnvironment.PhosphoricAcidDeriv,
        "182": ChemicalEnvironment.PhosphoricAcid,
        "183": ChemicalEnvironment.PhosphoricAcidEster,
        "184": ChemicalEnvironment.PhosphoricAcidHalide,
        "185": ChemicalEnvironment.PhosphoricAcidAmide,
        "186": ChemicalEnvironment.ThiophosphoricAcidDeriv,
        "187": ChemicalEnvironment.ThiophosphoricAcid,
        "188": ChemicalEnvironment.ThiophosphoricAcidEster,
        "189": ChemicalEnvironment.ThiophosphoricAcidHalide,
        "190": ChemicalEnvironment.ThiophosphoricAcidAmide,
        "191": ChemicalEnvironment.PhosphonicAcidDeriv,
        "192": ChemicalEnvironment.PhosphonicAcid,
        "193": ChemicalEnvironment.PhosphonicAcidEster,
        "194": ChemicalEnvironment.Phosphine,
        "195": ChemicalEnvironment.Phosphinoxide,
        "196": ChemicalEnvironment.BoronicAcidDeriv,
        "197": ChemicalEnvironment.BoronicAcid,
        "198": ChemicalEnvironment.BoronicAcidEster,
        "199": ChemicalEnvironment.Alkene,
        "200": ChemicalEnvironment.Alkyne,
        "201": ChemicalEnvironment.Aromatic,
        "202": ChemicalEnvironment.Heterocycle,
        "203": ChemicalEnvironment.AlphaAminoacid,
        "204": ChemicalEnvironment.AlphaHydroxyacid,
    }
    return checkmol_code_map[checkmol_code]


@functools.lru_cache(1000)
def analyse_functional_groups(smiles):
    """Employs checkmol to determine which chemical moieties
    are encoded by a given smiles pattern.

    Notes
    -----
    See https://homepage.univie.ac.at/norbert.haider/cheminf/fgtable.pdf
    for information about the group numbers (i.e moiety types).

    Parameters
    ----------
    smiles: str
        The smiles pattern to examine.

    Returns
    -------
    dict of ChemicalEnvironment and int, optional
        A dictionary where each key corresponds to the `checkmol` defined group
        number, and each value if the number of instances of that moiety. If
        `checkmol` did not execute correctly, returns None.
    """
    import shutil
    import subprocess
    import tempfile

    from openff.toolkit.topology import Molecule

    if smiles == "O" or smiles == "[H]O[H]":
        return {ChemicalEnvironment.Aqueous: 1}
    if smiles == "N":
        return {ChemicalEnvironment.Amine: 1}

    # Make sure the checkmol utility has been installed separately.
    if shutil.which("checkmol") is None:

        raise MissingOptionalDependency(
            "checkmol",
            False,
            "Checkmol can be obtianed for free from "
            "http://merian.pch.univie.ac.at/~nhaider/cheminf/cmmm.html.",
        )

    openff_molecule: Molecule = Molecule.from_smiles(
        smiles, allow_undefined_stereo=True
    )

    # Save the smile pattern out as an SDF file, ready to use as input to checkmol.
    with tempfile.NamedTemporaryFile(suffix=".sdf") as file:

        openff_molecule.to_file(file.name, "SDF")

        # Execute checkmol.
        try:

            result = subprocess.check_output(
                ["checkmol", "-p", file.name],
                stderr=subprocess.STDOUT,
            ).decode()

        except subprocess.CalledProcessError:
            result = None

    if result is None:
        return None
    elif len(result) == 0:
        return {ChemicalEnvironment.Alkane: 1}

    groups = {}

    for group in result.splitlines():

        group_code, group_count, _ = group.split(":")

        group_environment = checkmol_code_to_environment(group_code[1:])
        groups[group_environment] = int(group_count)

    return groups
