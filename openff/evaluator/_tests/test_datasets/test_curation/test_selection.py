from typing import Tuple

import numpy
import pandas
import pytest
from openff.units import unit

from openff.evaluator.datasets import (
    MeasurementSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.datasets.curation.components.selection import (
    FingerPrintType,
    SelectDataPoints,
    SelectDataPointsSchema,
    SelectNumRepresentation,
    SelectNumRepresentationSchema,
    SelectStratifiedSplit,
    SelectStratifiedSplitSchema,
    SelectSubstances,
    SelectSubstancesSchema,
    State,
    TargetState,
)
from openff.evaluator.properties import Density, EnthalpyOfVaporization
from openff.evaluator.substances import Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.checkmol import ChemicalEnvironment
from openff.evaluator.utils.exceptions import MissingOptionalDependency

try:
    import openeye.oechem
except ImportError:
    openeye = None


@pytest.fixture(scope="module")
def data_frame() -> pandas.DataFrame:
    temperatures = [303.15, 298.15]
    property_types = [Density, EnthalpyOfVaporization]

    data_set_entries = []

    def _temperature_noise():
        return (numpy.random.rand() / 2.0 + 0.51) / 10.0

    for temperature in temperatures:
        for index, property_type in enumerate(property_types):
            noise = _temperature_noise()
            noise *= 1 if index == 0 else -1

            data_set_entries.append(
                property_type(
                    thermodynamic_state=ThermodynamicState(
                        temperature=temperature * unit.kelvin,
                        pressure=101.325 * unit.kilopascal,
                    ),
                    phase=PropertyPhase.Liquid,
                    value=1.0 * property_type.default_unit(),
                    uncertainty=1.0 * property_type.default_unit(),
                    source=MeasurementSource(doi=" "),
                    substance=Substance.from_components("C"),
                ),
            )
            data_set_entries.append(
                property_type(
                    thermodynamic_state=ThermodynamicState(
                        temperature=(temperature + noise) * unit.kelvin,
                        pressure=101.325 * unit.kilopascal,
                    ),
                    phase=PropertyPhase.Liquid,
                    value=1.0 * property_type.default_unit(),
                    uncertainty=1.0 * property_type.default_unit(),
                    source=MeasurementSource(doi=" "),
                    substance=Substance.from_components("C"),
                ),
            )

    data_set = PhysicalPropertyDataSet()
    data_set.add_properties(*data_set_entries)

    data_frame = data_set.to_pandas()
    return data_frame


@pytest.mark.parametrize(
    "target_temperatures, expected_temperatures",
    [([300.0], [298.15]), ([301.0], [303.15]), ([300.0, 301.0], [298.15, 303.15])],
)
def test_select_data_points(target_temperatures, expected_temperatures, data_frame):
    """Tests that data points are selected in a reasonably optimal way."""

    states = [
        State(temperature=target_temperature, pressure=101.325, mole_fractions=(1.0,))
        for target_temperature in target_temperatures
    ]

    # Define target states for ambient conditions
    schema = SelectDataPointsSchema(
        target_states=[
            TargetState(
                property_types=[("Density", 1), ("EnthalpyOfVaporization", 1)],
                states=states,
            )
        ]
    )

    selected_data = SelectDataPoints.apply(data_frame, schema)

    assert len(selected_data) == len(expected_temperatures) * 2
    assert len(selected_data["Temperature (K)"].unique()) == len(expected_temperatures)

    selected_temperatures = sorted(selected_data["Temperature (K)"].unique())
    expected_temperatures = sorted(expected_temperatures)

    assert numpy.allclose(selected_temperatures, expected_temperatures)


def test_select_data_points_duplicate():
    """Tests that the select data points component will only select one data
    point per property type and per target state, even when there are duplicate
    data points."""

    data_rows = [
        {
            "N Components": 1,
            "Temperature (K)": 298.15004,
            "Pressure (kPa)": 101.325,
            "Component 1": "C",
            "Mole Fraction 1": 1.0,
            "Density Value (g / ml)": 1.0,
        },
        {
            "N Components": 1,
            "Temperature (K)": 298.15002,
            "Pressure (kPa)": 101.325,
            "Component 1": "C",
            "Mole Fraction 1": 1.0,
            "Density Value (g / ml)": 2.0,
        },
        {
            "N Components": 1,
            "Phase": "Liquid",
            "Temperature (K)": 298.15002,
            "Pressure (kPa)": 101.325,
            "Component 1": "CCCCCCC(C)O",
            "Mole Fraction 1": 1.0,
            "EnthalpyOfVaporization Value (kJ / mol)": 2.0,
        },
        {
            "N Components": 1,
            "Phase": "Liquid",
            "Temperature (K)": 298.15004,
            "Pressure (kPa)": 101.325,
            "Component 1": "CCCCCCC(C)O",
            "Mole Fraction 1": 1.0,
            "EnthalpyOfVaporization Value (kJ / mol)": 1.0,
        },
    ]
    data_frame = pandas.DataFrame(data_rows)

    states = [State(temperature=298.15, pressure=101.325, mole_fractions=(1.0,))]

    # Define target states for ambient conditions
    schema = SelectDataPointsSchema(
        target_states=[
            TargetState(
                property_types=[("Density", 1), ("EnthalpyOfVaporization", 1)],
                states=states,
            )
        ]
    )

    selected_data = SelectDataPoints.apply(data_frame, schema)

    assert len(selected_data) == 2

    assert len(selected_data["Temperature (K)"].unique()) == 1
    assert 298.15001 < selected_data["Temperature (K)"].unique()[0] < 298.15003

    density_header = "Density Value (g / ml)"
    enthalpy_header = "EnthalpyOfVaporization Value (kJ / mol)"

    density_data = selected_data[selected_data[density_header].notna()]
    assert len(density_data[density_header].unique()) == 1
    assert numpy.isclose(density_data[density_header].unique(), 2.0)

    enthalpy_data = selected_data[selected_data[enthalpy_header].notna()]
    assert len(enthalpy_data[enthalpy_header].unique()) == 1
    assert numpy.isclose(enthalpy_data[enthalpy_header].unique(), 2.0)


@pytest.mark.skipif(
    openeye is None or not openeye.oechem.OEChemIsLicensed(),
    reason="OpenEye is required for this test.",
)
def test_mixture_distance_metric():
    """Tests that the distance metric between mixtures behaves as
    expected."""

    assert (
        SelectSubstances._compute_distance(
            ("CCCCC", "CCCC=O"), ("CCCC=O", "CCCCC"), FingerPrintType.Tree
        )
        == 0.0
    )
    assert (
        SelectSubstances._compute_distance(
            ("CCCCC", "CCCC=O"), ("C#N", "CCCCC"), FingerPrintType.Tree
        )
        == 1.0
    )


@pytest.mark.parametrize(
    "mixture_a", [("Oc1occc1", "c1ccncc1"), ("c1ccncc1", "Oc1occc1")]
)
@pytest.mark.parametrize("mixture_b", [("CCCC=O", "CC(C)C#N"), ("CC(C)C#N", "CCCC=O")])
@pytest.mark.skipif(
    openeye is None or not openeye.oechem.OEChemIsLicensed(),
    reason="OpenEye is required for this test.",
)
def test_mixture_distance_metric_symmetry(
    mixture_a: Tuple[str, str], mixture_b: Tuple[str, str]
):
    """Tests that the distance metric between mixtures behaves as
    expected and symmetrically under permutation"""

    distance_a_b = SelectSubstances._compute_distance(
        mixture_a, mixture_b, FingerPrintType.Tree
    )
    distance_b_a = SelectSubstances._compute_distance(
        mixture_b, mixture_a, FingerPrintType.Tree
    )

    assert numpy.isclose(distance_a_b, distance_b_a)


@pytest.mark.skipif(
    openeye is None or not openeye.oechem.OEChemIsLicensed(),
    reason="OpenEye is required for this test.",
)
def test_select_substances():
    training_substances = [("CCCCC",), ("CCCCC", "CCCCCO")]

    data_rows = [
        {"N Components": 1, "Component 1": "CCCCC"},
        {"N Components": 1, "Component 1": "CCCCCC"},
        {"N Components": 1, "Component 1": "CCC(C)C"},
        {"N Components": 2, "Component 1": "CCCCC", "Component 2": "CCCCCO"},
        {"N Components": 2, "Component 1": "CCCCCC", "Component 2": "CCCCCCO"},
        {"N Components": 2, "Component 1": "CCC(C)C", "Component 2": "CCCCCO"},
    ]

    data_frame = pandas.DataFrame(data_rows)

    schema = SelectSubstancesSchema(
        target_environments=[ChemicalEnvironment.Alkane, ChemicalEnvironment.Alcohol],
        n_per_environment=1,
        substances_to_exclude=training_substances,
        per_property=False,
    )

    selected_data_frame = SelectSubstances.apply(data_frame, schema, 1)

    assert len(selected_data_frame) == 2
    assert selected_data_frame["Component 1"].iloc[0] == "CCC(C)C"
    assert numpy.isnan(selected_data_frame["Component 2"].iloc[0])
    assert selected_data_frame["Component 1"].iloc[1] == "CCC(C)C"
    assert selected_data_frame["Component 2"].iloc[1] == "CCCCCO"


@pytest.mark.skipif(
    openeye is None or not openeye.oechem.OEChemIsLicensed(),
    reason="OpenEye is required for this test.",
)
def test_select_substances_per_property():
    """Tests that the SelectSubstances works as expected with both
    the `per_property` option enabled and disabled."""

    training_substances = [("CC(O)C",)]

    data_rows = [
        {"N Components": 1, "Component 1": "CCC(O)CC", "Density Value (g / ml)": 1.0},
        {"N Components": 1, "Component 1": "C(O)CCC(O)", "Density Value (g / ml)": 1.0},
        {"N Components": 1, "Component 1": "CCC(O)CC", "Enthalpy Value (g / ml)": 1.0},
    ]

    data_frame = pandas.DataFrame(data_rows)

    schema = SelectSubstancesSchema(
        target_environments=[ChemicalEnvironment.Alcohol],
        n_per_environment=1,
        substances_to_exclude=training_substances,
        per_property=False,
    )

    selected_data_frame = SelectSubstances.apply(data_frame, schema, 1)

    assert len(selected_data_frame) == 1
    assert selected_data_frame["Component 1"].iloc[0] == "C(O)CCC(O)"

    assert "Density Value (g / ml)" in selected_data_frame
    assert (
        len(selected_data_frame[selected_data_frame["Density Value (g / ml)"].notna()])
        == 1
    )
    assert (
        len(selected_data_frame[selected_data_frame["Enthalpy Value (g / ml)"].notna()])
        == 0
    )

    schema = SelectSubstancesSchema(
        target_environments=[ChemicalEnvironment.Alcohol],
        n_per_environment=1,
        substances_to_exclude=training_substances,
        per_property=True,
    )

    selected_data_frame = SelectSubstances.apply(data_frame, schema, 1)

    assert len(selected_data_frame) == 2
    assert all(selected_data_frame["Component 1"].isin(["C(O)CCC(O)", "CCC(O)CC"]))

    assert (
        len(selected_data_frame[selected_data_frame["Density Value (g / ml)"].notna()])
        == 1
    )
    assert (
        len(selected_data_frame[selected_data_frame["Enthalpy Value (g / ml)"].notna()])
        == 1
    )


def test_select_num_representation():
    data_rows = [
        {"N Components": 1, "Component 1": "C"},
        {"N Components": 1, "Component 1": "CCO"},
        {"N Components": 1, "Component 1": "CCN"},
        {"N Components": 1, "Component 1": "CCN"},
        {"N Components": 1, "Component 1": "CCN"},
        {"N Components": 2, "Component 1": "C", "Component 2": "CCO"},
        {"N Components": 2, "Component 1": "C", "Component 2": "CCN"},
        {"N Components": 2, "Component 1": "C", "Component 2": "CCN"},
        {"N Components": 2, "Component 1": "C", "Component 2": "CCF"},
        {"N Components": 2, "Component 1": "C", "Component 2": "CCF"},
        {"N Components": 2, "Component 1": "C", "Component 2": "CCF"},
    ]
    data_frame = pandas.DataFrame(data_rows)

    # test minimum and maximum for mixtures
    schema = SelectNumRepresentationSchema(
        minimum_representation=2, maximum_representation=2, per_component=False
    )
    selected_data_frame = SelectNumRepresentation.apply(data_frame, schema, 1)
    assert len(selected_data_frame) == 2
    assert all(selected_data_frame["N Components"] == 2)
    assert all(selected_data_frame["Component 2"] == "CCN")
    assert all(selected_data_frame["Component 1"] == "C")

    # test no maximum
    schema = SelectNumRepresentationSchema(
        minimum_representation=2, per_component=False
    )
    selected_data_frame = SelectNumRepresentation.apply(data_frame, schema, 1)
    assert len(selected_data_frame) == 8
    assert selected_data_frame["Component 1"].values.tolist() == ["CCN"] * 3 + ["C"] * 5

    component_2_values = selected_data_frame["Component 2"].values.tolist()
    expected_values = [pandas.NA] * 3 + ["CCN"] * 2 + ["CCF"] * 3
    assert len(component_2_values) == len(expected_values)
    for actual, expected in zip(component_2_values, expected_values):
        if pandas.isna(expected):
            assert pandas.isna(actual)
        else:
            assert actual == expected

    # test per_component
    schema = SelectNumRepresentationSchema(minimum_representation=3, per_component=True)
    selected_data_frame = SelectNumRepresentation.apply(data_frame, schema, 1)
    assert len(selected_data_frame) == 9
    assert not any(selected_data_frame["Component 1"] == "CCO")
    assert not any(selected_data_frame["Component 2"] == "CCO")

    schema = SelectNumRepresentationSchema(maximum_representation=2, per_component=True)
    selected_data_frame = SelectNumRepresentation.apply(data_frame, schema, 1)
    assert len(selected_data_frame) == 1
    assert selected_data_frame["Component 1"].iloc[0] == "CCO"
    assert selected_data_frame["N Components"].iloc[0] == 1


def _df(smiles_list):
    """Minimal one-row-per-substance DataFrame."""
    return pandas.DataFrame(
        [{"N Components": 1, "Component 1": smi} for smi in smiles_list]
    )


def test_select_stratified_split_fraction():
    df = _df(["CC", "CCC", "CCCC", "CCO", "CCOCC", "CC(C)O"])
    result = SelectStratifiedSplit.apply(df, SelectStratifiedSplitSchema(keep_fraction=0.5, seed=0))
    assert len(result["Component 1"].unique()) == 3


def test_select_stratified_split_deterministic():
    df = _df(["CC", "CCC", "CCCC", "CCO", "CCOCC", "CC(C)O"])
    schema = SelectStratifiedSplitSchema(keep_fraction=0.5, seed=42)
    a = SelectStratifiedSplit.apply(df, schema)
    b = SelectStratifiedSplit.apply(df, schema)
    assert set(a["Component 1"].unique()) == set(b["Component 1"].unique())


def test_select_stratified_split_smiles_strata():
    # Global budget: round(5 * 0.5) = 2.
    # "O" has a higher priority score than the rest and should always be selected.
    df = _df(["CCCC", "CCCCC", "CCCO", "CC(O)C", "O"])
    result = SelectStratifiedSplit.apply(
        df, SelectStratifiedSplitSchema(keep_fraction=0.5, seed=0, smiles_strata=["O"])
    )
    result_smiles = set(result["Component 1"].unique())
    assert len(result_smiles) == 2
    assert "O" in result_smiles


def test_select_stratified_split_custom_smiles_weight_priority():
    # All three substances match exactly one smiles stratum and are otherwise
    # equivalent. Custom weight should prioritize water.
    df = _df(["O", "CC", "CCC"])
    result = SelectStratifiedSplit.apply(
        df,
        SelectStratifiedSplitSchema(
            keep_fraction=1 / 3,
            seed=0,
            diversity_selection=False,
            smiles_strata=["O", "CC", "CCC"],
            strata_weights={"O": 2.0},
        ),
    )
    result_smiles = set(result["Component 1"].unique())
    assert result_smiles == {"O"}


def test_select_stratified_split_smiles_strata_canonicalized_for_comparison():
    # "CCO" in schema must match "OCC" in data; DataFrame not mutated.
    df = _df(["OCC", "CCCC", "CCCCC", "CCCCCC"])
    result = SelectStratifiedSplit.apply(
        df, SelectStratifiedSplitSchema(keep_fraction=0.5, seed=0, smiles_strata=["CCO"])
    )
    assert "OCC" in set(result["Component 1"].unique())
    assert df["Component 1"].tolist() == ["OCC", "CCCC", "CCCCC", "CCCCCC"]


def test_select_stratified_split_target_environment_strata():
    # Global budget: round(8 * 0.5) = 4.
    # NCCO / NCCCO match both requested environments and should be prioritized.
    df = _df(["NCCO", "NCCCO", "CCO", "CCCO", "CCN", "CCCN", "CC", "CCC"])
    result = SelectStratifiedSplit.apply(
        df,
        SelectStratifiedSplitSchema(
            keep_fraction=0.5,
            seed=0,
            target_environment_strata=[ChemicalEnvironment.Alcohol, ChemicalEnvironment.Amine],
        ),
    )
    result_smiles = set(result["Component 1"].unique())
    assert len(result_smiles) == 4
    assert {"NCCO", "NCCCO"}.issubset(result_smiles)


def test_select_stratified_split_property_type_strata():
    # Global budget: round(4 * 0.5) = 2.
    # Density-covered substances should be preferred.
    df = pandas.DataFrame([
        {"N Components": 1, "Component 1": "CCCC",    "Density Value (g / ml)": 0.7},
        {"N Components": 1, "Component 1": "CCCCC",   "Density Value (g / ml)": 0.7},
        {"N Components": 1, "Component 1": "CCCCCC",  "Density Value (g / ml)": None},
        {"N Components": 1, "Component 1": "CCCCCCC", "Density Value (g / ml)": None},
    ])
    result = SelectStratifiedSplit.apply(
        df, SelectStratifiedSplitSchema(keep_fraction=0.5, seed=0, property_type_strata=["Density"])
    )
    result_smiles = set(result["Component 1"].unique())
    assert len(result_smiles) == 2
    assert result_smiles == {"CCCC", "CCCCC"}


def test_select_stratified_split_combined_strata():
    # Global budget: round(6 * 0.5) = 3.
    # O and CCO have the highest combined scores and should be selected,
    # with the final slot taken from the next priority tier.
    df = pandas.DataFrame([
        {"N Components": 1, "Component 1": "O",    "Density Value (g / ml)": 1.0},
        {"N Components": 1, "Component 1": "CCO",  "Density Value (g / ml)": 0.7},
        {"N Components": 1, "Component 1": "CCCO", "Density Value (g / ml)": None},
        {"N Components": 1, "Component 1": "CC",   "Density Value (g / ml)": None},
        {"N Components": 1, "Component 1": "CCC",  "Density Value (g / ml)": None},
        {"N Components": 1, "Component 1": "CCCC", "Density Value (g / ml)": None},
    ])
    result = SelectStratifiedSplit.apply(
        df,
        SelectStratifiedSplitSchema(
            keep_fraction=0.5,
            seed=0,
            smiles_strata=["O"],
            target_environment_strata=[ChemicalEnvironment.Alcohol],
            property_type_strata=["Density"],
        ),
    )
    result_smiles = set(result["Component 1"].unique())
    assert len(result_smiles) == 3
    assert {"O", "CCO", "CCCO"}.issubset(result_smiles)


def test_select_stratified_split_priority_tiers_smiles_and_properties():
    # Global budget: round(6 * 0.5) = 3.
    # Priority score is the sum of matched smiles_strata and property_type_strata flags.
    # The two mixtures that cover both properties should be selected first.
    df = pandas.DataFrame(
        [
            {
                "N Components": 2,
                "Component 1": "CO",
                "Component 2": "O",
                "Density Value (g / ml)": 0.99,
                "EnthalpyOfMixing Value (kJ / mol)": -1.1,
            },
            {
                "N Components": 2,
                "Component 1": "CCO",
                "Component 2": "O",
                "Density Value (g / ml)": 0.95,
                "EnthalpyOfMixing Value (kJ / mol)": -0.8,
            },
            {
                "N Components": 2,
                "Component 1": "CCO",
                "Component 2": "CO",
                "Density Value (g / ml)": 0.90,
                "EnthalpyOfMixing Value (kJ / mol)": None,
            },
            {
                "N Components": 2,
                "Component 1": "O",
                "Component 2": "CCN",
                "Density Value (g / ml)": 0.98,
                "EnthalpyOfMixing Value (kJ / mol)": None,
            },
            {
                "N Components": 2,
                "Component 1": "CO",
                "Component 2": "CCN",
                "Density Value (g / ml)": None,
                "EnthalpyOfMixing Value (kJ / mol)": -0.3,
            },
            {
                "N Components": 2,
                "Component 1": "CC",
                "Component 2": "CCC",
                "Density Value (g / ml)": 0.72,
                "EnthalpyOfMixing Value (kJ / mol)": None,
            },
        ]
    )

    result = SelectStratifiedSplit.apply(
        df,
        SelectStratifiedSplitSchema(
            keep_fraction=0.5,
            seed=0,
            smiles_strata=["O", "CO", "CCO"],
            property_type_strata=["Density", "EnthalpyOfMixing"],
            diversity_selection=False,
            property_balance=False,
        ),
    )

    selected_substances = set(
        tuple(sorted([row["Component 1"], row["Component 2"]]))
        for _, row in result[["Component 1", "Component 2"]].drop_duplicates().iterrows()
    )

    assert len(selected_substances) == 3
    assert tuple(sorted(["O", "CO"])) in selected_substances
    assert tuple(sorted(["O", "CCO"])) in selected_substances


def test_select_stratified_split_diversity():
    df = _df(["CCCC", "CCCCC", "CCCCCC", "CCO", "CCOCC", "CC(C)O"])
    result = SelectStratifiedSplit.apply(
        df, SelectStratifiedSplitSchema(keep_fraction=0.5, seed=0, diversity_selection=True)
    )
    assert len(result["Component 1"].unique()) == 3


def test_select_stratified_split_include_subset_substances():
    # round(4 * 0.25) = 1 selected substance. The top-priority substance is a binary
    # with both requested property types; include_subset_substances should also retain
    # pure-component rows whose substance keys are subsets of that binary.
    df = pandas.DataFrame(
        [
            {
                "N Components": 2,
                "Component 1": "O",
                "Component 2": "CCO",
                "Density Value (g / ml)": 0.95,
                "EnthalpyOfMixing Value (kJ / mol)": -0.7,
            },
            {
                "N Components": 1,
                "Component 1": "O",
                "Density Value (g / ml)": 1.0,
                "EnthalpyOfMixing Value (kJ / mol)": None,
            },
            {
                "N Components": 1,
                "Component 1": "CCO",
                "Density Value (g / ml)": 0.79,
                "EnthalpyOfMixing Value (kJ / mol)": None,
            },
            {
                "N Components": 1,
                "Component 1": "CCCC",
                "Density Value (g / ml)": 0.71,
                "EnthalpyOfMixing Value (kJ / mol)": None,
            },
        ]
    )

    base = SelectStratifiedSplit.apply(
        df,
        SelectStratifiedSplitSchema(
            keep_fraction=0.25,
            seed=0,
            property_type_strata=["Density", "EnthalpyOfMixing"],
            diversity_selection=False,
            property_balance=False,
            include_subset_substances=False,
        ),
    )
    assert len(base) == 1
    base_key = tuple(
        sorted(
            base.loc[base.index[0], f"Component {j + 1}"]
            for j in range(int(base.loc[base.index[0], "N Components"]))
        )
    )

    with_subsets = SelectStratifiedSplit.apply(
        df,
        SelectStratifiedSplitSchema(
            keep_fraction=0.25,
            seed=0,
            property_type_strata=["Density", "EnthalpyOfMixing"],
            diversity_selection=False,
            property_balance=False,
            include_subset_substances=True,
        ),
    )

    selected_keys = {
        tuple(
            sorted(
                with_subsets.loc[i, f"Component {j + 1}"]
                for j in range(int(with_subsets.loc[i, "N Components"]))
            )
        )
        for i in with_subsets.index
    }

    # include_subset_substances should never drop the initially selected key.
    assert base_key in selected_keys

    # If the selected key is the binary, include its pure subsets.
    if base_key == ("CCO", "O"):
        assert ("O",) in selected_keys
        assert ("CCO",) in selected_keys

    assert ("CCCC",) not in selected_keys


def test_select_stratified_split_property_balance_row_mode_after_refill():
    # 6 unique substances -> round(6 * 0.83) = 5 selected substances.
    # Substance-level counts are initially balanced (3 vs 3), but row-level
    # counts are heavily skewed toward density due to repeated pure-density rows.
    rows = [
        {
            "N Components": 1,
            "Component 1": "O",
            "Density Value (g / ml)": 1.0,
            "EnthalpyOfMixing Value (kJ / mol)": None,
        },
        {
            "N Components": 1,
            "Component 1": "O",
            "Density Value (g / ml)": None,
            "EnthalpyOfMixing Value (kJ / mol)": -0.1,
        },
    ]

    rows.extend(
        {
            "N Components": 1,
            "Component 1": "CC",
            "Density Value (g / ml)": 0.7,
            "EnthalpyOfMixing Value (kJ / mol)": None,
        }
        for _ in range(8)
    )
    rows.extend(
        {
            "N Components": 1,
            "Component 1": "CCC",
            "Density Value (g / ml)": 0.72,
            "EnthalpyOfMixing Value (kJ / mol)": None,
        }
        for _ in range(8)
    )
    rows.append(
        {
            "N Components": 1,
            "Component 1": "N",
            "Density Value (g / ml)": None,
            "EnthalpyOfMixing Value (kJ / mol)": -0.2,
        }
    )
    rows.append(
        {
            "N Components": 1,
            "Component 1": "CO",
            "Density Value (g / ml)": None,
            "EnthalpyOfMixing Value (kJ / mol)": -0.3,
        }
    )
    rows.append(
        {
            "N Components": 1,
            "Component 1": "CCO",
            "Density Value (g / ml)": None,
            "EnthalpyOfMixing Value (kJ / mol)": None,
        }
    )

    df = pandas.DataFrame(rows)

    substance_mode = SelectStratifiedSplit.apply(
        df,
        SelectStratifiedSplitSchema(
            keep_fraction=0.83,
            seed=0,
            property_type_strata=["Density", "EnthalpyOfMixing"],
            diversity_selection=False,
            property_balance=True,
            property_balance_mode="substance",
            max_property_ratio=1.2,
        ),
    )
    substance_density_rows = int(substance_mode["Density Value (g / ml)"].notna().sum())
    substance_hmix_rows = int(
        substance_mode["EnthalpyOfMixing Value (kJ / mol)"].notna().sum()
    )
    assert substance_density_rows > int(1.2 * substance_hmix_rows)

    row_mode = SelectStratifiedSplit.apply(
        df,
        SelectStratifiedSplitSchema(
            keep_fraction=0.83,
            seed=0,
            property_type_strata=["Density", "EnthalpyOfMixing"],
            diversity_selection=False,
            property_balance=True,
            property_balance_mode="row",
            max_property_ratio=1.2,
        ),
    )

    row_density_rows = int(row_mode["Density Value (g / ml)"].notna().sum())
    row_hmix_rows = int(row_mode["EnthalpyOfMixing Value (kJ / mol)"].notna().sum())

    assert row_hmix_rows > 0
    assert row_density_rows <= int(1.2 * row_hmix_rows)


def test_select_stratified_split_property_balance_by_n_components():
    # Pure bucket is balanced (1 density + 1 hmix), binary bucket is density-heavy.
    # With per-n_components balancing on, binary trimming should not remove pure rows.
    df = pandas.DataFrame(
        [
            {
                "N Components": 1,
                "Component 1": "O",
                "Density Value (g / ml)": 1.0,
                "EnthalpyOfMixing Value (kJ / mol)": None,
            },
            {
                "N Components": 1,
                "Component 1": "N",
                "Density Value (g / ml)": None,
                "EnthalpyOfMixing Value (kJ / mol)": -0.2,
            },
            {
                "N Components": 2,
                "Component 1": "O",
                "Component 2": "CC",
                "Density Value (g / ml)": 0.92,
                "EnthalpyOfMixing Value (kJ / mol)": None,
            },
            {
                "N Components": 2,
                "Component 1": "O",
                "Component 2": "CCC",
                "Density Value (g / ml)": 0.90,
                "EnthalpyOfMixing Value (kJ / mol)": None,
            },
            {
                "N Components": 2,
                "Component 1": "N",
                "Component 2": "CC",
                "Density Value (g / ml)": None,
                "EnthalpyOfMixing Value (kJ / mol)": -0.3,
            },
        ]
    )

    result = SelectStratifiedSplit.apply(
        df,
        SelectStratifiedSplitSchema(
            keep_fraction=1.0,
            seed=0,
            property_type_strata=["Density", "EnthalpyOfMixing"],
            diversity_selection=False,
            property_balance=True,
            property_balance_mode="substance",
            property_balance_by_n_components=True,
            max_property_ratio=1.2,
        ),
    )

    selected_keys = {
        tuple(
            sorted(
                result.loc[i, f"Component {j + 1}"]
                for j in range(int(result.loc[i, "N Components"]))
            )
        )
        for i in result.index
    }

    # Pure rows should remain because the pure bucket is already balanced.
    assert ("O",) in selected_keys
    assert ("N",) in selected_keys


@pytest.mark.skipif(
    openeye is not None and openeye.oechem.OEChemIsLicensed(),
    reason="OpenEye should not be installed for this test.",
)
def test_missing_openeye_dependency():
    data_frame = pandas.DataFrame([{"N Components": 1, "Component 1": "CCCCC"}])

    with pytest.raises(MissingOptionalDependency):
        SelectSubstances.apply(
            data_frame,
            SelectSubstancesSchema(
                target_environments=[ChemicalEnvironment.Alkane],
                n_per_environment=1,
                per_property=False,
            ),
        )
