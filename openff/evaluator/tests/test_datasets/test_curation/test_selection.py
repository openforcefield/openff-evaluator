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
