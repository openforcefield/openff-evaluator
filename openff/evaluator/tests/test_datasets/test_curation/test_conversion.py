import numpy
import pandas

from openff.evaluator.datasets.curation.components.conversion import (
    ConvertExcessDensityData,
    ConvertExcessDensityDataSchema,
)


def test_convert_density_v_excess():
    """Tests the `ConvertExcessDensityData` component."""

    data_rows = [
        {
            "N Components": 1,
            "Phase": "Liquid",
            "Temperature (K)": 298.15,
            "Pressure (kPa)": 101.325,
            "Component 1": "CCCCCCCC",
            "Mole Fraction 1": 1.0,
            "Density Value (g / ml)": 0.69867,
            "Source": "x",
        },
        {
            "N Components": 1,
            "Phase": "Liquid",
            "Temperature (K)": 298.15,
            "Pressure (kPa)": 101.325,
            "Component 1": "CCCCCCC(C)O",
            "Mole Fraction 1": 1.0,
            "Density Value (g / ml)": 0.81705,
            "Source": "y",
        },
        {
            "N Components": 2,
            "Phase": "Liquid",
            "Temperature (K)": 298.15,
            "Pressure (kPa)": 101.325,
            "Component 1": "CCCCCCCC",
            "Mole Fraction 1": 0.8,
            "Component 2": "CCCCCCC(C)O",
            "Mole Fraction 2": 0.2,
            "Density Value (g / ml)": 0.72157,
            "Source": "z",
        },
        {
            "N Components": 2,
            "Phase": "Liquid",
            "Temperature (K)": 298.15,
            "Pressure (kPa)": 101.325,
            "Component 1": "CCCCCCCC",
            "Mole Fraction 1": 0.8,
            "Component 2": "CCCCCCC(C)O",
            "Mole Fraction 2": 0.2,
            "ExcessMolarVolume Value (cm ** 3 / mol)": 0.06715,
            "Source": "w",
        },
    ]

    data_frame = pandas.DataFrame(data_rows)

    converted_data_frame = ConvertExcessDensityData.apply(
        data_frame, ConvertExcessDensityDataSchema(), 1
    )

    converted_data_frame = converted_data_frame[
        converted_data_frame["N Components"] == 2
    ]

    assert len(converted_data_frame) == 4

    excess_molar_volume = (
        converted_data_frame["ExcessMolarVolume Value (cm ** 3 / mol)"]
        .round(5)
        .unique()
    )
    excess_molar_volume = [x for x in excess_molar_volume if not pandas.isnull(x)]
    assert len(excess_molar_volume) == 1
    assert numpy.isclose(excess_molar_volume, 0.06715)

    density = converted_data_frame["Density Value (g / ml)"].round(5).unique()
    density = [x for x in density if not pandas.isnull(x)]
    assert len(density) == 1
    assert numpy.isclose(density, 0.72157)
