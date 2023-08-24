"""The module contains curation components for converting one type of property (e.g.
density) into another (e.g excess molar volume)"""
import functools
import logging
from typing import TYPE_CHECKING, Union

import pandas
from typing_extensions import Literal

from openff.evaluator._pydantic import Field
from openff.evaluator.datasets.curation.components import (
    CurationComponent,
    CurationComponentSchema,
)

if TYPE_CHECKING:
    conint = int
    PositiveInt = int
    PositiveFloat = float

else:
    from pydantic import conint

logger = logging.getLogger(__name__)


class ConvertExcessDensityDataSchema(CurationComponentSchema):
    type: Literal["ConvertExcessDensityDataSchema"] = "ConvertExcessDensityDataSchema"

    temperature_precision: conint(ge=0) = Field(
        2,
        description="The number of decimal places to compare temperatures (K) to "
        "within when attempting to identify compatible pure and binary data.",
    )
    pressure_precision: conint(ge=0) = Field(
        1,
        description="The number of decimal places to compare pressures (kPa) to "
        "within when attempting to identify compatible pure and binary data.",
    )


class ConvertExcessDensityData(CurationComponent):
    """A component for converting binary mass density data to excess molar volume
    data and vice versa where pure density data measured for the components is
    available.

    Notes
    -----
    This protocol may result in duplicate data points being generated. It is
    recommended to apply the de-duplication filter after this component has been
    applied.
    """

    @classmethod
    @functools.lru_cache(500)
    def _molecular_weight(cls, smiles):
        from openff.toolkit.topology import Molecule

        molecule = Molecule.from_smiles(smiles, allow_undefined_stereo=True)

        # Atom.mass is guaranteed to be in Daltons
        molecular_weight = sum(atom.mass.m for atom in molecule.atoms)

        return molecular_weight

    @classmethod
    def _find_overlapping_data_points(
        cls,
        pure_data_set: pandas.DataFrame,
        binary_data_set: pandas.DataFrame,
        schema: ConvertExcessDensityDataSchema,
    ):
        """Finds those binary data points for which there also exists pure
         data points for each component in the binary system.

        Parameters
        ----------
        pure_data_set
            The pure data set.
        binary_data_set
            The binary data set.
        schema
            The schema for this component.

        Returns
        -------
        pandas.DataFrame
            The data set containing the pure and binary data points
            measured for the same substances at the same state pounts
        """

        if len(pure_data_set) == 0 or len(binary_data_set) == 0:
            return pandas.DataFrame()

        pure_data_set = pure_data_set.dropna(axis=1, how="all")
        binary_data_set = binary_data_set.dropna(axis=1, how="all")

        # Round the floats which will be compared.
        pure_data_set["Temperature (K)"] = pure_data_set["Temperature (K)"].round(
            schema.temperature_precision
        )
        pure_data_set["Pressure (kPa)"] = pure_data_set["Pressure (kPa)"].round(
            schema.pressure_precision
        )

        binary_data_set["Temperature (K)"] = binary_data_set["Temperature (K)"].round(
            schema.temperature_precision
        )
        binary_data_set["Pressure (kPa)"] = binary_data_set["Pressure (kPa)"].round(
            schema.pressure_precision
        )

        # Only consider pure measurements which only have mole fractions defined
        if "Exact Amount 1" in pure_data_set:
            pure_data_set = pure_data_set[pure_data_set["Exact Amount 1"].isna()]

        if "Mole Fraction 1" not in pure_data_set:
            return pandas.DataFrame()

        pure_data_set = pure_data_set[pure_data_set["Mole Fraction 1"].notna()]

        # Retain only the minimally informative pure data columns.
        data_columns = [
            "Temperature (K)",
            "Pressure (kPa)",
            "Phase",
            "Component 1",
            "Density Value (g / ml)",
            "Source",
        ]

        if "Density Uncertainty (g / ml)" in pure_data_set:
            data_columns.append("Density Uncertainty (g / ml)")

        pure_data_set = pure_data_set[data_columns]

        pure_data_set = pandas.merge(
            pure_data_set,
            pure_data_set,
            how="inner",
            on=["Temperature (K)", "Pressure (kPa)", "Phase"],
        )

        overlapping_set = pandas.merge(
            binary_data_set,
            pure_data_set,
            how="inner",
            left_on=[
                "Temperature (K)",
                "Pressure (kPa)",
                "Phase",
                "Component 1",
                "Component 2",
            ],
            right_on=[
                "Temperature (K)",
                "Pressure (kPa)",
                "Phase",
                "Component 1_x",
                "Component 1_y",
            ],
            suffixes=("", ""),
        )

        return overlapping_set

    @classmethod
    def _convert_density_to_v_excess(
        cls, density_data_set: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Converts a pandas data frame containing both binary mass densities
        and pure mass densities into one which contains excess molar volume
        measurements.

        Parameters
        ----------
        density_data_set
            The data frame containing both pure and binary
            density measurements. This should be generated using the
            `find_overlapping_data_points` function.

        Returns
        -------
            A data frame which contains the excess molar volume measurements.
        """

        m_1 = density_data_set["Component 1"].apply(cls._molecular_weight)
        m_1_x_1 = m_1 * density_data_set["Mole Fraction 1"]

        m_2 = density_data_set["Component 2"].apply(cls._molecular_weight)
        m_2_x_2 = m_2 * density_data_set["Mole Fraction 2"]

        v_excess = (
            (m_1_x_1 + m_2_x_2) / density_data_set["Density Value (g / ml)"]
            - m_1_x_1 / density_data_set["Density Value (g / ml)_x"]
            - m_2_x_2 / density_data_set["Density Value (g / ml)_y"]
        )

        source = density_data_set[["Source", "Source_x", "Source_y"]].agg(
            " + ".join, axis=1
        )

        # Add the new values to a new data frame.
        columns_to_drop = [
            x for x in density_data_set if x.endswith("_x") or x.endswith("_y")
        ]
        columns_to_drop.append("Density Value (g / ml)")
        columns_to_drop.append("Source")

        if "Density Uncertainty (g / ml)" in density_data_set:
            columns_to_drop.append("Density Uncertainty (g / ml)")

        v_excess_data_set = density_data_set.drop(columns=columns_to_drop).copy()

        v_excess_data_set.insert(
            v_excess_data_set.shape[1],
            "ExcessMolarVolume Value (cm ** 3 / mol)",
            v_excess,
        )
        v_excess_data_set.insert(v_excess_data_set.shape[1], "Source", source)

        return v_excess_data_set

    @classmethod
    def _convert_v_excess_to_density(
        cls, v_excess_data_set: pandas.DataFrame
    ) -> pandas.DataFrame:
        """Converts a pandas data frame containing both excess molar volumes
        and pure mass densities into one which contains binary mass density
        measurements.

        Parameters
        ----------
        v_excess_data_set
            The data frame containing both pure density and excess molar
            volume measurements. This should be generated using the
            `find_overlapping_data_points` function.

        Returns
        -------
            A data frame which contains the excess molar volume measurements.
        """

        m_1 = v_excess_data_set["Component 1"].apply(cls._molecular_weight)
        m_1_x_1 = m_1 * v_excess_data_set["Mole Fraction 1"]

        m_2 = v_excess_data_set["Component 2"].apply(cls._molecular_weight)
        m_2_x_2 = m_2 * v_excess_data_set["Mole Fraction 2"]

        v_excess = v_excess_data_set["ExcessMolarVolume Value (cm ** 3 / mol)"]

        denominator = (
            v_excess
            + m_1_x_1 / v_excess_data_set["Density Value (g / ml)_x"]
            + m_2_x_2 / v_excess_data_set["Density Value (g / ml)_y"]
        )

        rho_binary = (m_1_x_1 + m_2_x_2) / denominator

        source = v_excess_data_set[["Source", "Source_x", "Source_y"]].agg(
            " + ".join, axis=1
        )

        # Add the new values to a new data frame.
        columns_to_drop = [
            x for x in v_excess_data_set if x.endswith("_x") or x.endswith("_y")
        ]
        columns_to_drop.append("ExcessMolarVolume Value (cm ** 3 / mol)")
        columns_to_drop.append("Source")

        if "ExcessMolarVolume Uncertainty (cm ** 3 / mol)" in v_excess_data_set:
            columns_to_drop.append("ExcessMolarVolume Uncertainty (cm ** 3 / mol)")

        density_data_set = v_excess_data_set.drop(columns=columns_to_drop).copy()

        density_data_set.insert(
            density_data_set.shape[1] - 1, "Density Value (g / ml)", rho_binary
        )
        density_data_set.insert(density_data_set.shape[1] - 1, "Source", source)

        return density_data_set

    @classmethod
    def _apply(
        cls,
        data_frame: pandas.DataFrame,
        schema: ConvertExcessDensityDataSchema,
        n_processes,
    ) -> pandas.DataFrame:
        if len(data_frame) == 0:
            return data_frame

        # Check to make sure the data frame contains at least a
        # density column which may store pure densities.
        if "Density Value (g / ml)" not in data_frame:
            return data_frame

        # Separate out the data sets of interest
        pure_density_data = data_frame[
            (data_frame["Density Value (g / ml)"].notna())
            & (data_frame["N Components"] == 1)
        ]

        pure_density_data = pure_density_data.dropna(axis=1, how="all")

        # Exit early if no pure densities can be found.
        if len(pure_density_data) == 0:
            return data_frame

        # Add the pure data to the binary data sets to make conversion easier.
        binary_density_data = data_frame[
            (data_frame["Density Value (g / ml)"].notna())
            & (data_frame["N Components"] == 2)
        ]
        binary_density_data = binary_density_data.dropna(axis=1, how="all")

        binary_density_data = cls._find_overlapping_data_points(
            pure_density_data, binary_density_data, schema
        )

        v_excess_data = pandas.DataFrame()

        if "ExcessMolarVolume Value (cm ** 3 / mol)" in data_frame:
            v_excess_data = data_frame[
                (data_frame["ExcessMolarVolume Value (cm ** 3 / mol)"].notna())
                & (data_frame["N Components"] == 2)
            ]
            v_excess_data = v_excess_data.dropna(axis=1, how="all")
            v_excess_data = cls._find_overlapping_data_points(
                pure_density_data, v_excess_data, schema
            )

        if len(binary_density_data) == 0 and len(v_excess_data) == 0:
            return data_frame

        # Inter-convert the two sets
        data_to_concat = [data_frame]

        if len(binary_density_data) > 0:
            v_excess_from_density = cls._convert_density_to_v_excess(
                binary_density_data
            )
            data_to_concat.append(v_excess_from_density)

        if len(v_excess_data) > 0:
            density_from_v_excess = cls._convert_v_excess_to_density(v_excess_data)
            data_to_concat.append(density_from_v_excess)

        if len(data_to_concat) > 1:
            converted_data = pandas.concat(
                data_to_concat,
                ignore_index=True,
                sort=False,
            )

        else:
            converted_data = data_frame

        return converted_data


ConversionComponentSchema = Union[ConvertExcessDensityDataSchema]
