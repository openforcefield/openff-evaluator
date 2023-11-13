import io
import logging
import re
from typing import List, Union

import pandas
import requests
from typing_extensions import Literal

from openff.evaluator.datasets import (
    MeasurementSource,
    PhysicalPropertyDataSet,
    PropertyPhase,
)
from openff.evaluator.datasets.curation.components import (
    CurationComponent,
    CurationComponentSchema,
)
from openff.evaluator.properties import SolvationFreeEnergy
from openff.evaluator.substances import Component, ExactAmount, MoleFraction, Substance
from openff.evaluator.thermodynamics import ThermodynamicState

logger = logging.getLogger(__name__)


class ImportFreeSolvSchema(CurationComponentSchema):
    type: Literal["ImportFreeSolv"] = "ImportFreeSolv"


class ImportFreeSolv(CurationComponent):
    """A component which will import the latest version of the FreeSolv
    data set from the GitHub repository where it is stored.
    """

    @classmethod
    def _download_free_solv(cls) -> pandas.DataFrame:
        """Downloads the FreeSolv data set from GitHub.

        Returns
        -------
            The Free Solv data stored in a pandas data frame.
        """

        # Download the database from GitHub
        download_request = requests.get(
            "https://raw.githubusercontent.com/MobleyLab/FreeSolv/master/database.txt"
        )
        download_request.raise_for_status()

        text_contents = download_request.text

        # Unify the delimiter
        text_contents = text_contents.replace("; ", ";")

        # Convert the set to a pandas object
        text_buffer = io.StringIO(text_contents)
        free_solv_data_frame = pandas.read_csv(text_buffer, delimiter=";", skiprows=2)

        return free_solv_data_frame

    @classmethod
    def _validate_doi(cls, doi: str):
        """Attempts to validate a string which may contain a (or multiple)
        digital object identifier. If a valid DOI is not found, the FreeSolv
        DOI itself is returned."""

        fall_back_doi = "10.5281/zenodo.596537"

        # From https://www.crossref.org/blog/dois-and-matching-regular-expressions/
        doi_patterns = [
            r"^10.\d{4,9}/[-._;()/:A-Z0-9]+$",
            r"^10.1002/[^\s]+$",
            r"^10.\d{4}/\d+-\d+X?(\d+)\d+<[\d\w]+:[\d\w]*>\d+.\d+.\w+;\d$",
            r"^10.1021/\w\w\d+$",
            r"^10.1207/[\w\d]+\&\d+_\d+$",
        ]

        # Split the string to try and catch concatenated DOIs
        doi_split = doi.split(" and ")

        matched_dois: List[str] = []

        for split_doi in doi_split:
            matched_doi = None

            for doi_pattern in doi_patterns:
                regex_match = re.match(doi_pattern, split_doi, re.I)

                if not regex_match:
                    continue

                matched_doi = regex_match.group()
                break

            if not isinstance(matched_doi, str):
                continue

            matched_dois.append(matched_doi)

        final_doi = (
            fall_back_doi if len(matched_dois) == 0 else " + ".join(matched_dois)
        )
        return final_doi

    @classmethod
    def _apply(
        cls,
        data_frame: pandas.DataFrame,
        schema: ImportFreeSolvSchema,
        n_processes,
    ) -> pandas.DataFrame:
        from openff.units import unit

        from openff.evaluator import properties, substances

        # Convert the data frame into data rows.
        free_solv_data_frame = cls._download_free_solv()

        data_entries = []

        for _, row in free_solv_data_frame.iterrows():
            # Extract and standardize the SMILES pattern of the
            solute_smiles = row["SMILES"].lstrip().rstrip()
            solute_smiles = substances.Component(solute_smiles).smiles

            # Build the substance.
            substance = Substance()
            substance.add_component(Component(smiles="O"), MoleFraction(1.0))
            substance.add_component(
                Component(smiles=solute_smiles, role=Component.Role.Solute),
                ExactAmount(1),
            )

            # Extract the value and uncertainty
            value = (
                float(row["experimental value (kcal/mol)"])
                * unit.kilocalorie
                / unit.mole
            )
            std_error = (
                float(row["experimental uncertainty (kcal/mol)"])
                * unit.kilocalorie
                / unit.mole
            )

            # Attempt to extract a DOI
            original_source = row[
                "experimental reference (original or paper this value was taken from)"
            ]
            doi = cls._validate_doi(original_source)

            data_entry = SolvationFreeEnergy(
                thermodynamic_state=ThermodynamicState(
                    temperature=298.15 * unit.kelvin,
                    pressure=101.325 * unit.kilopascal,
                ),
                phase=PropertyPhase.Liquid,
                substance=substance,
                value=value.to(properties.SolvationFreeEnergy.default_unit()),
                uncertainty=std_error.to(properties.SolvationFreeEnergy.default_unit()),
                source=MeasurementSource(doi=doi),
            )
            data_entries.append(data_entry)

        data_set = PhysicalPropertyDataSet()
        data_set.add_properties(*data_entries)

        free_solv_data_frame = data_set.to_pandas()

        data_frame = pandas.concat(
            [data_frame, free_solv_data_frame], ignore_index=True, sort=False
        )

        return data_frame


FreeSolvComponentSchema = Union[ImportFreeSolvSchema]
