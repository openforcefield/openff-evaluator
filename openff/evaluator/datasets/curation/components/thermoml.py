import glob
import io
import logging
import os
import tarfile
from multiprocessing import Pool
from typing import List, Optional, Union

import pandas
import requests
from pydantic import Field, HttpUrl
from typing_extensions import Literal

from openff.evaluator.datasets.curation.components import (
    CurationComponent,
    CurationComponentSchema,
)
from openff.evaluator.datasets.thermoml import ThermoMLDataSet
from openff.evaluator.utils.utils import temporarily_change_directory

logger = logging.getLogger(__name__)


def _default_journals():
    return ["JCED", "JCT", "FPE", "TCA", "IJT"]


class ImportThermoMLDataSchema(CurationComponentSchema):

    type: Literal["ImportThermoMLData"] = "ImportThermoMLData"

    retain_uncertainties: bool = Field(
        True,
        description="If False, all uncertainties in measured property values will be "
        "stripped from the final data set.",
    )

    cache_file_name: Optional[str] = Field(
        None,
        description="The path to the file to store the output of this component "
        "into, and to restore the output of this component from.",
    )

    journal_names: List[Literal["JCED", "JCT", "FPE", "TCA", "IJT"]] = Field(
        default_factory=_default_journals,
        description="The abbreviated names of the journals to import data from.",
    )
    root_archive_url: HttpUrl = Field(
        default="https://trc.nist.gov/ThermoML",
        description="The root url where the ThermoML archives can be downloaded from.",
    )


class ImportThermoMLData(CurationComponent):
    """A component which will import all supported data from the
    NIST ThermoML archive for (optionally) specified journals.
    """

    @classmethod
    def _download_data(cls, schema: ImportThermoMLDataSchema):

        for journal in schema.journal_names:

            # Download the archive of all properties from the journal.
            request = requests.get(
                f"{schema.root_archive_url}/{journal}.tgz", stream=True
            )

            # Make sure the request went ok.
            try:
                request.raise_for_status()
            except requests.exceptions.HTTPError as error:
                print(error.response.text)
                raise

                # Unzip the files into the temporary directory.
            tar_file = tarfile.open(fileobj=io.BytesIO(request.content))
            tar_file.extractall()

    @classmethod
    def _process_archive(cls, file_path: str) -> pandas.DataFrame:

        logger.debug(f"Processing {file_path}")

        # noinspection PyBroadException
        try:
            data_set = ThermoMLDataSet.from_file(file_path)

        except Exception:

            logger.exception(
                f"An exception was raised when processing {file_path}. This file will "
                f"be skipped."
            )
            return pandas.DataFrame()

        # A data set will be none if no 'valid' properties were found
        # in the archive file.
        if data_set is None:
            return pandas.DataFrame()

        data_frame = data_set.to_pandas()
        return data_frame

    @classmethod
    def _apply(
        cls,
        data_frame: pandas.DataFrame,
        schema: ImportThermoMLDataSchema,
        n_processes,
    ) -> pandas.DataFrame:

        if schema.cache_file_name is not None and os.path.isfile(
            schema.cache_file_name
        ):

            cached_data = pandas.read_csv(schema.cache_file_name)
            return cached_data

        with temporarily_change_directory():

            logger.debug("Downloading archive data")

            cls._download_data(schema)

            # Get the names of the extracted files
            file_names = glob.glob("*.xml")

            logger.debug("Processing archives")

            with Pool(processes=n_processes) as pool:
                data_frames = [*pool.imap(cls._process_archive, file_names)]

            pool.join()

        logger.debug("Joining archives")

        thermoml_data_frame = pandas.concat(data_frames, ignore_index=True, sort=False)

        for header in thermoml_data_frame:

            if header.find(" Uncertainty ") >= 0 and not schema.retain_uncertainties:
                thermoml_data_frame = thermoml_data_frame.drop(header, axis=1)

        data_frame = pandas.concat(
            [data_frame, thermoml_data_frame], ignore_index=True, sort=False
        )

        if schema.cache_file_name is not None:
            data_frame.to_csv(schema.cache_file_name, index=False)

        return data_frame


ThermoMLComponentSchema = Union[ImportThermoMLDataSchema]
