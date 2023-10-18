import os
import tarfile
from tempfile import NamedTemporaryFile

import pandas

from openff.evaluator.datasets.curation.components.thermoml import (
    ImportThermoMLData,
    ImportThermoMLDataSchema,
)
from openff.evaluator.utils import get_data_filename


def test_import_thermoml_data(requests_mock):
    """Tests that ThermoML archive files can be imported from a
    remote source."""

    # Create a tarball to be downloaded.
    source_path = get_data_filename(os.path.join("test", "properties", "mass.xml"))

    with NamedTemporaryFile(suffix=".tgz") as tar_file:
        with tarfile.open(tar_file.name, "w:gz") as tar:
            tar.add(
                source_path,
                arcname=os.path.join("10.1021", os.path.basename(source_path)),
            )

        schema = ImportThermoMLDataSchema()

        with open(tar_file.name, "rb") as file:
            requests_mock.get(schema.root_archive_url, content=file.read())

        data_frame = ImportThermoMLData.apply(pandas.DataFrame(), schema)

        assert data_frame is not None and len(data_frame) == 1
