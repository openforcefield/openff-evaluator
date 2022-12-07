import pandas
import pytest

from openff.evaluator.datasets.curation.components.freesolv import (
    ImportFreeSolv,
    ImportFreeSolvSchema,
)


def test_import_free_solv_data():
    """Tests that the FreeSolv data set can be imported from a
    remote source."""

    free_solv_data_frame = ImportFreeSolv._download_free_solv()

    data_frame = ImportFreeSolv.apply(pandas.DataFrame(), ImportFreeSolvSchema())
    assert data_frame is not None and len(data_frame) == len(free_solv_data_frame)
