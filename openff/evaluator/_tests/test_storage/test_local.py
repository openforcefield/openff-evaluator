"""
Units tests for openff.evaluator.storage.localfile
"""

import os

from openff.evaluator.storage import LocalFileStorage


def test_root_directory(tmp_path):
    local_storage_path = os.path.join(tmp_path, "stored-data")
    local_storage = LocalFileStorage(root_directory=local_storage_path)

    assert os.path.isdir(local_storage_path)
    assert local_storage.root_directory == local_storage_path
