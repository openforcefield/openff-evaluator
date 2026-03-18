"""Unit tests for MutableLocalFileStorage — copy-on-store behaviour."""
import os
import tempfile

from openff.evaluator._tests.utils import create_dummy_simulation_data
from openff.evaluator.storage import MutableLocalFileStorage
from openff.evaluator.substances import Substance


def test_store_object_copies_not_moves():
    """store_object() copies ancillary data, leaving the source directory intact."""
    substance = Substance.from_components("C")
    with tempfile.TemporaryDirectory() as base_dir:
        storage_dir = os.path.join(base_dir, "storage")
        data_dir = os.path.join(base_dir, "data_directory")

        data = create_dummy_simulation_data(data_dir, substance)

        storage = MutableLocalFileStorage(storage_dir)
        storage.store_object(data, data_dir)

        # Source directory must still exist after storing
        assert os.path.isdir(data_dir), "Source ancillary directory was moved (destroyed)"
        assert os.path.isfile(os.path.join(data_dir, data.coordinate_file_name))
        assert os.path.isfile(os.path.join(data_dir, data.trajectory_file_name))


def test_store_object_copy_stores_correctly():
    """The stored copy can be retrieved and contains the expected data."""
    substance = Substance.from_components("C")
    with tempfile.TemporaryDirectory() as base_dir:
        storage_dir = os.path.join(base_dir, "storage")
        data_dir = os.path.join(base_dir, "data_directory")

        data = create_dummy_simulation_data(data_dir, substance)

        storage = MutableLocalFileStorage(storage_dir)
        key = storage.store_object(data, data_dir)

        retrieved, retrieved_dir = storage.retrieve_object(key)
        assert retrieved is not None
        assert retrieved.substance.json() == data.substance.json()
        assert os.path.isdir(retrieved_dir)


def test_parent_move_behaviour_differs():
    """Confirm LocalFileStorage (parent) moves, while MutableLocalFileStorage copies."""
    from openff.evaluator.storage import LocalFileStorage

    substance = Substance.from_components("C")
    with tempfile.TemporaryDirectory() as base_dir:
        # Parent: source directory is destroyed after store
        parent_storage_dir = os.path.join(base_dir, "parent_storage")
        parent_data_dir = os.path.join(base_dir, "parent_data")
        parent_data = create_dummy_simulation_data(parent_data_dir, substance)
        parent_storage = LocalFileStorage(parent_storage_dir)
        parent_storage.store_object(parent_data, parent_data_dir)
        assert not os.path.isdir(parent_data_dir), (
            "LocalFileStorage should have moved (destroyed) the source directory"
        )

        # Subclass: source directory survives
        mutable_storage_dir = os.path.join(base_dir, "mutable_storage")
        mutable_data_dir = os.path.join(base_dir, "mutable_data")
        mutable_data = create_dummy_simulation_data(mutable_data_dir, substance)
        mutable_storage = MutableLocalFileStorage(mutable_storage_dir)
        mutable_storage.store_object(mutable_data, mutable_data_dir)
        assert os.path.isdir(mutable_data_dir), (
            "MutableLocalFileStorage should have copied, not moved, the source directory"
        )
