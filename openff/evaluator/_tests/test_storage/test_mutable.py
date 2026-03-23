import os
import tempfile
import pytest

from openff.evaluator._tests.utils import (
    create_dummy_equilibration_data,
    create_dummy_simulation_data,
)
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.storage import MutableLocalFileStorage
from openff.evaluator.storage.data import ForceFieldData, StoredSimulationData
from openff.evaluator.substances import Substance

DATA_FACTORIES = [
    pytest.param(create_dummy_simulation_data, id="simulation"),
    pytest.param(create_dummy_equilibration_data, id="equilibration"),
]

def test_combine_storages_update():
    """update() merges all objects from other into self."""
    substance_a = Substance.from_components("C")
    substance_b = Substance.from_components("O")

    with tempfile.TemporaryDirectory() as base_dir:
        dir_a = os.path.join(base_dir, "storage_a")
        dir_b = os.path.join(base_dir, "storage_b")
        data_a = os.path.join(base_dir, "data_a")
        data_b = os.path.join(base_dir, "data_b")

        obj_a = create_dummy_simulation_data(data_a, substance_a)
        obj_b = create_dummy_simulation_data(data_b, substance_b)

        storage_a = MutableLocalFileStorage(dir_a)
        storage_a.store_object(obj_a, data_a)

        storage_b = MutableLocalFileStorage(dir_b)
        storage_b.store_object(obj_b, data_b)

        storage_a.update(storage_b)

        keys = storage_a._stored_object_keys[StoredSimulationData.__name__]
        assert len(keys) == 2


def test_combine_storages_iadd():
    """``+=`` operator merges other into self."""
    substance_a = Substance.from_components("C")
    substance_b = Substance.from_components("O")

    with tempfile.TemporaryDirectory() as base_dir:
        dir_a = os.path.join(base_dir, "storage_a")
        dir_b = os.path.join(base_dir, "storage_b")
        data_a = os.path.join(base_dir, "data_a")
        data_b = os.path.join(base_dir, "data_b")

        obj_a = create_dummy_simulation_data(data_a, substance_a)
        obj_b = create_dummy_simulation_data(data_b, substance_b)

        storage_a = MutableLocalFileStorage(dir_a)
        storage_a.store_object(obj_a, data_a)

        storage_b = MutableLocalFileStorage(dir_b)
        storage_b.store_object(obj_b, data_b)

        storage_a += storage_b

        keys = storage_a._stored_object_keys[StoredSimulationData.__name__]
        assert len(keys) == 2


def test_combine_update_does_not_modify_other():
    """update() does not remove objects from the source storage."""
    substance = Substance.from_components("C")

    with tempfile.TemporaryDirectory() as base_dir:
        dir_a = os.path.join(base_dir, "storage_a")
        dir_b = os.path.join(base_dir, "storage_b")
        data_b = os.path.join(base_dir, "data_b")

        obj = create_dummy_simulation_data(data_b, substance)
        storage_a = MutableLocalFileStorage(dir_a)
        storage_b = MutableLocalFileStorage(dir_b)
        storage_b.store_object(obj, data_b)

        storage_a.update(storage_b)

        # original storage_b still has the object
        keys_b = storage_b._stored_object_keys[StoredSimulationData.__name__]
        assert len(keys_b) == 1


def test_combine_deduplicates_force_fields():
    """Combining two storages with the same force field stores it only once."""
    force_field_source = SmirnoffForceFieldSource.from_path("openff-2.2.1.offxml")

    with tempfile.TemporaryDirectory() as base_dir:
        dir_a = os.path.join(base_dir, "storage_a")
        dir_b = os.path.join(base_dir, "storage_b")

        storage_a = MutableLocalFileStorage(dir_a)
        storage_b = MutableLocalFileStorage(dir_b)

        storage_a.store_force_field(force_field_source)
        storage_b.store_force_field(force_field_source)

        storage_a.update(storage_b)

        keys = storage_a._stored_object_keys[ForceFieldData.__name__]
        assert len(keys) == 1

@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_store_data_copies_not_moves(factory):
    """store_object() copies ancillary data, leaving the source directory intact."""
    substance = Substance.from_components("C")
    with tempfile.TemporaryDirectory() as base_dir:
        storage_dir = os.path.join(base_dir, "storage")
        data_dir = os.path.join(base_dir, "data_directory")

        data = factory(data_dir, substance)

        storage = MutableLocalFileStorage(storage_dir)
        storage.store_object(data, data_dir)

        # Source directory must still exist after storing
        assert os.path.isdir(data_dir), "Source ancillary directory was moved (destroyed)"
        assert os.path.isfile(os.path.join(data_dir, data.coordinate_file_name))
        if hasattr(data, "trajectory_file_name"):
            assert os.path.isfile(os.path.join(data_dir, data.trajectory_file_name))


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_store_data_copy_stores_correctly(factory):
    """The stored copy can be retrieved and contains the expected data."""
    substance = Substance.from_components("C")
    with tempfile.TemporaryDirectory() as base_dir:
        storage_dir = os.path.join(base_dir, "storage")
        data_dir = os.path.join(base_dir, "data_directory")

        data = factory(data_dir, substance)

        storage = MutableLocalFileStorage(storage_dir)
        key = storage.store_object(data, data_dir)

        retrieved, retrieved_dir = storage.retrieve_object(key)
        assert retrieved is not None
        assert retrieved.substance.json() == data.substance.json()
        if hasattr(data, "max_number_of_molecules"):
            assert retrieved.max_number_of_molecules == data.max_number_of_molecules
        assert os.path.isdir(retrieved_dir)


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_parent_move_behaviour_differs(factory):
    """LocalFileStorage moves; MutableLocalFileStorage copies."""
    from openff.evaluator.storage import LocalFileStorage

    substance = Substance.from_components("C")
    with tempfile.TemporaryDirectory() as base_dir:
        # Parent: source directory is destroyed after store
        parent_storage_dir = os.path.join(base_dir, "parent_storage")
        parent_data_dir = os.path.join(base_dir, "parent_data")
        parent_data = factory(parent_data_dir, substance)
        parent_storage = LocalFileStorage(parent_storage_dir)
        parent_storage.store_object(parent_data, parent_data_dir)
        assert not os.path.isdir(parent_data_dir), (
            "LocalFileStorage should have moved (destroyed) the source directory"
        )

        # Subclass: source directory survives
        mutable_storage_dir = os.path.join(base_dir, "mutable_storage")
        mutable_data_dir = os.path.join(base_dir, "mutable_data")
        mutable_data = factory(mutable_data_dir, substance)
        mutable_storage = MutableLocalFileStorage(mutable_storage_dir)
        mutable_storage.store_object(mutable_data, mutable_data_dir)
        assert os.path.isdir(mutable_data_dir), (
            "MutableLocalFileStorage should have copied, not moved, the source directory"
        )

def test_combine_storages_update():
    """update() merges all objects from other into self."""
    substance_a = Substance.from_components("C")
    substance_b = Substance.from_components("O")

    with tempfile.TemporaryDirectory() as base_dir:
        dir_a = os.path.join(base_dir, "storage_a")
        dir_b = os.path.join(base_dir, "storage_b")
        data_a = os.path.join(base_dir, "data_a")
        data_b = os.path.join(base_dir, "data_b")

        obj_a = create_dummy_simulation_data(data_a, substance_a)
        obj_b = create_dummy_simulation_data(data_b, substance_b)

        storage_a = MutableLocalFileStorage(dir_a)
        storage_a.store_object(obj_a, data_a)

        storage_b = MutableLocalFileStorage(dir_b)
        storage_b.store_object(obj_b, data_b)

        storage_a.update(storage_b)

        keys = storage_a._stored_object_keys[StoredSimulationData.__name__]
        assert len(keys) == 2


def test_combine_storages_iadd():
    """``+=`` operator merges other into self."""
    substance_a = Substance.from_components("C")
    substance_b = Substance.from_components("O")

    with tempfile.TemporaryDirectory() as base_dir:
        dir_a = os.path.join(base_dir, "storage_a")
        dir_b = os.path.join(base_dir, "storage_b")
        data_a = os.path.join(base_dir, "data_a")
        data_b = os.path.join(base_dir, "data_b")

        obj_a = create_dummy_simulation_data(data_a, substance_a)
        obj_b = create_dummy_simulation_data(data_b, substance_b)

        storage_a = MutableLocalFileStorage(dir_a)
        storage_a.store_object(obj_a, data_a)

        storage_b = MutableLocalFileStorage(dir_b)
        storage_b.store_object(obj_b, data_b)

        storage_a += storage_b

        keys = storage_a._stored_object_keys[StoredSimulationData.__name__]
        assert len(keys) == 2


def test_combine_update_does_not_modify_other():
    """update() does not remove objects from the source storage."""
    substance = Substance.from_components("C")

    with tempfile.TemporaryDirectory() as base_dir:
        dir_a = os.path.join(base_dir, "storage_a")
        dir_b = os.path.join(base_dir, "storage_b")
        data_b = os.path.join(base_dir, "data_b")

        obj = create_dummy_simulation_data(data_b, substance)
        storage_a = MutableLocalFileStorage(dir_a)
        storage_b = MutableLocalFileStorage(dir_b)
        storage_b.store_object(obj, data_b)

        storage_a.update(storage_b)

        # original storage_b still has the object
        keys_b = storage_b._stored_object_keys[StoredSimulationData.__name__]
        assert len(keys_b) == 1


def test_combine_deduplicates_force_fields():
    """Combining two storages with the same force field stores it only once."""
    force_field_source = SmirnoffForceFieldSource.from_path("openff-2.2.1.offxml")

    with tempfile.TemporaryDirectory() as base_dir:
        dir_a = os.path.join(base_dir, "storage_a")
        dir_b = os.path.join(base_dir, "storage_b")

        storage_a = MutableLocalFileStorage(dir_a)
        storage_b = MutableLocalFileStorage(dir_b)

        storage_a.store_force_field(force_field_source)
        storage_b.store_force_field(force_field_source)

        storage_a.update(storage_b)

        keys = storage_a._stored_object_keys[ForceFieldData.__name__]
        assert len(keys) == 1
