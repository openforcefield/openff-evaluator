import os
import tempfile

import pytest

from openff.evaluator._tests.utils import (
    create_dummy_equilibration_data,
    create_dummy_simulation_data,
)
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.storage import MutableLocalFileStorage
from openff.evaluator.storage.data import (
    ForceFieldData,
    StoredEquilibrationData,
    StoredSimulationData,
)
from openff.evaluator.substances import Component, Substance

DATA_FACTORIES = [
    pytest.param(create_dummy_simulation_data, id="simulation"),
    pytest.param(create_dummy_equilibration_data, id="equilibration"),
]

STORED_DATA_CASES = [
    pytest.param(
        create_dummy_simulation_data,
        StoredSimulationData,
        id="simulation",
    ),
    pytest.param(
        create_dummy_equilibration_data,
        StoredEquilibrationData,
        id="equilibration",
    ),
]

COMBINE_OPS = [
    pytest.param(lambda a, b: a.update(b), id="update"),
    pytest.param(lambda a, b: a.__iadd__(b), id="iadd"),
]


def _create_stored_data(
    factory,
    directory,
    substance,
    calculation_id,
    statistical_inefficiency,
):
    # Keep the generated objects consistent across test cases so that
    # overlap / replacement behavior depends only on the fields we vary.
    kwargs = {
        "calculation_id": calculation_id,
        "statistical_inefficiency": statistical_inefficiency,
        "number_of_molecules": 100,
    }

    if factory is create_dummy_equilibration_data:
        kwargs["max_number_of_molecules"] = 100

    return factory(directory, substance, **kwargs)


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
        assert os.path.isdir(
            data_dir
        ), "Source ancillary directory was moved (destroyed)"
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
        assert not os.path.isdir(
            parent_data_dir
        ), "LocalFileStorage should have moved (destroyed) the source directory"

        # Subclass: source directory survives
        mutable_storage_dir = os.path.join(base_dir, "mutable_storage")
        mutable_data_dir = os.path.join(base_dir, "mutable_data")
        mutable_data = factory(mutable_data_dir, substance)
        mutable_storage = MutableLocalFileStorage(mutable_storage_dir)
        mutable_storage.store_object(mutable_data, mutable_data_dir)
        assert os.path.isdir(
            mutable_data_dir
        ), "MutableLocalFileStorage should have copied, not moved, the source directory"


@pytest.mark.parametrize("combine_op", COMBINE_OPS)
@pytest.mark.parametrize("factory, stored_data_class", STORED_DATA_CASES)
def test_combine_storages(combine_op, factory, stored_data_class):
    """update() and += both merge all objects from other into self."""
    substance_a = Substance.from_components("C")
    substance_b = Substance.from_components("O")

    with tempfile.TemporaryDirectory() as base_dir:
        dir_a = os.path.join(base_dir, "storage_a")
        dir_b = os.path.join(base_dir, "storage_b")
        data_a = os.path.join(base_dir, "data_a")
        data_b = os.path.join(base_dir, "data_b")

        obj_a = factory(data_a, substance_a)
        obj_b = factory(data_b, substance_b)

        storage_a = MutableLocalFileStorage(dir_a)
        storage_a.store_object(obj_a, data_a)

        storage_b = MutableLocalFileStorage(dir_b)
        storage_b.store_object(obj_b, data_b)

        combine_op(storage_a, storage_b)

        keys = storage_a._stored_object_keys[stored_data_class.__name__]
        assert len(keys) == 2


@pytest.mark.parametrize("factory, stored_data_class", STORED_DATA_CASES)
def test_combine_update_does_not_modify_other(factory, stored_data_class):
    """update() does not remove objects from the source storage."""
    substance_a = Substance.from_components("C")
    substance_b = Substance.from_components("O")

    with tempfile.TemporaryDirectory() as base_dir:
        dir_a = os.path.join(base_dir, "storage_a")
        dir_b = os.path.join(base_dir, "storage_b")
        data_a = os.path.join(base_dir, "data_a")
        data_b1 = os.path.join(base_dir, "data_b1")
        data_b2 = os.path.join(base_dir, "data_b2")

        storage_a = MutableLocalFileStorage(dir_a)
        storage_b = MutableLocalFileStorage(dir_b)

        # Ensure target storage already has data before merging.
        obj_a = _create_stored_data(
            factory,
            data_a,
            substance_a,
            calculation_id="a-1",
            statistical_inefficiency=1.0,
        )
        key_a = storage_a.store_object(obj_a, data_a)

        # Populate source storage with multiple entries so the test checks
        # that update() leaves all source keys intact, not just a single key.
        obj_b1 = _create_stored_data(
            factory,
            data_b1,
            substance_a,
            calculation_id="b-1",
            statistical_inefficiency=0.8,
        )
        obj_b2 = _create_stored_data(
            factory,
            data_b2,
            substance_b,
            calculation_id="b-2",
            statistical_inefficiency=1.2,
        )

        key_b1 = storage_b.store_object(obj_b1, data_b1)
        key_b2 = storage_b.store_object(obj_b2, data_b2)

        # Merge source into target.
        storage_a.update(storage_b)

        # target storage retains its original object and includes source objects
        keys_a = storage_a._stored_object_keys[stored_data_class.__name__]
        assert len(keys_a) == 3
        assert key_a in keys_a

        # original storage_b still has all objects
        keys_b = storage_b._stored_object_keys[stored_data_class.__name__]
        assert len(keys_b) == 2
        assert key_b1 in keys_b
        assert key_b2 in keys_b


@pytest.mark.parametrize("factory, stored_data_class", STORED_DATA_CASES)
def test_update_replaces_overlapping_data_by_statistical_inefficiency(
    factory, stored_data_class
):
    """update() keeps one key per overlap and stores the lower inefficiency data."""
    substance_a = Substance.from_components("C")
    substance_b = Substance.from_components("O")
    substance_c = Substance.from_components("CC")

    with tempfile.TemporaryDirectory() as base_dir:
        dir_a = os.path.join(base_dir, "storage_a")
        dir_b = os.path.join(base_dir, "storage_b")

        storage_a = MutableLocalFileStorage(dir_a)
        storage_b = MutableLocalFileStorage(dir_b)

        # Target storage A has three entries. One entry is intentionally created
        # to match an entry that will be added in B (same query identity), but
        # with worse statistical inefficiency.
        obj_a_overlap = _create_stored_data(
            factory,
            os.path.join(base_dir, "a_overlap"),
            substance_a,
            calculation_id="shared-id",
            statistical_inefficiency=2.5,
        )
        obj_a_unique_1 = _create_stored_data(
            factory,
            os.path.join(base_dir, "a_unique_1"),
            substance_b,
            calculation_id="a-only-1",
            statistical_inefficiency=1.3,
        )
        obj_a_unique_2 = _create_stored_data(
            factory,
            os.path.join(base_dir, "a_unique_2"),
            substance_c,
            calculation_id="a-only-2",
            statistical_inefficiency=1.1,
        )

        overlap_key = storage_a.store_object(
            obj_a_overlap, os.path.join(base_dir, "a_overlap")
        )
        storage_a.store_object(obj_a_unique_1, os.path.join(base_dir, "a_unique_1"))
        storage_a.store_object(obj_a_unique_2, os.path.join(base_dir, "a_unique_2"))

        # Source storage B has three entries. One entry overlaps A by query
        # identity and should replace the overlap in A because it has lower
        # statistical inefficiency (more informative data).
        obj_b_overlap = _create_stored_data(
            factory,
            os.path.join(base_dir, "b_overlap"),
            substance_a,
            calculation_id="shared-id",
            statistical_inefficiency=2.5 - 1e-6,
        )
        obj_b_unique_1 = _create_stored_data(
            factory,
            os.path.join(base_dir, "b_unique_1"),
            substance_b,
            calculation_id="b-only-1",
            statistical_inefficiency=1.4,
        )
        obj_b_unique_2 = _create_stored_data(
            factory,
            os.path.join(base_dir, "b_unique_2"),
            substance_c,
            calculation_id="b-only-2",
            statistical_inefficiency=1.6,
        )

        storage_b.store_object(obj_b_overlap, os.path.join(base_dir, "b_overlap"))
        storage_b.store_object(obj_b_unique_1, os.path.join(base_dir, "b_unique_1"))
        storage_b.store_object(obj_b_unique_2, os.path.join(base_dir, "b_unique_2"))

        # Merge B into A.
        storage_a.update(storage_b)

        # 3 in A + 3 in B - 1 overlap = 5 total.
        merged_keys = storage_a._stored_object_keys[stored_data_class.__name__]
        assert len(merged_keys) == 5

        # The overlap key remains, but its payload should now match the better
        # (lower statistical inefficiency) overlapping object from B.
        replaced_overlap, _ = storage_a.retrieve_object(overlap_key)
        assert replaced_overlap.statistical_inefficiency < 2.5


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


def _make_storage_with_substances(base_dir, factory, *substances_or_smiles):
    """Create a MutableLocalFileStorage with one data object per substance.

    Each entry may be a SMILES string (converted to a single-component
    Substance) or a pre-built Substance object.
    """
    storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
    for i, s in enumerate(substances_or_smiles):
        substance = Substance.from_components(s) if isinstance(s, str) else s
        data_dir = os.path.join(base_dir, f"data_{i}")
        obj = factory(data_dir, substance)
        storage.store_object(obj, data_dir)
    return storage


def _data_keys(storage):
    """All stored-object keys excluding ForceFieldData."""
    return [
        key
        for type_name, keys in storage._stored_object_keys.items()
        if type_name != ForceFieldData.__name__
        for key in keys
    ]


@pytest.fixture
def substance_c():
    return Substance.from_components("C")


@pytest.fixture
def substance_o():
    return Substance.from_components("O")


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_subset_include_substance(factory, substance_c):
    """Only the object matching include_substances is in the result."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = _make_storage_with_substances(base_dir, factory, "C", "O", "CC")
        result_dir = os.path.join(base_dir, "result")

        result = storage.subset(result_dir, include_substances=[substance_c])
        keys = _data_keys(result)
        assert len(keys) == 1
        obj, _ = result.retrieve_object(keys[0])
        assert obj.substance == substance_c


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_subset_include_component(factory, substance_c):
    """Only objects whose substance contains the component are included."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = _make_storage_with_substances(base_dir, factory, "C", "O")
        result_dir = os.path.join(base_dir, "result")

        result = storage.subset(result_dir, include_components=[Component("C")])

        keys = _data_keys(result)
        assert len(keys) == 1
        obj, _ = result.retrieve_object(keys[0])
        assert obj.substance == substance_c


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_subset_exclude_substance(factory, substance_c, substance_o):
    """Object matching exclude_substances is absent from the result."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = _make_storage_with_substances(base_dir, factory, "C", "O")
        result_dir = os.path.join(base_dir, "result")

        result = storage.subset(result_dir, exclude_substances=[substance_o])
        keys = _data_keys(result)
        assert len(keys) == 1
        obj, _ = result.retrieve_object(keys[0])
        assert obj.substance == substance_c


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_subset_exclude_component(factory, substance_c):
    """Object whose substance contains the excluded component is absent."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = _make_storage_with_substances(base_dir, factory, "C", "O")
        result_dir = os.path.join(base_dir, "result")

        result = storage.subset(result_dir, exclude_components=[Component("O")])

        keys = _data_keys(result)
        assert len(keys) == 1
        obj, _ = result.retrieve_object(keys[0])
        assert obj.substance == substance_c


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_subset_combined_filters(factory, substance_c):
    """include_components, exclude_substances, and exclude_components each
    eliminate a distinct subset.

    Storage (7 objects):
      pure_c, pure_o, pure_cc,
      mix(C, O),            ← methane + water
      mix(CC, O),           ← ethane + water
      mix(C, CC),           ← methane + ethane
      mix(C, O, CC)         ← methane + water + ethane

    1. include_components=[Component("C")]
          keeps substances whose components include methane:
          pure_c, mix(C,O), mix(C,CC), mix(C,O,CC)
          removes: pure_o, pure_cc, mix(CC,O)

    2. exclude_substances=[mix(C,O,CC)]
          removes the three-component mixture exactly
          remaining: pure_c, mix(C,O), mix(C,CC)

    3. exclude_components=[Component("O")]
          removes substances that contain the water component:
          mix(C,O) is removed
          remaining: pure_c, mix(C,CC)

    Expected survivors: pure_c (methane) and mix(C,CC) (methane+ethane).
    """
    mix_c_cc = Substance.from_components("C", "CC")
    mix_c_o = Substance.from_components("C", "O")
    mix_c_o_cc = Substance.from_components("C", "O", "CC")

    with tempfile.TemporaryDirectory() as base_dir:
        storage = _make_storage_with_substances(
            base_dir,
            factory,
            "C",
            "O",
            "CC",
            mix_c_o,
            Substance.from_components("CC", "O"),
            mix_c_cc,
            mix_c_o_cc,
        )
        result_dir = os.path.join(base_dir, "result")

        result = storage.subset(
            result_dir,
            include_components=[Component("C")],
            exclude_substances=[mix_c_o_cc],
            exclude_components=[Component("O")],
        )

        keys = _data_keys(result)
        assert len(keys) == 2
        substances = {result.retrieve_object(k)[0].substance for k in keys}
        assert substances == {substance_c, mix_c_cc}


def test_subset_raises_on_overlapping_substances(substance_c):
    """ValueError if the same substance is in both include and exclude."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        with pytest.raises(ValueError, match="substance"):
            storage.subset(
                os.path.join(base_dir, "result"),
                include_substances=[substance_c],
                exclude_substances=[substance_c],
            )


def test_subset_raises_on_overlapping_components():
    """ValueError if the same component is in both include and exclude."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        with pytest.raises(ValueError, match="component"):
            storage.subset(
                os.path.join(base_dir, "result"),
                include_components=[Component("C")],
                exclude_components=[Component("C")],
            )


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_subset_no_filters_returns_all(factory):
    """subset() with no filters returns an identical copy of the storage."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = _make_storage_with_substances(base_dir, factory, "C", "O")
        result_dir = os.path.join(base_dir, "result")

        result = storage.subset(result_dir)
        assert len(_data_keys(result)) == 2


def test_subset_force_field_data_included_when_no_substance_filter():
    """ForceFieldData is included when no include filters are active."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        storage.store_force_field(
            SmirnoffForceFieldSource.from_path("openff-2.2.1.offxml")
        )

        result = storage.subset(os.path.join(base_dir, "result"))

        assert len(result._stored_object_keys.get(ForceFieldData.__name__, [])) == 1


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_subset_force_field_excluded_when_include_substance_active(
    factory, substance_c
):
    """ForceFieldData is excluded when an include_substances filter is active."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        storage.store_force_field(
            SmirnoffForceFieldSource.from_path("openff-2.2.1.offxml")
        )
        data_dir = os.path.join(base_dir, "data")
        storage.store_object(factory(data_dir, substance_c), data_dir)

        result = storage.subset(
            os.path.join(base_dir, "result"), include_substances=[substance_c]
        )

        # ForceFieldData has no substance, so it cannot match the include filter
        assert len(result._stored_object_keys.get(ForceFieldData.__name__, [])) == 0
        # But the data object is present
        assert len(_data_keys(result)) == 1


def test_remove_object_deletes_json_and_directory():
    """remove_object() deletes both the JSON file and the ancillary directory."""
    substance = Substance.from_components("C")
    with tempfile.TemporaryDirectory() as base_dir:
        storage_dir = os.path.join(base_dir, "storage")
        data_dir = os.path.join(base_dir, "data")

        obj = create_dummy_simulation_data(data_dir, substance)
        storage = MutableLocalFileStorage(storage_dir)
        key = storage.store_object(obj, data_dir)

        json_path = os.path.join(storage_dir, f"{key}.json")
        ancillary_path = os.path.join(storage_dir, key)
        assert os.path.isfile(json_path)
        assert os.path.isdir(ancillary_path)

        storage.remove_object(key)

        assert not os.path.isfile(json_path)
        assert not os.path.isdir(ancillary_path)


def test_remove_object_removes_from_registry():
    """Key is absent from _stored_object_keys after removal."""
    substance = Substance.from_components("C")
    with tempfile.TemporaryDirectory() as base_dir:
        storage_dir = os.path.join(base_dir, "storage")
        data_dir = os.path.join(base_dir, "data")

        obj = create_dummy_simulation_data(data_dir, substance)
        storage = MutableLocalFileStorage(storage_dir)
        key = storage.store_object(obj, data_dir)

        assert key in storage._stored_object_keys[StoredSimulationData.__name__]
        storage.remove_object(key)
        assert key not in storage._stored_object_keys.get(StoredSimulationData.__name__, [])


def test_remove_object_clears_hash_index():
    """For HashableStoredData, the corresponding hash entry is removed."""
    ff = SmirnoffForceFieldSource.from_path("openff-2.2.1.offxml")
    with tempfile.TemporaryDirectory() as base_dir:
        storage_dir = os.path.join(base_dir, "storage")
        storage = MutableLocalFileStorage(storage_dir)
        key = storage.store_force_field(ff)

        assert any(k == key for k in storage._object_hashes.values())
        storage.remove_object(key)
        assert not any(k == key for k in storage._object_hashes.values())


def test_remove_object_clears_cache():
    """With cache_objects_in_memory=True, the cache entry is evicted."""
    substance = Substance.from_components("C")
    with tempfile.TemporaryDirectory() as base_dir:
        storage_dir = os.path.join(base_dir, "storage")
        data_dir = os.path.join(base_dir, "data")

        obj = create_dummy_simulation_data(data_dir, substance)
        storage = MutableLocalFileStorage(storage_dir, cache_objects_in_memory=True)
        key = storage.store_object(obj, data_dir)

        # Warm the cache
        storage.retrieve_object(key)
        assert key in storage._cached_retrieved_objects

        storage.remove_object(key)
        assert key not in storage._cached_retrieved_objects


def test_remove_object_key_not_found_raises():
    """remove_object raises KeyError for an unregistered key."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        with pytest.raises(KeyError):
            storage.remove_object("nonexistent_key")


def test_remove_object_then_store_again():
    """After removal an object can be re-stored and retrieved correctly."""
    substance = Substance.from_components("C")
    with tempfile.TemporaryDirectory() as base_dir:
        storage_dir = os.path.join(base_dir, "storage")
        data_dir_1 = os.path.join(base_dir, "data1")
        data_dir_2 = os.path.join(base_dir, "data2")

        import uuid as _uuid

        obj1 = create_dummy_simulation_data(
            data_dir_1, substance, calculation_id=str(_uuid.uuid4())
        )
        storage = MutableLocalFileStorage(storage_dir)
        key1 = storage.store_object(obj1, data_dir_1)

        storage.remove_object(key1)

        # Re-store with a fresh ancillary directory
        obj2 = create_dummy_simulation_data(
            data_dir_2, substance, calculation_id=str(_uuid.uuid4())
        )
        key2 = storage.store_object(obj2, data_dir_2)

        retrieved, _ = storage.retrieve_object(key2)
        assert retrieved is not None
        assert retrieved.substance == substance


def test_remove_object_only_affects_target():
    """Removing one object does not disturb other stored objects."""
    substance_c = Substance.from_components("C")
    substance_o = Substance.from_components("O")

    with tempfile.TemporaryDirectory() as base_dir:
        storage_dir = os.path.join(base_dir, "storage")
        data_c = os.path.join(base_dir, "data_c")
        data_o = os.path.join(base_dir, "data_o")

        obj_c = create_dummy_simulation_data(data_c, substance_c)
        obj_o = create_dummy_simulation_data(data_o, substance_o)

        storage = MutableLocalFileStorage(storage_dir)
        key_c = storage.store_object(obj_c, data_c)
        key_o = storage.store_object(obj_o, data_o)

        storage.remove_object(key_c)

        retrieved_o, _ = storage.retrieve_object(key_o)
        assert retrieved_o is not None
        assert retrieved_o.substance == substance_o
