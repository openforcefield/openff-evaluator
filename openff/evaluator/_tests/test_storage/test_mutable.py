"""Unit tests for MutableLocalFileStorage — search/retrieve by substance/component."""
import os
import tempfile

from openff.evaluator._tests.utils import create_dummy_simulation_data
from openff.evaluator.storage import MutableLocalFileStorage
from openff.evaluator.substances import Substance


def _store_substance(base_dir, storage, smiles, tag):
    """Helper: store one SimulationData for *smiles* and return the substance."""
    substance = Substance.from_components(smiles)
    data_dir = os.path.join(base_dir, f"data_{tag}")
    obj = create_dummy_simulation_data(data_dir, substance)
    storage.store_object(obj, data_dir)
    return substance


def test_retrieve_by_substance_exact_match():
    """retrieve_by_substance returns a dict with one key for an exact match."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        substance_c = _store_substance(base_dir, storage, "C", "c")
        _store_substance(base_dir, storage, "O", "o")

        result = storage.retrieve_by_substance(substance_c)

        assert len(result) == 1
        assert substance_c in result
        entries = result[substance_c]
        assert len(entries) == 1
        key, obj, _ = entries[0]
        assert isinstance(key, str)
        assert obj.substance == substance_c


def test_retrieve_by_substance_no_match():
    """retrieve_by_substance returns an empty dict when nothing matches."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        _store_substance(base_dir, storage, "C", "c")

        unknown = Substance.from_components("CC")
        result = storage.retrieve_by_substance(unknown)
        assert result == {}


def test_retrieve_by_substance_multiple_objects_same_substance():
    """Multiple objects with the same substance are all returned under one key."""
    substance = Substance.from_components("C")

    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        # Store two objects with the same substance but different calculation IDs
        import uuid
        for i in range(2):
            data_dir = os.path.join(base_dir, f"data_{i}")
            obj = create_dummy_simulation_data(
                data_dir, substance, calculation_id=str(uuid.uuid4())
            )
            storage.store_object(obj, data_dir)

        result = storage.retrieve_by_substance(substance)
        assert len(result) == 1
        assert len(result[substance]) == 2


def test_retrieve_by_component_present():
    """retrieve_by_component groups objects by substance."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        substance_c = _store_substance(base_dir, storage, "C", "c")
        _store_substance(base_dir, storage, "O", "o")

        component = list(Substance.from_components("C").components)[0]
        result = storage.retrieve_by_component(component)

        assert len(result) == 1
        assert substance_c in result
        assert len(result[substance_c]) == 1


def test_retrieve_by_component_not_present():
    """retrieve_by_component returns empty dict when component absent."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        _store_substance(base_dir, storage, "C", "c")

        unknown_component = list(Substance.from_components("CC").components)[0]
        result = storage.retrieve_by_component(unknown_component)
        assert result == {}


def test_retrieve_by_component_multi_substance():
    """retrieve_by_component groups results across multiple matching substances."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        substance_c = _store_substance(base_dir, storage, "C", "c")
        substance_o = _store_substance(base_dir, storage, "O", "o")

        # Search for the methane component — only substance_c should match
        methane_component = list(Substance.from_components("C").components)[0]
        result = storage.retrieve_by_component(methane_component)

        assert substance_c in result
        assert substance_o not in result


def test_retrieve_by_component_ignores_non_substance_objects():
    """ForceFieldData (no substance attribute) is silently skipped."""
    from openff.evaluator.forcefield import SmirnoffForceFieldSource

    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        ff = SmirnoffForceFieldSource.from_path("openff-2.2.1.offxml")
        storage.store_force_field(ff)

        component = list(Substance.from_components("C").components)[0]
        result = storage.retrieve_by_component(component)
        assert result == {}
