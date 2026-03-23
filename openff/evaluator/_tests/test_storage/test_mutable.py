import os
import tempfile
import uuid

import pytest

from openff.evaluator._tests.utils import (
    create_dummy_equilibration_data,
    create_dummy_simulation_data,
)
from openff.evaluator.storage import MutableLocalFileStorage
from openff.evaluator.substances import Component, MoleFraction, Substance

DATA_FACTORIES = [
    pytest.param(create_dummy_simulation_data, id="simulation"),
    pytest.param(create_dummy_equilibration_data, id="equilibration"),
]


def _make_mixture(*smiles_list):
    """Build an equimolar Substance from two or more SMILES."""
    substance = Substance()
    mf = MoleFraction(1.0 / len(smiles_list))
    for smi in smiles_list:
        substance.add_component(Component(smi), mf)
    return substance


def _store(base_dir, storage, substance, tag, factory):
    """Store one data object for *substance* and return the substance."""
    data_dir = os.path.join(base_dir, f"data_{tag}")
    obj = factory(data_dir, substance)
    storage.store_object(obj, data_dir)
    return substance


@pytest.fixture
def methane_component():
    return Component("C")


@pytest.fixture
def substance_c():
    return Substance.from_components("C")


@pytest.fixture
def substance_o():
    return Substance.from_components("O")


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_retrieve_by_substance_exact_match(factory, substance_c, substance_o):
    """retrieve_by_substance returns a dict with one key for an exact match."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        _store(base_dir, storage, substance_c, "c", factory)
        _store(base_dir, storage, substance_o, "o", factory)

        result = storage.retrieve_by_substance(substance_c)

        assert len(result) == 1
        assert substance_c in result
        entries = result[substance_c]
        assert len(entries) == 1
        key, obj, _ = entries[0]
        assert isinstance(key, str)
        assert obj.substance == substance_c


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_retrieve_by_substance_no_match(factory, substance_c):
    """retrieve_by_substance returns an empty dict when nothing matches."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        _store(base_dir, storage, substance_c, "c", factory)

        result = storage.retrieve_by_substance(Substance.from_components("CC"))
        assert result == {}


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_retrieve_by_substance_multiple_objects_same_substance(factory, substance_c):
    """Multiple objects with the same substance are all returned under one key."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        for i in range(2):
            data_dir = os.path.join(base_dir, f"data_{i}")
            obj = factory(data_dir, substance_c, calculation_id=str(uuid.uuid4()))
            storage.store_object(obj, data_dir)

        result = storage.retrieve_by_substance(substance_c)
        assert len(result) == 1
        assert len(result[substance_c]) == 2


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_retrieve_by_substance_multicomponent(factory, substance_c):
    """retrieve_by_substance finds a multi-component mixture by exact substance."""
    mixture = _make_mixture("C", "O")
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        _store(base_dir, storage, mixture, "mixture", factory)
        _store(base_dir, storage, substance_c, "c", factory)

        result = storage.retrieve_by_substance(mixture)

        assert len(result) == 1
        assert mixture in result
        assert result[mixture][0][1].substance == mixture


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_retrieve_by_substance_does_not_match_mixture(factory, substance_c):
    """Searching for a pure substance does not return a mixture containing that component."""
    mixture = _make_mixture("C", "O")
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        _store(base_dir, storage, mixture, "mixture", factory)

        result = storage.retrieve_by_substance(substance_c)
        assert result == {}


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_retrieve_by_component_present(factory, methane_component, substance_c, substance_o):
    """retrieve_by_component returns the substance whose component matches."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        _store(base_dir, storage, substance_c, "c", factory)
        _store(base_dir, storage, substance_o, "o", factory)

        result = storage.retrieve_by_component(methane_component)

        assert len(result) == 1
        assert substance_c in result
        assert len(result[substance_c]) == 1


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_retrieve_by_component_not_present(factory, methane_component, substance_c):
    """retrieve_by_component returns empty dict when component is absent."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        _store(base_dir, storage, substance_c, "c", factory)

        unknown = Component("CC")
        result = storage.retrieve_by_component(unknown)
        assert result == {}


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_retrieve_by_component_finds_mixture(factory, methane_component, substance_o):
    """retrieve_by_component returns a multi-component substance containing the component."""
    mixture = _make_mixture("C", "O")
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        _store(base_dir, storage, mixture, "mixture", factory)
        _store(base_dir, storage, substance_o, "o_pure", factory)

        result = storage.retrieve_by_component(methane_component)

        assert mixture in result
        assert len(result[mixture]) == 1


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_retrieve_by_component_shared_between_pure_and_mixture(
    factory, methane_component, substance_c, substance_o
):
    """A component shared by a pure substance and a mixture produces two separate keys."""
    mixture_co = _make_mixture("C", "O")

    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        _store(base_dir, storage, substance_c, "pure_c", factory)
        _store(base_dir, storage, mixture_co, "mixture_co", factory)
        _store(base_dir, storage, substance_o, "pure_o", factory)

        result = storage.retrieve_by_component(methane_component)

        assert substance_c in result
        assert mixture_co in result
        assert len(result) == 2
        assert len(result[substance_c]) == 1
        assert len(result[mixture_co]) == 1


@pytest.mark.parametrize("factory", DATA_FACTORIES)
def test_retrieve_by_component_multi_substance(factory, methane_component, substance_c, substance_o):
    """retrieve_by_component excludes substances that do not contain the component."""
    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        _store(base_dir, storage, substance_c, "c", factory)
        _store(base_dir, storage, substance_o, "o", factory)

        result = storage.retrieve_by_component(methane_component)

        assert substance_c in result
        assert substance_o not in result


def test_retrieve_by_component_ignores_non_substance_objects(methane_component):
    """ForceFieldData (no substance attribute) is silently skipped."""
    from openff.evaluator.forcefield import SmirnoffForceFieldSource

    with tempfile.TemporaryDirectory() as base_dir:
        storage = MutableLocalFileStorage(os.path.join(base_dir, "storage"))
        ff = SmirnoffForceFieldSource.from_path("openff-2.2.1.offxml")
        storage.store_force_field(ff)

        result = storage.retrieve_by_component(methane_component)
        assert result == {}
