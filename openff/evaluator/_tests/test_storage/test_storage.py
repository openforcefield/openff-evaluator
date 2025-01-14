"""
Units tests for openff.evaluator.storage
"""

import os
import tempfile

import pytest

from openff.evaluator._tests.test_storage.data import HashableData, SimpleData
from openff.evaluator._tests.utils import create_dummy_simulation_data, create_dummy_equilibration_data
from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.storage import LocalFileStorage
from openff.evaluator.storage.data import StoredSimulationData, StoredEquilibrationData
from openff.evaluator.storage.query import SimulationDataQuery, SubstanceQuery, EquilibrationDataQuery
from openff.evaluator.substances import Substance


@pytest.mark.parametrize("data_class", [SimpleData, HashableData])
def test_simple_store_and_retrieve(data_class):
    """Tests storing and retrieving a simple object."""
    with tempfile.TemporaryDirectory() as temporary_directory:
        local_storage = LocalFileStorage(temporary_directory)

        storage_object = data_class()

        # Make sure the validation fails
        with pytest.raises(ValueError):
            local_storage.store_object(storage_object)

        # This should now pass.
        storage_object.some_attribute = 10

        storage_key = local_storage.store_object(storage_object)
        assert local_storage.has_object(storage_object)

        retrieved_object, _ = local_storage.retrieve_object(storage_key)

        assert retrieved_object is not None
        assert storage_object.json() == retrieved_object.json()

        # Ensure that the same key is returned when storing duplicate
        # data
        new_storage_key = local_storage.store_object(storage_object)
        assert storage_key == new_storage_key


def test_force_field_storage():
    """A simple test to that force fields can be stored and
    retrieved using the local storage backend."""

    force_field_source = SmirnoffForceFieldSource.from_path(
        "smirnoff99Frosst-1.1.0.offxml"
    )

    with tempfile.TemporaryDirectory() as temporary_directory:
        local_storage = LocalFileStorage(temporary_directory)

        force_field_id = local_storage.store_force_field(force_field_source)
        retrieved_force_field = local_storage.retrieve_force_field(force_field_id)

        assert force_field_source.json() == retrieved_force_field.json()

        local_storage_new = LocalFileStorage(temporary_directory)
        assert local_storage_new.has_force_field(force_field_source)

        new_force_field_id = local_storage_new.store_force_field(force_field_source)
        assert new_force_field_id == force_field_id


@pytest.mark.parametrize("data_types", [
    {"data_class": StoredSimulationData, "data_function": create_dummy_simulation_data},
    {"data_class": StoredEquilibrationData, "data_function": create_dummy_equilibration_data},
])
def test_base_simulation_data_storage(data_types):
    substance = Substance.from_components("C")

    with tempfile.TemporaryDirectory() as base_directory:
        data_directory = os.path.join(base_directory, "data_directory")
        data_object = data_types["data_function"](data_directory, substance)

        backend_directory = os.path.join(base_directory, "storage_dir")

        storage = LocalFileStorage(backend_directory)
        storage_key = storage.store_object(data_object, data_directory)

        # Regenerate the data directory.
        os.makedirs(data_directory, exist_ok=True)

        assert storage.has_object(data_object)
        assert storage_key == storage.store_object(data_object, data_directory)

        retrieved_object, retrieved_directory = storage.retrieve_object(
            storage_key, data_types["data_class"]
        )

        assert backend_directory in retrieved_directory
        assert data_object.json() == retrieved_object.json()


def test_base_simulation_data_query():
    substance_a = Substance.from_components("C")
    substance_b = Substance.from_components("CO")

    substance_full = Substance.from_components("C", "CO")

    substances = [substance_a, substance_b, substance_full]

    with tempfile.TemporaryDirectory() as base_directory:
        backend_directory = os.path.join(base_directory, "storage_dir")
        storage = LocalFileStorage(backend_directory)

        for substance in substances:
            data_directory = os.path.join(base_directory, f"{substance.identifier}")
            data_object = create_dummy_simulation_data(data_directory, substance)

            storage.store_object(data_object, data_directory)

        for substance in substances:
            substance_query = SimulationDataQuery()
            substance_query.substance = substance

            results = storage.query(substance_query)
            assert results is not None and len(results) == 1
            assert len(next(iter(results.values()))[0]) == 3

        component_query = SimulationDataQuery()
        component_query.substance = substance_full
        component_query.substance_query = SubstanceQuery()
        component_query.substance_query.components_only = True

        results = storage.query(component_query)
        assert results is not None and len(results) == 2


@pytest.mark.parametrize(
    "data_generation_function",
    [create_dummy_simulation_data, create_dummy_equilibration_data]
)
@pytest.mark.parametrize("reverse_order", [True, False])
def test_duplicate_simulation_data_storage(data_generation_function, reverse_order):
    substance = Substance.from_components("CO")

    with tempfile.TemporaryDirectory() as base_directory_path:
        storage_directory = os.path.join(base_directory_path, "storage")
        local_storage = LocalFileStorage(storage_directory)

        # Construct some data to store with increasing
        # statistical inefficiencies.
        data_to_store = []

        for index in range(3):
            data_directory = os.path.join(base_directory_path, f"data_{index}")
            coordinate_name = f"data_{index}.pdb"

            data_object = data_generation_function(
                directory_path=data_directory,
                substance=substance,
                force_field_id="ff_id_1",
                coordinate_file_name=coordinate_name,
                statistical_inefficiency=float(index),
                calculation_id="id",
            )
            data_to_store.append((data_object, data_directory))

        # Keep a track of the storage keys.
        all_storage_keys = set()

        iterator = enumerate(data_to_store)

        if reverse_order:
            iterator = reversed(list(iterator))

        # Store the data
        for index, data in iterator:
            data_object, data_directory = data

            storage_key = local_storage.store_object(data_object, data_directory)
            all_storage_keys.add(storage_key)

            retrieved_object, stored_directory = local_storage.retrieve_object(
                storage_key
            )

            # Handle the case where we haven't reversed the order of
            # the data to store. Here only the first object in the list
            # should be stored an never replaced as it has the lowest
            # statistical inefficiency.
            if not reverse_order:
                expected_index = 0
            # Handle the case where we have reversed the order of
            # the data to store. Here only the each new piece of
            # data should replace the last, as it will have a lower
            # statistical inefficiency.
            else:
                expected_index = index

            assert retrieved_object.json() == data_to_store[expected_index][0].json()

            # Make sure the directory has been correctly overwritten / retained
            # depending on the data order.
            coordinate_path = os.path.join(
                stored_directory, f"data_{expected_index}.pdb"
            )
            assert os.path.isfile(coordinate_path)

        # Make sure all pieces of data got assigned the same key if
        # reverse order.
        assert len(all_storage_keys) == 1
