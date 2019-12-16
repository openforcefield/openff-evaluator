"""
Units tests for propertyestimator.storage
"""
import os
import tempfile

import pytest

from propertyestimator.forcefield import SmirnoffForceFieldSource
from propertyestimator.storage import LocalFileStorage
from propertyestimator.storage.attributes import QueryAttribute, StorageAttribute
from propertyestimator.storage.data import (
    BaseStoredData,
    HashableStoredData,
    StoredSimulationData,
)
from propertyestimator.storage.query import (
    BaseDataQuery,
    SimulationDataQuery,
    SubstanceQuery,
)
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import create_dummy_simulation_data


class SimpleData(BaseStoredData):

    some_attribute = StorageAttribute(docstring="", type_hint=int)

    @classmethod
    def has_ancillary_data(cls):
        return False

    def to_storage_query(self):
        return SimpleDataQuery.from_data_object(self)


class SimpleDataQuery(BaseDataQuery):
    @classmethod
    def data_class(cls):
        return SimpleData

    some_attribute = QueryAttribute(docstring="", type_hint=int)


class HashableData(HashableStoredData):

    some_attribute = StorageAttribute(docstring="", type_hint=int)

    @classmethod
    def has_ancillary_data(cls):
        return False

    def to_storage_query(self):
        raise NotImplementedError()

    def __hash__(self):
        return hash(self.some_attribute)


@pytest.mark.parametrize("data_class", [SimpleData, HashableData])
def test_simple_store_and_retrieve(data_class):
    """Tests storing and retrieving a simple object.
    """
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


def test_simulation_data_storage():

    substance = Substance.from_components("C")

    with tempfile.TemporaryDirectory() as base_directory:

        data_directory = os.path.join(base_directory, "data_directory")
        data_object = create_dummy_simulation_data(data_directory, substance)

        backend_directory = os.path.join(base_directory, "storage_dir")

        storage = LocalFileStorage(backend_directory)
        storage_key = storage.store_object(data_object, data_directory)

        assert storage.has_object(data_object)
        assert storage_key == storage.store_object(data_object, data_directory)

        retrieved_object, retrieved_directory = storage.retrieve_object(
            storage_key, StoredSimulationData
        )

        assert backend_directory in retrieved_directory
        assert data_object.json() == retrieved_object.json()


def test_simulation_data_query():

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


def test_duplicate_simulation_data_storage():
    # TODO: Implement.
    pytest.skip("Not yet implemented.")

    # with tempfile.TemporaryDirectory() as base_directory_path:
    #
    # dummy_substance = create_dummy_substance(1)
    #
    # dummy_data_path_1 = os.path.join(base_directory_path, "data_1")
    # data_1_coordinate_name = "data_1.pdb"
    #
    # dummy_data_1 = create_dummy_simulation_data(
    #     directory_path=dummy_data_path_1,
    #     substance=dummy_substance,
    #     force_field_id="ff_id_1",
    #     coordinate_file_name=data_1_coordinate_name,
    #     statistical_inefficiency=1.0,
    # )
    #
    # dummy_data_path_2 = os.path.join(base_directory_path, "data_2")
    # data_2_coordinate_name = "data_2.pdb"
    #
    # dummy_data_2 = create_dummy_simulation_data(
    #     directory_path=dummy_data_path_2,
    #     substance=dummy_substance,
    #     force_field_id="ff_id_1",
    #     coordinate_file_name=data_2_coordinate_name,
    #     statistical_inefficiency=2.0,
    # )
    #
    # storage_directory = os.path.join(base_directory_path, "storage")
    # local_storage = LocalFileStorage(storage_directory)
    #
    # local_storage.store_simulation_data(dummy_data_1, dummy_data_path_1)
    # local_storage.store_simulation_data(dummy_data_2, dummy_data_path_2)
    #
    # stored_data = local_storage.retrieve_simulation_data(dummy_substance)
    #
    # assert len(stored_data[dummy_substance.identifier]) == 1
    #
    # # Make sure the correct data got retained.
    # stored_data_object, stored_data_directory = stored_data[
    #     dummy_substance.identifier
    # ][0]
    # assert stored_data_object.coordinate_file_name == data_2_coordinate_name
    #
    # dummy_data_path_3 = os.path.join(base_directory_path, "data_3")
    # data_3_coordinate_name = "data_3.pdb"
    #
    # dummy_data_3 = create_dummy_simulation_data(
    #     directory_path=dummy_data_path_3,
    #     substance=dummy_substance,
    #     force_field_id="ff_id_2",
    #     coordinate_file_name=data_3_coordinate_name,
    #     statistical_inefficiency=3.0,
    # )
    #
    # local_storage.store_simulation_data(dummy_data_3, dummy_data_path_3)
    #
    # stored_data = local_storage.retrieve_simulation_data(dummy_substance)
    # assert len(stored_data[dummy_substance.identifier]) == 2
    #
    # # Make sure the correct data got retained.
    # stored_data_object, stored_data_directory = stored_data[
    #     dummy_substance.identifier
    # ][1]
    # assert stored_data_object.coordinate_file_name == data_3_coordinate_name
