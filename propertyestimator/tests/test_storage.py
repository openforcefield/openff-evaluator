"""
Units tests for propertyestimator.storage
"""
import os
import tempfile

from propertyestimator import unit
from propertyestimator.storage import LocalFileStorage, StoredSimulationData
from propertyestimator.storage.dataclasses import BaseStoredData, StoredDataCollection
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import create_dummy_stored_simulation_data, create_dummy_substance
from propertyestimator.thermodynamics import ThermodynamicState


class DummyDataClass1(BaseStoredData):

    @classmethod
    def merge(cls, stored_data_1, stored_data_2):
        return stored_data_1


class DummyDataClass2(BaseStoredData):

    @classmethod
    def merge(cls, stored_data_1, stored_data_2):
        return stored_data_1


def test_local_force_field_storage():
    """A simple test to that force fields can be stored and
    retrieved using the local storage backend."""

    from openforcefield.typing.engines import smirnoff
    force_field = smirnoff.ForceField('smirnoff99Frosst-1.1.0.offxml')

    with tempfile.TemporaryDirectory() as temporary_directory:

        local_storage = LocalFileStorage(temporary_directory)

        force_field_id = local_storage.store_force_field(force_field)
        retrieved_force_field = local_storage.retrieve_force_field(force_field_id)

        force_field_string = force_field.to_string()
        retrieved_force_field_string = retrieved_force_field.to_string()

        assert force_field_string == retrieved_force_field_string

        local_storage_new = LocalFileStorage(temporary_directory)
        assert local_storage_new.has_force_field(force_field)


def test_local_simulation_storage():
    """A simple test to that force fields can be stored and
    retrieved using the local storage backend."""

    substance = Substance()
    substance.add_component(Substance.Component(smiles='C'),
                            Substance.MoleFraction())

    dummy_simulation_data = StoredSimulationData()

    dummy_simulation_data.thermodynamic_state = ThermodynamicState(298.0*unit.kelvin,
                                                                   1.0*unit.atmosphere)

    dummy_simulation_data.statistical_inefficiency = 1.0
    dummy_simulation_data.force_field_id = 'tmp_ff_id'

    dummy_simulation_data.substance = substance

    with tempfile.TemporaryDirectory() as base_directory:

        temporary_data_directory = os.path.join(base_directory, 'temp_data')
        temporary_backend_directory = os.path.join(base_directory, 'storage_dir')

        os.makedirs(temporary_data_directory)
        os.makedirs(temporary_backend_directory)

        local_storage = LocalFileStorage(temporary_backend_directory)
        dummy_simulation_data.unique_id = local_storage.store_simulation_data(dummy_simulation_data,
                                                                              temporary_data_directory)

        retrieved_data_directories = local_storage.retrieve_simulation_data(substance)

        assert substance.identifier in retrieved_data_directories
        assert len(retrieved_data_directories[substance.identifier]) == 1

        retrieved_data, retrieved_data_directory = retrieved_data_directories[substance.identifier][0]

        assert dummy_simulation_data.thermodynamic_state == retrieved_data.thermodynamic_state
        assert dummy_simulation_data.statistical_inefficiency == retrieved_data.statistical_inefficiency
        assert dummy_simulation_data.force_field_id == retrieved_data.force_field_id
        assert dummy_simulation_data.substance == retrieved_data.substance

        local_storage_new = LocalFileStorage(temporary_backend_directory)

        retrieved_data, retrieved_data_directory = local_storage_new.retrieve_simulation_data_by_id(
            dummy_simulation_data.unique_id)

        assert retrieved_data is not None
        assert os.path.isdir(retrieved_data_directory)


def test_data_class_retrieval():
    """A simple test to that force fields can be stored and
    retrieved using the local storage backend."""

    substance = Substance()
    substance.add_component(Substance.Component('C'), Substance.MoleFraction(1.0))

    with tempfile.TemporaryDirectory() as base_directory_path:

        storage_directory = os.path.join(base_directory_path, 'storage')
        local_storage = LocalFileStorage(storage_directory)

        for data_class_type in [DummyDataClass1, DummyDataClass2]:

            data_directory = os.path.join(base_directory_path, data_class_type.__name__)
            os.makedirs(data_directory, exist_ok=True)

            data_object = data_class_type()
            data_object.substance = substance

            local_storage.store_simulation_data(data_object, data_directory)

        retrieved_data_directories = local_storage.retrieve_simulation_data(substance, data_class=BaseStoredData)
        assert len(retrieved_data_directories[substance.identifier]) == 2

        retrieved_data_directories = local_storage.retrieve_simulation_data(substance, data_class=DummyDataClass1)
        assert len(retrieved_data_directories[substance.identifier]) == 1

        retrieved_data_directories = local_storage.retrieve_simulation_data(substance, data_class=DummyDataClass2)
        assert len(retrieved_data_directories[substance.identifier]) == 1

        retrieved_data_directories = local_storage.retrieve_simulation_data(substance, data_class=StoredSimulationData)
        assert len(retrieved_data_directories[substance.identifier]) == 0


def test_simulation_data_merging():
    """A test that compatible simulation data gets merged
    together within the`LocalStorage` system."""

    with tempfile.TemporaryDirectory() as base_directory_path:

        dummy_substance = create_dummy_substance(1)

        dummy_data_path_1 = os.path.join(base_directory_path, 'data_1')
        data_1_coordinate_name = 'data_1.pdb'

        dummy_data_1 = create_dummy_stored_simulation_data(directory_path=dummy_data_path_1,
                                                           substance=dummy_substance,
                                                           force_field_id='ff_id_1',
                                                           coordinate_file_name=data_1_coordinate_name,
                                                           statistical_inefficiency=1.0)

        dummy_data_path_2 = os.path.join(base_directory_path, 'data_2')
        data_2_coordinate_name = 'data_2.pdb'

        dummy_data_2 = create_dummy_stored_simulation_data(directory_path=dummy_data_path_2,
                                                           substance=dummy_substance,
                                                           force_field_id='ff_id_1',
                                                           coordinate_file_name=data_2_coordinate_name,
                                                           statistical_inefficiency=2.0)

        storage_directory = os.path.join(base_directory_path, 'storage')
        local_storage = LocalFileStorage(storage_directory)

        local_storage.store_simulation_data(dummy_data_1, dummy_data_path_1)
        local_storage.store_simulation_data(dummy_data_2, dummy_data_path_2)

        stored_data = local_storage.retrieve_simulation_data(dummy_substance)

        assert len(stored_data[dummy_substance.identifier]) == 1

        # Make sure the correct data got retained.
        stored_data_object, stored_data_directory = stored_data[dummy_substance.identifier][0]
        assert stored_data_object.coordinate_file_name == data_2_coordinate_name

        dummy_data_path_3 = os.path.join(base_directory_path, 'data_3')
        data_3_coordinate_name = 'data_3.pdb'

        dummy_data_3 = create_dummy_stored_simulation_data(directory_path=dummy_data_path_3,
                                                           substance=dummy_substance,
                                                           force_field_id='ff_id_2',
                                                           coordinate_file_name=data_3_coordinate_name,
                                                           statistical_inefficiency=3.0)

        local_storage.store_simulation_data(dummy_data_3, dummy_data_path_3)

        stored_data = local_storage.retrieve_simulation_data(dummy_substance)
        assert len(stored_data[dummy_substance.identifier]) == 2

        # Make sure the correct data got retained.
        stored_data_object, stored_data_directory = stored_data[dummy_substance.identifier][1]
        assert stored_data_object.coordinate_file_name == data_3_coordinate_name


def test_data_collection_merging():
    """A test that compatible simulation data collections get
    merged together within the`LocalStorage` system."""

    with tempfile.TemporaryDirectory() as base_directory_path:

        dummy_substance = create_dummy_substance(1)

        storage_directory = os.path.join(base_directory_path, 'storage')
        local_storage = LocalFileStorage(storage_directory)

        for collection_index in range(2):

            data_collection_directory = os.path.join(base_directory_path,
                                                     f'dummy_collection_{collection_index}')

            os.makedirs(data_collection_directory, exist_ok=True)

            dummy_data_collection = StoredDataCollection()
            dummy_data_collection.substance = dummy_substance
            dummy_data_collection.force_field_id = 'ff_id'

            for inner_data_index in range(2):

                data_key = f'data_{inner_data_index}'
                data_path = os.path.join(data_collection_directory, data_key)

                data_coordinate_name = f'data_{collection_index}_{inner_data_index}.pdb'

                inner_data = create_dummy_stored_simulation_data(directory_path=data_path,
                                                                 substance=dummy_substance,
                                                                 force_field_id=f'ff_id',
                                                                 coordinate_file_name=data_coordinate_name,
                                                                 statistical_inefficiency=float(collection_index))

                dummy_data_collection.data[data_key] = inner_data

            local_storage.store_simulation_data(dummy_data_collection, data_collection_directory)

        stored_data = local_storage.retrieve_simulation_data(substance=dummy_substance,
                                                             include_component_data=True,
                                                             data_class=StoredDataCollection)

        assert len(stored_data[dummy_substance.identifier]) == 1

        data_object, data_directory = stored_data[dummy_substance.identifier][0]

        assert data_object.data['data_0'].coordinate_file_name == 'data_1_0.pdb'
        assert data_object.data['data_1'].coordinate_file_name == 'data_1_1.pdb'
