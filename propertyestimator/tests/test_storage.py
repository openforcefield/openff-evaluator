"""
Units tests for propertyestimator.storage
"""
import json
import os
import tempfile
from shutil import rmtree

from propertyestimator.tests.utils import create_dummy_stored_simulation_data, create_dummy_substance
from simtk import unit

from propertyestimator.storage import LocalFileStorage, StoredSimulationData
from propertyestimator.storage.dataclasses import BaseStoredData, StoredDataCollection
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import get_data_filename
from propertyestimator.utils.serialization import serialize_force_field, TypedJSONEncoder, TypedJSONDecoder


class DummyDataClass1(BaseStoredData):
    pass


class DummyDataClass2(BaseStoredData):
    pass


def test_local_force_field_storage():
    """A simple test to that force fields can be stored and
    retrieved using the local storage backend."""

    from openforcefield.typing.engines import smirnoff
    force_field = smirnoff.ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))

    with tempfile.TemporaryDirectory() as temporary_directory:

        local_storage = LocalFileStorage(temporary_directory)
        local_storage.store_force_field('tmp_id', force_field)

        retrieved_force_field = local_storage.retrieve_force_field('tmp_id')

        serialized_force_field = serialize_force_field(force_field)
        serialized_retrieved_force_field = serialize_force_field(retrieved_force_field)

        assert json.dumps(serialized_force_field) == json.dumps(serialized_retrieved_force_field)

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

    temporary_data_directory = 'temp_data'
    temporary_backend_directory = 'storage_dir'

    if os.path.isdir(temporary_data_directory):
        rmtree(temporary_data_directory)

    if os.path.isdir(temporary_backend_directory):
        rmtree(temporary_backend_directory)

    os.makedirs(temporary_data_directory)
    os.makedirs(temporary_backend_directory)

    with open(os.path.join(temporary_data_directory, 'data.json'), 'w') as file:
        json.dump(dummy_simulation_data, file, cls=TypedJSONEncoder)

    local_storage = LocalFileStorage(temporary_backend_directory)
    dummy_simulation_data.unique_id = local_storage.store_simulation_data(substance.identifier,
                                                                          temporary_data_directory)

    retrieved_data_directories = local_storage.retrieve_simulation_data(substance)
    assert len(retrieved_data_directories) == 1

    retrieved_data_directory = retrieved_data_directories[substance.identifier][0]

    with open(os.path.join(retrieved_data_directory, 'data.json'), 'r') as file:
        retrieved_data = json.load(file, cls=TypedJSONDecoder)

    assert dummy_simulation_data.thermodynamic_state == retrieved_data.thermodynamic_state
    assert dummy_simulation_data.statistical_inefficiency == retrieved_data.statistical_inefficiency
    assert dummy_simulation_data.force_field_id == retrieved_data.force_field_id
    assert dummy_simulation_data.substance == retrieved_data.substance

    local_storage_new = LocalFileStorage(temporary_backend_directory)
    assert local_storage_new.has_object(dummy_simulation_data.unique_id)

    if os.path.isdir(temporary_data_directory):
        rmtree(temporary_data_directory)

    if os.path.isdir(temporary_backend_directory):
        rmtree(temporary_backend_directory)


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

            data_class = data_class_type()

            with open(os.path.join(data_directory, 'data.json'), 'w') as file:
                json.dump(data_class, file, cls=TypedJSONEncoder)

            local_storage.store_simulation_data(substance.identifier, data_directory)

        retrieved_data_directories = local_storage.retrieve_simulation_data(substance, data_class=BaseStoredData)
        assert len(retrieved_data_directories[substance.identifier]) == 2

        retrieved_data_directories = local_storage.retrieve_simulation_data(substance, data_class=DummyDataClass1)
        assert len(retrieved_data_directories[substance.identifier]) == 1

        retrieved_data_directories = local_storage.retrieve_simulation_data(substance, data_class=DummyDataClass2)
        assert len(retrieved_data_directories[substance.identifier]) == 1

        retrieved_data_directories = local_storage.retrieve_simulation_data(substance, data_class=StoredSimulationData)
        assert len(retrieved_data_directories[substance.identifier]) == 0


def test_simulation_data_merging():
    
    with tempfile.TemporaryDirectory() as base_directory_path:

        dummy_substance = create_dummy_substance(1)

        dummy_data_path_1 = os.path.join(base_directory_path, 'data_1')
        data_1_coordinate_name = 'data_1.pdb'

        create_dummy_stored_simulation_data(directory_path=dummy_data_path_1,
                                            substance=dummy_substance,
                                            force_field_id='ff_id_1',
                                            coordinate_file_name=data_1_coordinate_name,
                                            statistical_inefficiency=1.0)

        dummy_data_path_2 = os.path.join(base_directory_path, 'data_2')
        data_2_coordinate_name = 'data_2.pdb'

        create_dummy_stored_simulation_data(directory_path=dummy_data_path_2,
                                            substance=dummy_substance,
                                            force_field_id='ff_id_1',
                                            coordinate_file_name=data_2_coordinate_name,
                                            statistical_inefficiency=2.0)

        storage_directory = os.path.join(base_directory_path, 'storage')
        local_storage = LocalFileStorage(storage_directory)

        local_storage.store_simulation_data(dummy_substance.identifier, dummy_data_path_1)
        local_storage.store_simulation_data(dummy_substance.identifier, dummy_data_path_2)

        stored_data = local_storage.retrieve_simulation_data(dummy_substance)

        assert len(stored_data[dummy_substance.identifier]) == 1

        # Make sure the correct data got retained.
        with open(os.path.join(stored_data[dummy_substance.identifier][0], 'data.json')) as file:

            stored_data_object = json.load(file, cls=TypedJSONDecoder)
            assert stored_data_object.coordinate_file_name == data_2_coordinate_name

        dummy_data_path_3 = os.path.join(base_directory_path, 'data_3')
        data_3_coordinate_name = 'data_3.pdb'

        create_dummy_stored_simulation_data(directory_path=dummy_data_path_3,
                                            substance=dummy_substance,
                                            force_field_id='ff_id_2',
                                            coordinate_file_name=data_3_coordinate_name,
                                            statistical_inefficiency=3.0)

        local_storage.store_simulation_data(dummy_substance.identifier, dummy_data_path_3)

        stored_data = local_storage.retrieve_simulation_data(dummy_substance)
        assert len(stored_data[dummy_substance.identifier]) == 2

        # Make sure the correct data got retained.
        with open(os.path.join(stored_data[dummy_substance.identifier][1], 'data.json')) as file:

            stored_data_object = json.load(file, cls=TypedJSONDecoder)
            assert stored_data_object.coordinate_file_name == data_3_coordinate_name


def test_data_collection_merging():

    with tempfile.TemporaryDirectory() as base_directory_path:

        dummy_substance = create_dummy_substance(1)
        dummy_data_paths = []

        for index in range(4):

            dummy_data_path = os.path.join(base_directory_path, f'data_{index}')
            dummy_data_paths.append(dummy_data_path)

            data_coordinate_name = f'data_{index}.pdb'

            create_dummy_stored_simulation_data(directory_path=dummy_data_path,
                                                substance=dummy_substance,
                                                force_field_id=f'ff_id_{index}',
                                                coordinate_file_name=data_coordinate_name,
                                                statistical_inefficiency=float(index))

        dummy_data_collection_1 = StoredDataCollection()

        dummy_data_collection_1.data = {
            'data_1': dummy_data_paths[0],
            'data_2': dummy_data_paths[1],
        }

        dummy_data_collection_2 = StoredDataCollection()
        dummy_data_collection_2.data = {
            'data_1': dummy_data_paths[2],
            'data_2': dummy_data_paths[3],
        }

        storage_directory = os.path.join(base_directory_path, 'storage')
        local_storage = LocalFileStorage(storage_directory)

        local_storage.store_simulation_data(dummy_substance.identifier, dummy_data_path_1)
        local_storage.store_simulation_data(dummy_substance.identifier, dummy_data_path_2)

        stored_data = local_storage.retrieve_simulation_data(dummy_substance)

        assert len(stored_data[dummy_substance.identifier]) == 1

        # Make sure the correct data got retained.
        with open(os.path.join(stored_data[dummy_substance.identifier][0], 'data.json')) as file:
            stored_data_object = json.load(file, cls=TypedJSONDecoder)
            assert stored_data_object.coordinate_file_name == data_2_coordinate_name

        dummy_data_path_3 = os.path.join(base_directory_path, 'data_3')
        data_3_coordinate_name = 'data_3.pdb'

        create_dummy_stored_simulation_data(directory_path=dummy_data_path_3,
                                            substance=dummy_substance,
                                            force_field_id='ff_id_2',
                                            coordinate_file_name=data_3_coordinate_name,
                                            statistical_inefficiency=3.0)

        local_storage.store_simulation_data(dummy_substance.identifier, dummy_data_path_3)

        stored_data = local_storage.retrieve_simulation_data(dummy_substance)
        assert len(stored_data[dummy_substance.identifier]) == 2

        # Make sure the correct data got retained.
        with open(os.path.join(stored_data[dummy_substance.identifier][1], 'data.json')) as file:
            stored_data_object = json.load(file, cls=TypedJSONDecoder)
            assert stored_data_object.coordinate_file_name == data_3_coordinate_name
