"""
Units tests for propertyestimator.storage
"""
import json
import os
import tempfile
from shutil import rmtree

from simtk import unit

from propertyestimator.storage import LocalFileStorage, StoredSimulationData
from propertyestimator.storage.dataclasses import BaseStoredData
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
