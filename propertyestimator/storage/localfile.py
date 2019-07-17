"""
A local file based storage backend.
"""
import json
from os import path, makedirs
from shutil import move

from propertyestimator.storage import StoredSimulationData
from propertyestimator.storage.dataclasses import BaseStoredData
from propertyestimator.substances import Substance
from propertyestimator.utils.serialization import TypedJSONEncoder, TypedJSONDecoder
from .storage import PropertyEstimatorStorage


class LocalFileStorage(PropertyEstimatorStorage):
    """A storage backend which stores files normally on the local
    disk.
    """

    @property
    def root_directory(self):
        """str: Returns the directory in which all stored objects are located."""
        return self.root_directory

    def __init__(self, root_directory='stored_data'):

        self._root_directory = root_directory

        if not path.isdir(root_directory):
            makedirs(root_directory)

        super().__init__()

    def _store_object(self, storage_key, object_to_store):

        file_path = path.join(self._root_directory, storage_key)

        # If the object to store is a simple string we write that
        # directly, otherwise we try and JSONify the object.
        if not isinstance(object_to_store, str):
            object_to_store = json.dumps(object_to_store, cls=TypedJSONEncoder)

        with open(file_path, 'w') as file:
            file.write(object_to_store)

        super(LocalFileStorage, self)._store_object(storage_key, object_to_store)

    def _retrieve_object(self, storage_key):

        if not self._has_object(storage_key):
            return None

        file_path = path.join(self._root_directory, storage_key)

        with open(file_path, 'r') as file:
            loaded_object_string = file.read()

        try:
            loaded_object = json.loads(loaded_object_string, cls=TypedJSONDecoder)
        except json.JSONDecodeError:
            loaded_object = loaded_object_string

        return loaded_object

    def _has_object(self, storage_key):

        file_path = path.join(self._root_directory, storage_key)

        if not path.isfile(file_path):
            return False

        return True

    def store_simulation_data(self, data_object, data_directory):

        unique_id = super(LocalFileStorage, self).store_simulation_data(data_object,
                                                                        data_directory)

        move(data_directory, path.join(self._root_directory, f'{unique_id}_data'))
        return unique_id

    def retrieve_simulation_data_by_id(self, unique_id):
        """Attempts to retrieve a storage piece of simulation data
        from it's unique id.

        Parameters
        ----------
        unique_id: str
            The unique id assigned to the data.

        Returns
        -------
        BaseStoredData
            The stored data object.
        str
            The path to the data's corresponding directory.
        """
        stored_object = self._retrieve_object(unique_id)

        # Make sure the stored object is a valid object.
        if not isinstance(stored_object, BaseStoredData):
            return None, None

        data_directory = path.join(self._root_directory, f'{unique_id}_data')
        return stored_object, data_directory

    def retrieve_simulation_data(self, substance, include_component_data=True,
                                 data_class=StoredSimulationData):

        substance_ids = {substance.identifier}

        # Find the substance identifiers of the substance components if
        # we should include component data.
        if isinstance(substance, Substance) and include_component_data is True:

            for component in substance.components:

                component_substance = Substance()
                component_substance.add_component(component, Substance.MoleFraction())

                substance_ids.add(component_substance.identifier)

        return_data = {}

        for substance_id in substance_ids:

            if substance_id not in self._simulation_data_by_substance:
                continue

            return_data[substance_id] = []

            for data_key in self._simulation_data_by_substance[substance_id]:

                data_object, data_directory = self.retrieve_simulation_data_by_id(data_key)

                if data_object is None:
                    continue

                if not isinstance(data_object, data_class):
                    continue

                return_data[substance_id].append((data_object, data_directory))

        return return_data
