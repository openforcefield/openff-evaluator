"""
A local file based storage backend.
"""
import json
import shutil
from os import makedirs, path

from evaluator.storage import StorageBackend
from evaluator.storage.data import BaseStoredData
from evaluator.utils.serialization import TypedJSONEncoder


class LocalFileStorage(StorageBackend):
    """A storage backend which stores files in directories on the local
    disk.
    """

    @property
    def root_directory(self):
        """str: Returns the directory in which all stored objects are located."""
        return self.root_directory

    def __init__(self, root_directory="stored_data"):

        self._root_directory = root_directory

        if not path.isdir(root_directory) and len(root_directory) > 0:
            makedirs(root_directory)

        super().__init__()

    def _store_object(
        self, object_to_store, storage_key=None, ancillary_data_path=None
    ):

        file_path = path.join(self._root_directory, f"{storage_key}.json")
        directory_path = path.join(self._root_directory, f"{storage_key}")

        with open(file_path, "w") as file:
            json.dump(object_to_store, file, cls=TypedJSONEncoder)

        if object_to_store.has_ancillary_data():

            if path.isdir(directory_path):
                shutil.rmtree(directory_path, ignore_errors=True)

            shutil.move(ancillary_data_path, directory_path)

    def _retrieve_object(self, storage_key, expected_type=None):

        if not self._object_exists(storage_key):
            return None, None

        file_path = path.join(self._root_directory, f"{storage_key}.json")
        directory_path = None

        loaded_object = BaseStoredData.from_json(file_path)

        # Make sure the data has the correct type.
        if expected_type is not None and not isinstance(loaded_object, expected_type):

            raise ValueError(
                f"The retrieve object is of type {loaded_object.__class__.__name__} and not "
                f"{expected_type.__name__} as expected."
            )

        assert isinstance(loaded_object, BaseStoredData)

        # Check whether there is a data directory
        if loaded_object.has_ancillary_data():
            directory_path = path.join(self._root_directory, f"{storage_key}")

        return loaded_object, directory_path

    def _object_exists(self, storage_key):

        file_path = path.join(self._root_directory, f"{storage_key}.json")

        if not path.isfile(file_path):
            return False

        return True
