"""
A local file based storage backend.
"""

import logging
import pickle
from os import path, makedirs
from shutil import move

from propertyestimator.substances import Mixture
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

    def store_object(self, storage_key, object_to_store):

        file_path = path.join(self._root_directory, storage_key)

        try:

            with open(file_path, 'wb') as file:
                pickle.dump(object_to_store, file)

        except pickle.PicklingError:
            logging.warning('Unable to pickle an object to {}'.format(storage_key))

        super(LocalFileStorage, self).store_object(storage_key, object_to_store)

    def retrieve_object(self, storage_key):

        if not self.has_object(storage_key):
            return None

        file_path = path.join(self._root_directory, storage_key)

        loaded_object = None

        try:

            with open(file_path, 'rb') as file:
                loaded_object = pickle.load(file)

        except pickle.UnpicklingError:
            logging.warning('Unable to unpickle the object at {}'.format(storage_key))

        return loaded_object

    def has_object(self, storage_key):

        file_path = path.join(self._root_directory, storage_key)

        if not path.isfile(file_path):
            return False

        return True

    def store_simulation_data(self, substance_id, simulation_data_directory):

        unique_id = super(LocalFileStorage, self).store_simulation_data(substance_id,
                                                                        simulation_data_directory)

        move(simulation_data_directory, path.join(self._root_directory, f'{unique_id}_data'))
        return unique_id

    def retrieve_simulation_data(self, substance, include_pure_data=True):

        substance_ids = [substance.identifier]

        if isinstance(substance, Mixture) and include_pure_data is True:

            for component in substance.components:

                component_mixture = Mixture()
                component_mixture.add_component(component.smiles, 1.0, False)

                if component_mixture.identifier not in substance_ids:
                    substance_ids.append(component_mixture.identifier)

        return_paths = {}

        for substance_id in substance_ids:

            if substance_id not in self._simulation_data_by_substance:
                continue

            return_paths[substance_id] = []

            for simulation_data_key in self._simulation_data_by_substance[substance_id]:

                stored_object = self.retrieve_object(simulation_data_key)
                return_paths[substance_id].append(path.join(self._root_directory, f'{stored_object.unique_id}_data'))

        return return_paths
