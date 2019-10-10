"""
Defines the base API for the property estimator storage backend.
"""

import hashlib
import uuid
from os import path

from propertyestimator.forcefield import ForceFieldSource
from propertyestimator.storage import StoredSimulationData
from propertyestimator.storage.dataclasses import BaseStoredData
from propertyestimator.utils.string import sanitize_smiles_file_name


class PropertyEstimatorStorage:
    """An abstract base representation of how the property estimator will
    interact with and store simulation data.

    Notes
    -----
    Any inheriting class must provide an implementation for the
    `store_object`, `retrieve_object` and `has_object` methods
    """

    def __init__(self):
        """Constructs a new PropertyEstimatorStorage object.
        """

        self._stored_object_keys = set()
        self._stored_object_keys_file = 'internal_object_keys'

        # Store a map between the unique id of a force field,
        # and its hash value for easy comparision of force fields.
        self._force_field_id_map = {}
        self._force_field_id_map_file = 'internal_force_field_map'

        self._simulation_data_by_substance = {}
        self._simulation_data_by_substance_file = 'internal_simulation_data_map'

        self._load_stored_object_keys()
        self._load_force_field_hashes()
        self._load_simulation_data_map()

    def _load_stored_object_keys(self):
        """Load the unique key to each object stored in the storage system.
        """
        stored_object_keys = self._retrieve_object(self._stored_object_keys_file)

        if stored_object_keys is None:
            stored_object_keys = set()

        for unique_key in stored_object_keys:

            if not self._has_object(unique_key):
                # The stored entry key does not exist in the system, so skip the entry.
                continue

            self._stored_object_keys.add(unique_key)

        # Store a fresh copy of the key dictionary so that only entries
        # that exist in the system actually referenced.
        self._save_stored_object_keys()

    def _save_stored_object_keys(self):
        """Save the unique key of each of the objects stored in the storage system.
        """
        self._store_object(self._stored_object_keys_file, self._stored_object_keys)

    def _store_object(self, storage_key, object_to_store):
        """Store an object in the estimators storage system.

        Parameters
        ----------
        storage_key: str
            A unique key that describes where the object will be stored in
             the storage system.
        object_to_store: Any
            The object to store. The object must be pickle serializable.
        """

        if object_to_store is None:
            raise ValueError('The object to store cannot be None.')

        if storage_key in self._stored_object_keys:
            return

        self._stored_object_keys.add(storage_key)
        self._save_stored_object_keys()

    def _retrieve_object(self, storage_key):
        """Retrieves a stored object for the estimators storage system.

        Parameters
        ----------
        storage_key: str
            A unique key that describes where the stored object can be found
            within the storage system.

        Returns
        -------
        Any, optional
            The stored object if the object key is found, otherwise None.
        """
        raise NotImplementedError()

    def _has_object(self, storage_key):
        """Check whether an object with the specified key exists in the
        storage system.

        Parameters
        ----------
        storage_key: str
            A unique key that describes where the stored object can be found
            within the storage system.

        Returns
        -------
        True if the object is within the storage system.
        """
        raise NotImplementedError()

    def _load_force_field_hashes(self):
        """Load the unique id and hash keys of each of the force fields which
         have been stored in the force field directory (``self._force_field_root``).
        """
        force_field_id_map = self._retrieve_object(self._force_field_id_map_file)

        if force_field_id_map is None:
            force_field_id_map = {}

        for unique_id in force_field_id_map:

            force_field_key = 'force_field_{}'.format(unique_id)

            if not self._has_object(force_field_key):
                # The force field file does not exist, so skip the entry.
                continue

            self._force_field_id_map[unique_id] = force_field_id_map[unique_id]

        # Store a fresh copy of the hashes so that only force fields that
        # exist are actually referenced.
        self._save_force_field_hashes()

    def _save_force_field_hashes(self):
        """Save the unique id and force field hash key dictionary.
        """
        self._store_object(self._force_field_id_map_file, self._force_field_id_map)

    @staticmethod
    def _force_field_to_hash(force_field):
        """Converts a ForceFieldSource object to a hash
        string.

        Parameters
        ----------
        force_field: ForceFieldSource
            The force field to hash.

        Returns
        -------
        str
            The hash key of the force field.
        """
        force_field_string = force_field.json()
        return hashlib.sha256(force_field_string.encode()).hexdigest()

    def has_force_field(self, force_field):
        """Checks whether the force field has been previously
        stored in the force field directory.

        Parameters
        ----------
        force_field: ForceFieldSource
            The force field to check for.

        Returns
        -------
        str, optional
            None if the force field has not been cached, otherwise
            the unique id of the cached force field.
        """

        hash_string = self._force_field_to_hash(force_field)

        for unique_id in self._force_field_id_map:

            existing_hash = self._force_field_id_map[unique_id]

            if hash_string != existing_hash:
                continue

            force_field_key = 'force_field_{}'.format(unique_id)

            if not self._has_object(force_field_key):
                # For some reason the force field got deleted..
                continue

            return unique_id

        return None

    def retrieve_force_field(self, unique_id):
        """Retrieves a force field from storage, if it exists.

        Parameters
        ----------
        unique_id: str
            The unique id of the force field to retrieve

        Returns
        -------
        ForceFieldSource, optional
            The force field if present in the storage system with the given key, otherwise None.
        """
        force_field_key = 'force_field_{}'.format(unique_id)
        force_field_source = self._retrieve_object(force_field_key)

        if force_field_source is None:

            raise KeyError(f'The force field with id {unique_id} does not exist '
                           f'in the storage system.')

        if not isinstance(force_field_source, ForceFieldSource):
            raise ValueError(f'The stored force field is invalid.')

        return force_field_source

    def store_force_field(self, force_field):
        """Store the force field in the cached force field
        directory.

        Parameters
        ----------
        force_field: ForceFieldSource
            The force field to store.

        Returns
        -------
        str
            The unique id of the stored force field.
        """

        unique_id = str(uuid.uuid4())

        # Be extra cautious and mash sure there wasn't
        # a hash collision.
        while unique_id in self._force_field_id_map:
            unique_id = str(uuid.uuid4())

        hash_string = self._force_field_to_hash(force_field)
        force_field_key = 'force_field_{}'.format(unique_id)

        # We make sure to strip the cosmetic attributes from the stored FF as these should
        # not affect the science of the FF, and aren't currently consumed by the estimator.
        self._store_object(force_field_key, force_field)

        # Make sure to hash the force field for easy access.
        if (unique_id not in self._force_field_id_map or
            hash_string != self._force_field_id_map[unique_id]):

            self._force_field_id_map[unique_id] = hash_string
            self._save_force_field_hashes()

        return unique_id

    def _load_simulation_data_map(self):
        """Load the dictionary which tracks which stored simulation data
        was calculated for a specific substance.
        """
        _simulation_data_by_substance = self._retrieve_object(self._simulation_data_by_substance_file)

        if _simulation_data_by_substance is None:
            _simulation_data_by_substance = {}

        for substance_id in _simulation_data_by_substance:

            self._simulation_data_by_substance[substance_id] = []

            for unique_id in _simulation_data_by_substance[substance_id]:

                data_object, data_directory = self.retrieve_simulation_data_by_id(unique_id)

                if data_object is None or not path.isdir(data_directory):
                    # The stored data does not exist, so skip the entry.
                    continue

                self._simulation_data_by_substance[substance_id].append(unique_id)

        # Store a fresh copy of the hashes so that only data that
        # exists is actually referenced.
        self._save_simulation_data_map()

    def _save_simulation_data_map(self):
        """Save the unique id and simulation data key by substance dictionary.
        """
        self._store_object(self._simulation_data_by_substance_file,
                           self._simulation_data_by_substance)

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
        raise NotImplementedError()

    def retrieve_simulation_data(self, substance, include_component_data=True,
                                 data_class=StoredSimulationData):

        """Retrieves any data that has been stored for a given substance.

        Parameters
        ----------
        substance: Substance
            The substance to check for.
        include_component_data: bool
            If the substance if a mixture where has multiple components and `include_component_data`
            is True, data will be returned for both the mixed system, and for the individual
            components, otherwise only data for the mixed system will be returned.
        data_class: subclass of BaseStoredData
            The type of data to retrieve.

        Returns
        -------
        dict of str and tuple of BaseStoredData and str
            A dictionary of the stored data objects and their corresponding directory paths
            partitioned by substance id.
        """
        raise NotImplementedError()

    def store_simulation_data(self, data_object, data_directory):
        """Store the simulation data.

        Notes
        -----
        If the storage system already contains equivalent information (i.e data stored
        for the same substance, thermodynamic state and parameter set) then the
        data will be merged according to the data objects `merge` method.

        Parameters
        ----------
        data_object: BaseStoredData
            The data object being stored.
        data_directory: str
            The directory which stores files associated with
            the data object such as trajectory files.

        Returns
        -------
        str
            The unique id of the stored data.
        """

        if not path.isdir(data_directory):

            raise ValueError(f'The {data_directory} data directory either could'
                             f' not be found or is invalid.')

        if not isinstance(data_object, BaseStoredData):
            raise ValueError('The data object must inherit from the `BaseStoredData` class.')

        if data_object.substance is None:
            raise ValueError('The data object must have a valid substance.')

        substance_id = data_object.substance.identifier

        existing_data_key = None
        data_to_store = None

        if substance_id in self._simulation_data_by_substance:

            # Check if any existing stored data is compatible with the
            # new data we are trying to store
            for stored_data_key in self._simulation_data_by_substance[substance_id]:

                stored_data = self._retrieve_object(stored_data_key)

                if not stored_data.can_merge(data_object):
                    continue

                data_to_store = stored_data.merge(stored_data, data_object)
                existing_data_key = stored_data_key

                break

        if existing_data_key is None:

            sanitized_id = sanitize_smiles_file_name(substance_id)
            existing_data_key = "{}_{}".format(sanitized_id, uuid.uuid4())
            data_to_store = data_object

        self._store_object(existing_data_key, data_to_store)

        # Store the unique id assigned to the data in the master
        # list of ids if not already present.
        if (substance_id not in self._simulation_data_by_substance or
            existing_data_key not in self._simulation_data_by_substance[substance_id]):

            if substance_id not in self._simulation_data_by_substance:
                self._simulation_data_by_substance[substance_id] = []

            self._simulation_data_by_substance[substance_id].append(existing_data_key)
            self._save_simulation_data_map()

        return existing_data_key
