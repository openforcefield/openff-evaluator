"""
Defines the base API for the property estimator storage backend.
"""
import abc
from collections import defaultdict
from typing import Dict

from propertyestimator.storage.attributes import StorageAttribute
from propertyestimator.storage.data import (
    BaseStoredData,
    HashableStoredData,
    MergeableStoredData,
)


class StorageBackend(abc.ABC):
    """An abstract base representation of how the property estimator will
    interact with and store simulation data.
    """

    class _ObjectKeyData(BaseStoredData):
        """An object which keeps track of the items in
        the storage system.
        """

        object_keys = StorageAttribute(
            docstring="The unique keys of the objects stored in a `StorageBackend`.",
            type_hint=dict,
        )

        @classmethod
        def has_ancillary_data(cls):
            return False

    def __init__(self):
        """Constructs a new StorageBackend object.
        """
        self._stored_object_keys = None
        self._stored_object_keys_id = "object_keys"

        # Store a map between the unique id of a stored object,
        # and its hash value for easy comparision.
        self._object_hashes: Dict[int, str] = dict()

        self._load_stored_object_keys()

    def _load_stored_object_keys(self):
        """Load the unique key to each object stored in the
        storage system.
        """
        keys_object = self.retrieve_object(self._stored_object_keys_id)
        assert isinstance(keys_object, StorageBackend._ObjectKeyData)

        stored_object_keys = {}
        all_object_keys = set()

        if keys_object is not None:
            stored_object_keys = keys_object.object_keys

        for data_type in stored_object_keys:

            for unique_key in stored_object_keys[data_type]:

                if not self._has_object(unique_key):
                    # The stored entry key does not exist in the system,
                    # so skip the entry. This may happen when the local
                    # file do not exist on disk any more for example.
                    continue

                if unique_key in all_object_keys:

                    raise IndexError(
                        "Two objects with the same unique key have been found."
                    )

                stored_object = self.retrieve_object(unique_key)

                # Make sure the data matches the expected type and is valid.
                assert stored_object.__class__.__name__ == data_type
                stored_object.validate()

                if isinstance(stored_object, HashableStoredData):
                    self._object_hashes[hash(stored_object)] = unique_key

                self._stored_object_keys[data_type].add(unique_key)
                all_object_keys.add(unique_key)

        # Store a fresh copy of the key dictionary so that only entries
        # that exist in the system actually referenced.
        self._save_stored_object_keys()

    def _save_stored_object_keys(self):
        """Save the unique key of each of the objects stored in the storage system.
        """
        keys_object = StorageBackend._ObjectKeyData()
        keys_object.object_keys = self._stored_object_keys

        self.store_object(self._stored_object_keys_id, keys_object)

    @abc.abstractmethod
    def store_object(self, storage_key, object_to_store, ancillary_data_path=None):
        """Store an object in the storage system, returning the key
        of the stored object. This may be different to `storage_key`
        depending on whether the same or a similar object was already
        present in the system.

        Parameters
        ----------
        storage_key: str
            A unique key to associate with the stored object.
        object_to_store: BaseStoredData
            The object to store.
        ancillary_data_path: str, optional
            The data path to the ancillary directory-like
            data to store alongside the object if the data
            type requires one.

        Returns
        -------
        str
            The unique key assigned to the stored object.
        """

        if object_to_store is None:
            raise ValueError("The object to store cannot be None.")

        # Make sure the object is a supported type.
        if not isinstance(object_to_store, BaseStoredData):

            raise ValueError(
                "Only objects inheriting from `BaseStoredData` can "
                "be stored in the storage system."
            )

        # Make sure we have ancillary data if required.
        if (
            object_to_store.__class__.has_ancillary_data()
            and ancillary_data_path is None
        ):
            raise ValueError("This object requires ancillary data.")

        # Make sure the key in unique.
        if any(
            (
                storage_key in self._stored_object_keys[data_type]
                for data_type in self._stored_object_keys
            )
        ):

            raise IndexError(
                f"An object with the key {storage_key} already "
                f"exists in the system."
            )

        # Check whether the exact same object already exists within
        # the storage system based on its hash.
        if (
            isinstance(object_to_store, HashableStoredData)
            and hash(object_to_store) in self._object_hashes
        ):
            return self._object_hashes[hash(object_to_store)]

        # Check whether a similar piece of data already exists in the
        # storage system, and decide which to keep.
        elif isinstance(object_to_store, MergeableStoredData):
            raise NotImplementedError()

        self._stored_object_keys.add(storage_key)
        self._save_stored_object_keys()

    @abc.abstractmethod
    def retrieve_object(self, storage_key, expected_type=None):
        """Retrieves a stored object for the estimators storage system.

        Parameters
        ----------
        storage_key: str
            A unique key that describes where the stored object can be found
            within the storage system.
        expected_type: type of BaseStoredData, optional
            The expected data type. An exception is raised if
            the retrieved data doesn't match the type.

        Returns
        -------
        BaseStoredData, optional
            The stored object if the object key is found, otherwise None.
        str, optional
            The path to the ancillary data if present.
        """
        raise NotImplementedError()

    @abc.abstractmethod
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

    def has_object(self, hashable_object):
        """Checks whether a given hashable object exists in the
        storage system.

        Parameters
        ----------
        hashable_object: HashableStoredData
            The object to check for.

        Returns
        -------
        str, optional
            The unique key of the object is in the system, `None` otherwise.
        """
        hash_key = hash(hashable_object)
        return self._object_hashes.get(hash_key, None)

    def query(self, data_query):
        """Query the storage backend for data matching the
        query criteria.

        Parameters
        ----------
        data_query: BaseDataQuery
            The query to perform.

        Returns
        -------
        dict of tuple and list of BaseStoredData
            The data that matches the query, partitioned
            by the query returned category.
        """

        data_class = data_query.supported_data_class()
        results = defaultdict(list)

        if len(self._stored_object_keys.get(data_class.__name__, [])) == 0:
            # Exit early of there are no objects of the correct type.
            return results

        for unique_key in self._stored_object_keys[data_class.__name__]:

            if not self._has_object(unique_key):
                # Make sure the object is still in the system.
                continue

            stored_object = self.retrieve_object(unique_key, data_class)

            matches = data_query.apply(stored_object)

            if matches is None:
                continue

            results[matches].append(stored_object)

        return results
