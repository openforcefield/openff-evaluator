"""
Defines the base API for the property estimator storage backend.
"""
import abc
import uuid
from collections import defaultdict
from threading import RLock
from typing import Dict

from propertyestimator.attributes import Attribute
from propertyestimator.storage.data import (
    BaseStoredData,
    ForceFieldData,
    HashableStoredData,
)


class StorageBackend(abc.ABC):
    """An abstract base representation of how the property estimator will
    interact with and store simulation data.

    Notes
    -----
    When implementing this class, only private methods should be overridden
    as the public methods only mainly implement thread locks, while their
    private version perform their actual function.
    """

    class _ObjectKeyData(BaseStoredData):
        """An object which keeps track of the items in
        the storage system.
        """

        object_keys = Attribute(
            docstring="The unique keys of the objects stored in a `StorageBackend`.",
            type_hint=dict,
            default_value=dict(),
        )

        @classmethod
        def has_ancillary_data(cls):
            return False

        def to_storage_query(self):
            # This should never be called so doesn't need an
            # implementation.
            raise NotImplementedError()

    def __init__(self):
        """Constructs a new StorageBackend object.
        """
        self._stored_object_keys = None
        self._stored_object_keys_id = "object_keys"

        # Store a map between the unique id of a stored object,
        # and its hash value for easy comparision.
        self._object_hashes: Dict[int, str] = dict()

        # Create a thread lock to prevent concurrent
        # thread access.
        self._lock = RLock()

        self._load_stored_object_keys()

    def _load_stored_object_keys(self):
        """Load the unique key to each object stored in the
        storage system.
        """
        keys_object, _ = self._retrieve_object(self._stored_object_keys_id)

        if keys_object is None:
            keys_object = StorageBackend._ObjectKeyData()

        assert isinstance(keys_object, StorageBackend._ObjectKeyData)

        stored_object_keys = keys_object.object_keys
        self._stored_object_keys = defaultdict(list)

        all_object_keys = set()

        for data_type in stored_object_keys:

            for unique_key in stored_object_keys[data_type]:

                if not self._object_exists(unique_key):
                    # The stored entry key does not exist in the system,
                    # so skip the entry. This may happen when the local
                    # file do not exist on disk any more for example.
                    continue

                if unique_key in all_object_keys:

                    raise KeyError(
                        "Two objects with the same unique key have been found."
                    )

                stored_object, _ = self.retrieve_object(unique_key)

                # Make sure the data matches the expected type and is valid.
                assert stored_object.__class__.__name__ == data_type
                stored_object.validate()

                if isinstance(stored_object, HashableStoredData):
                    self._object_hashes[hash(stored_object)] = unique_key

                self._stored_object_keys[data_type].append(unique_key)
                all_object_keys.add(unique_key)

        # Store a fresh copy of the key dictionary so that only entries
        # that exist in the system actually referenced.
        self._save_stored_object_keys()

    def _save_stored_object_keys(self):
        """Save the unique key of each of the objects stored in the storage system.
        """
        keys_object = StorageBackend._ObjectKeyData()
        keys_object.object_keys = self._stored_object_keys

        self._store_object(keys_object, self._stored_object_keys_id)

    @abc.abstractmethod
    def _object_exists(self, storage_key):
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

    def _is_key_unique(self, storage_key):
        """Checks whether a given key is already in the storage system.

        Parameters
        ----------
        storage_key: str
            The key to check for.

        Returns
        -------
        bool
            `True` if the key exists in the system, `False`
            otherwise.
        """
        # Make sure the key in unique.
        return not any(
            storage_key in self._stored_object_keys[data_type]
            for data_type in self._stored_object_keys
        )

    @abc.abstractmethod
    def _store_object(
        self, object_to_store, storage_key=None, ancillary_data_path=None
    ):
        """The internal implementation of the `store_object` method.
        It is safe to assume here that all object and key validation
        has already been performed, and that this method is called under
        a thread lock.

        Parameters
        ----------
        object_to_store: BaseStoredData
            The object to store.
        storage_key: str, optional
            A unique key to associate with the stored object. If `None`,
            one will be randomly generated
        ancillary_data_path: str, optional
            The data path to the ancillary directory-like
            data to store alongside the object if the data
            type requires one.
        """
        raise NotImplementedError()

    def store_object(self, object_to_store, ancillary_data_path=None):
        """Store an object in the storage system, returning the key
        of the stored object. This may be different to `storage_key`
        depending on whether the same or a similar object was already
        present in the system.

        Parameters
        ----------
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

        # Make sure the object is valid.
        if object_to_store is None:
            raise ValueError("The object to store cannot be None.")

        object_to_store.validate()

        # Make sure the object is a supported type.
        if not isinstance(object_to_store, BaseStoredData):

            raise ValueError(
                "Only objects inheriting from `BaseStoredData` can "
                "be stored in the storage system."
            )

        # Make sure we have ancillary data if required.
        object_class = object_to_store.__class__

        if object_class.has_ancillary_data() and ancillary_data_path is None:
            raise ValueError("This object requires ancillary data.")

        # Check whether the exact same object already exists within
        # the storage system based on its hash.
        storage_key = self.has_object(object_to_store)

        if storage_key is not None:
            return storage_key

        # Generate a unique id for this object.
        while storage_key is None or not self._is_key_unique(storage_key):
            storage_key = str(uuid.uuid4()).replace("-", "")

        # Hash this object if appropriate
        if isinstance(object_to_store, HashableStoredData):
            self._object_hashes[hash(object_to_store)] = storage_key

        # Save the object into the storage system with the given key.
        with self._lock:
            self._store_object(object_to_store, storage_key, ancillary_data_path)

        # Register the key in the storage system.
        if not isinstance(object_to_store, StorageBackend._ObjectKeyData):

            self._stored_object_keys[object_class.__name__].append(storage_key)
            self._save_stored_object_keys()

        return storage_key

    def store_force_field(self, force_field):
        """A convenience method for storing `ForceFieldSource` objects.

        Parameters
        ----------
        force_field: ForceFieldSource
            The force field to store.

        Returns
        -------
        str
            The unique id of the stored force field.
        """
        force_field_data = ForceFieldData()
        force_field_data.force_field_source = force_field

        return self.store_object(force_field_data)

    @abc.abstractmethod
    def _retrieve_object(self, storage_key, expected_type=None):
        """The internal implementation of the `retrieve_object` method.
        It is safe to assume that this method is called under a thread lock.

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
        with self._lock:
            return self._retrieve_object(storage_key, expected_type)

    def retrieve_force_field(self, storage_key):
        """A convenience method for retrieving `ForceFieldSource` objects.

        Parameters
        ----------
        storage_key: str
            The key of the force field to retrieve.

        Returns
        -------
        ForceFieldSource
            The retrieved force field source.
        """
        force_field_data, _ = self.retrieve_object(storage_key, ForceFieldData)

        if force_field_data is None:

            raise KeyError(
                f"The force field with id {storage_key} does not exist "
                f"in the storage system."
            )

        return force_field_data.force_field_source

    def _has_object(self, storage_object):
        """The internal implementation of the `has_object` method.
        It is safe to assume that this method is called under a
        thread lock.

        Parameters
        ----------
        storage_object: BaseStoredData
            The object to check for.

        Returns
        -------
        str, optional
            The unique key of the object if it is in the system, `None` otherwise.
        """

        if isinstance(storage_object, HashableStoredData):

            hash_key = hash(storage_object)
            return self._object_hashes.get(hash_key, None)

        data_query = storage_object.to_storage_query()
        query_results = self.query(data_query)

        if len(query_results) == 0:
            return None

        if len(query_results) > 1 or len(query_results[0]) > 1:

            raise ValueError(
                "The backend contains multiple copies of the "
                "same piece of data. This should not be possible."
            )

        storage_key, _ = next(iter(query_results.values()))[0]
        return storage_key

    def has_object(self, storage_object):
        """Checks whether a given hashable object exists in the
        storage system.

        Parameters
        ----------
        storage_object: BaseStoredData
            The object to check for.

        Returns
        -------
        str, optional
            The unique key of the object if it is in the system, `None` otherwise.
        """
        with self._lock:
            return self._has_object(storage_object)

    def has_force_field(self, force_field):
        """A convenience method for checking whether the specified
        `ForceFieldSource` object is stored in the backend.

        Parameters
        ----------
        force_field: ForceFieldSource
            The force field to look for.

        Returns
        -------
        str, optional
            The unique key of the object if it is in the system, `None` otherwise.
        """
        force_field_data = ForceFieldData()
        force_field_data.force_field_source = force_field

        return self.has_object(force_field_data)

    def _query(self, data_query):
        """The internal implementation of the `query` method.
        It is safe to assume that this method is called under a
        thread lock.

        Parameters
        ----------
        data_query: BaseDataQuery
            The query to perform.

        Returns
        -------
        dict of tuple and list of tuple of str and str
            The data that matches the query partitioned
            by the matched values..
        """

        data_class = data_query.data_class()
        results = defaultdict(list)

        if len(self._stored_object_keys.get(data_class.__name__, [])) == 0:
            # Exit early of there are no objects of the correct type.
            return results

        for unique_key in self._stored_object_keys[data_class.__name__]:

            if not self._object_exists(unique_key):
                # Make sure the object is still in the system.
                continue

            stored_object, stored_directory = self.retrieve_object(
                unique_key, data_class
            )

            matches = data_query.apply(stored_object)

            if matches is None:
                continue

            results[matches].append((unique_key, stored_directory))

        return results

    def query(self, data_query):
        """Query the storage backend for data matching the
        query criteria.

        Parameters
        ----------
        data_query: BaseDataQuery
            The query to perform.

        Returns
        -------
        dict of tuple and list of tuple of str and str
            The data that matches the query partitioned
            by the matched values..
        """
        with self._lock:
            return self._query(data_query)
