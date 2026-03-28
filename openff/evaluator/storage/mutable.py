"""
A mutable, extensible local file storage backend.
"""

import json
import shutil
from os import path

from openff.evaluator.storage.localfile import LocalFileStorage
from openff.evaluator.utils.serialization import TypedJSONEncoder


class MutableLocalFileStorage(LocalFileStorage):
    """A :class:`LocalFileStorage` subclass that adds mutating and
    compositional operations not present in the base class:

    - **Copy-on-store**: ancillary data directories are *copied* rather
      than moved, so the caller's original data is preserved.
    - **Combine**: merge another storage into this one via
      :meth:`update` or ``+=``.
    - **Subset**: create a new storage containing only objects whose
      substance matches a set of include/exclude filters.
    - **Search**: retrieve objects grouped by substance or component
      using :meth:`retrieve_by_substance` and
      :meth:`retrieve_by_component`.
    - **Remove**: delete individual stored objects with
      :meth:`remove_object`.

    Parameters
    ----------
    root_directory: str
        The directory in which all stored objects are located.
    cache_objects_in_memory: bool
        If True, objects will be cached in memory after they are
        retrieved from storage.
    """

    def update(self, other: LocalFileStorage) -> None:
        """Copy all objects from *other* into this storage.

        Existing objects are deduplicated automatically: for
        :class:`~openff.evaluator.storage.data.HashableStoredData`
        (e.g. force fields) the parent :meth:`store_object` skips
        storing if an identical object already exists.

        Parameters
        ----------
        other:
            Storage instance whose contents are merged into this one.
            *other* is not modified.
        """
        for _, keys in other._stored_object_keys.items():
            for key in keys:
                obj, ancillary = other.retrieve_object(key)
                self.store_object(obj, ancillary)

    def __iadd__(self, other: LocalFileStorage) -> "MutableLocalFileStorage":
        """Merge *other* into this storage in-place (``self += other``)."""
        self.update(other)
        return self

    def _store_object(
        self, object_to_store, storage_key=None, ancillary_data_path=None
    ):
        """Store *object_to_store*, **copying** any ancillary data directory
        rather than moving it, so the caller's source data is preserved.
        """
        file_path = path.join(self._root_directory, f"{storage_key}.json")

        with open(file_path, "w") as file:
            json.dump(object_to_store, file, cls=TypedJSONEncoder)

        if object_to_store.has_ancillary_data():
            directory_path = path.join(self._root_directory, f"{storage_key}")
            if path.isdir(directory_path):
                shutil.rmtree(directory_path, ignore_errors=True)
            shutil.copytree(ancillary_data_path, directory_path)

        if self._cache_objects_in_memory:
            self._cached_retrieved_objects[storage_key] = (
                object_to_store,
                ancillary_data_path,
            )
