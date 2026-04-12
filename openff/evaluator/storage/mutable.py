"""
A mutable, extensible local file storage backend.
"""

import json
import os
import shutil
from os import path
from typing import TYPE_CHECKING, Iterable

from openff.evaluator.storage.localfile import LocalFileStorage
from openff.evaluator.utils.serialization import TypedJSONEncoder

if TYPE_CHECKING:
    from openff.evaluator.substances import Substance
    from openff.evaluator.substances.components import Component


class MutableLocalFileStorage(LocalFileStorage):
    """A :class:`LocalFileStorage` subclass that adds mutating and
    compositional operations not present in the base class:

    - **Copy-on-store**: ancillary data directories are *copied* rather
      than moved, so the caller's original data is preserved.
    - **Combine**: merge another storage into this one via
      :meth:`update` or ``+=``.
    - **Subset**: create a new storage containing only objects whose
      substance matches a set of include/exclude filters.

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
        with other._lock:
            keys_snapshot = [
                key for keys in other._stored_object_keys.values() for key in list(keys)
            ]
        for key in keys_snapshot:
            obj, ancillary = other.retrieve_object(key)
            self.store_object(obj, ancillary)

    def __iadd__(self, other: LocalFileStorage) -> "MutableLocalFileStorage":
        """Merge *other* into this storage in-place (``self += other``)."""
        self.update(other)
        return self

    def _remove_object(self, storage_key: str) -> None:
        """Delete the JSON file and ancillary directory for *storage_key*."""
        file_path = path.join(self._root_directory, f"{storage_key}.json")
        directory_path = path.join(self._root_directory, f"{storage_key}")

        if path.isfile(file_path):
            os.remove(file_path)
        if path.isdir(directory_path):
            shutil.rmtree(directory_path)

    def remove_object(self, storage_key: str) -> None:
        """Remove a stored object by *storage_key*.

        Removes the object from the key registry, hash index, and
        in-memory cache, then deletes its files from disk.

        Parameters
        ----------
        storage_key:
            The storage key returned by :meth:`store_object`.

        Raises
        ------
        KeyError
            If *storage_key* is not registered in this storage.
        """
        with self._lock:
            # Find which type owns this key.
            type_name_found = None
            for type_name, keys in self._stored_object_keys.items():
                if storage_key in keys:
                    type_name_found = type_name
                    break

            if type_name_found is None:
                raise KeyError(
                    f"No stored object with key {storage_key!r} found in this storage."
                )

            # Remove from key registry.
            self._stored_object_keys[type_name_found].remove(storage_key)

            # Remove from hash index (any hash pointing to this key).
            hashes_to_remove = [
                h for h, k in self._object_hashes.items() if k == storage_key
            ]
            for h in hashes_to_remove:
                del self._object_hashes[h]

            # Remove from in-memory cache.
            self._cached_retrieved_objects.pop(storage_key, None)

            # Delete files from disk.
            self._remove_object(storage_key)

            # Persist the updated key registry.
            self._save_stored_object_keys()

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
            cached_ancillary = (
                path.join(self._root_directory, f"{storage_key}")
                if object_to_store.has_ancillary_data()
                else None
            )
            self._cached_retrieved_objects[storage_key] = (
                object_to_store,
                cached_ancillary,
            )

    @staticmethod
    def _substance_passes_filters(
        substance: "Substance | None",
        include_substances: "Iterable[Substance] | None",
        include_components: "Iterable[Component] | None",
        exclude_substances: "Iterable[Substance] | None",
        exclude_components: "Iterable[Component] | None",
    ) -> bool:
        """Return True if *substance* passes all active filters.

        Objects whose substance is ``None`` (e.g. :class:`ForceFieldData`)
        are included only when no *include* filters are active.
        """
        if substance is None:
            return include_substances is None and include_components is None

        component_ids = {c.identifier for c in substance.components}

        if include_substances is not None:
            if not any(substance == s for s in include_substances):
                return False

        if include_components is not None:
            if not all(c.identifier in component_ids for c in include_components):
                return False

        if exclude_substances is not None:
            if any(substance == s for s in exclude_substances):
                return False

        if exclude_components is not None:
            if any(c.identifier in component_ids for c in exclude_components):
                return False

        return True

    def subset(
        self,
        directory: str,
        include_substances: "Iterable[Substance] | None" = None,
        include_components: "Iterable[Component] | None" = None,
        exclude_substances: "Iterable[Substance] | None" = None,
        exclude_components: "Iterable[Component] | None" = None,
        require_empty_directory: bool = True,
    ) -> "MutableLocalFileStorage":
        """Return a new :class:`MutableLocalFileStorage` at *directory*
        containing only objects that match the given filters.

        Parameters
        ----------
        directory:
            Root directory for the new storage.
        include_substances:
            If given, only include objects whose substance exactly matches
            one of the listed
            :class:`~openff.evaluator.substances.Substance` instances.
        include_components:
            If given, only include objects whose substance contains
            *every* listed
            :class:`~openff.evaluator.substances.components.Component`.
        exclude_substances:
            Exclude objects whose substance exactly matches any listed
            substance.
        exclude_components:
            Exclude objects whose substance contains *any* listed
            component.
        require_empty_directory:
            If ``True`` (the default), raise a :exc:`ValueError` when
            *directory* already exists and is non-empty, ensuring the
            returned storage contains only the filtered objects.  Set to
            ``False`` to allow merging into a pre-populated directory.

        Returns
        -------
        MutableLocalFileStorage
            A new storage instance containing the matching objects.
        """
        if require_empty_directory and path.isdir(directory) and os.listdir(directory):
            raise ValueError(
                f"Target directory {directory!r} already exists and is non-empty. "
                "Pass an empty or non-existent directory, or set "
                "require_empty_directory=False to allow merging."
            )

        if include_substances and exclude_substances:
            overlap = [
                s for s in include_substances if any(s == e for e in exclude_substances)
            ]
            if overlap:
                raise ValueError(
                    "The same substance cannot appear in both include_substances "
                    "and exclude_substances."
                )

        if include_components is not None and exclude_components is not None:
            inc_ids = {c.identifier for c in include_components}
            exc_ids = {c.identifier for c in exclude_components}
            if inc_ids & exc_ids:
                raise ValueError(
                    "The same component cannot appear in both include_components "
                    "and exclude_components."
                )

        result = MutableLocalFileStorage(directory)

        with self._lock:
            keys_snapshot = [
                key for keys in self._stored_object_keys.values() for key in list(keys)
            ]
        for key in keys_snapshot:
            obj, ancillary = self.retrieve_object(key)
            substance = getattr(obj, "substance", None)

            if not self._substance_passes_filters(
                substance,
                include_substances,
                include_components,
                exclude_substances,
                exclude_components,
            ):
                continue

            result.store_object(obj, ancillary)

        return result
