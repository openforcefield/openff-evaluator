"""
A mutable, extensible local file storage backend.
"""

from openff.evaluator.storage.localfile import LocalFileStorage


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
