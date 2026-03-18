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

    def retrieve_by_substance(self, substance):
        """Return all stored objects whose substance equals *substance*.

        Parameters
        ----------
        substance:
            The :class:`~openff.evaluator.substances.Substance` to match
            exactly.

        Returns
        -------
        dict
            ``{substance: [(storage_key, data_object, ancillary_path), ...]}``.
            Returns an empty dict when nothing matches.
        """
        matches = []
        for type_name, keys in self._stored_object_keys.items():
            for key in list(keys):
                obj, ancillary = self.retrieve_object(key)
                obj_substance = getattr(obj, "substance", None)
                if obj_substance is not None and obj_substance == substance:
                    matches.append((key, obj, ancillary))
        if not matches:
            return {}
        return {substance: matches}

    def retrieve_by_component(self, component):
        """Return all stored objects whose substance contains *component*.

        Results are grouped by the matched substance.

        Parameters
        ----------
        component:
            The :class:`~openff.evaluator.substances.components.Component`
            to search for.

        Returns
        -------
        dict
            ``{substance: [(storage_key, data_object, ancillary_path), ...]}``.
            Returns an empty dict when nothing matches.
        """
        result = {}
        for type_name, keys in self._stored_object_keys.items():
            for key in list(keys):
                obj, ancillary = self.retrieve_object(key)
                substance = getattr(obj, "substance", None)
                if substance is None:
                    continue
                component_ids = {c.identifier for c in substance.components}
                if component.identifier in component_ids:
                    result.setdefault(substance, []).append((key, obj, ancillary))
        return result
