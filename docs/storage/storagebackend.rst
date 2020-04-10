.. |storage_backend|     replace:: :py:class:`~evaluator.storage.StorageBackend`
.. |base_stored_data|    replace:: :py:class:`~evaluator.storage.data.BaseStoredData`
.. |base_data_query|     replace:: :py:class:`~evaluator.storage.query.BaseDataQuery`
.. |substance|           replace:: :py:class:`~evaluator.substances.Substance`

.. |_store_object|       replace:: :py:meth:`~evaluator.storage.StorageBackend._store_object`
.. |_retrieve_object|    replace:: :py:meth:`~evaluator.storage.StorageBackend._retrieve_object`
.. |_object_exists|      replace:: :py:meth:`~evaluator.storage.StorageBackend._object_exists`



Storage Backends
================

A |storage_backend| is an object used to store data generated as part of property calculations, and to retrieve that
data for use in future calculations.

In general, most data stored in a storage backend is stored in two parts:

* A JSON serialized representation of this class (or a subclass), which contains lightweight information such as the
  state and composition of a system.
* A directory like structure (either directly a directory, or some NetCDF like compressed archive) of ancillary files
  which do not easily lend themselves to be serialized within a JSON object, such as simulation trajectories, whose
  files are referenced by their file name by the data object.

The ancillary directory-like structure is not required if the data may be suitably stored in the data object itself.

Data Storage / Retrieval
------------------------

Each piece of data which is stored in a backend must inherit from the |base_stored_data| class, will be assigned a
unique key. This unique key is both useful for tracking provenance if this data is re-used in future calculations, and
also can be used to retrieve the piece of data from the storage system.

In addition to retrieval using the data keys, each backend offers the ability to perform a 'query' to retrieve data
which matches a set of given criteria. Data queries are implemented via |base_data_query| objects, which expose
different options for querying for specific types of data (such a simulation data, trained models, etc.).

A query may be used for example to match all simulation data that was generated for a given |substance| in a
particular phase::

    # Look for all simulation data generated for liquid water
    substance_query = SimulationDataQuery()

    substance_query.substance = Substance.from_components("O")
    substance_query.property_phase = PropertyPhase.Liquid

    found_data = backend.query(substance_query)

The returned ``found_data`` will be a dictionary with keys of tuples and values as lists of tuples. Each key will be a
tuple of the values which were matched, for example the matched thermodynamic state, or the matched substance. For each
value tuple in the tuple list, the first item in the tuple is the unique key of the found data object, the second item
is the data object itself, and the final object is the file path to the ancillary data directory (or :py:class:`None`
if none is present).

See the :doc:`dataclasses` page for more information about the available data classes, queries and their details.

Implementation
--------------

A |storage_backend| must at minimum implement a structure of::

    class MyStorageBackend(StorageBackend):

        def _store_object(self, object_to_store, storage_key=None, ancillary_data_path=None):
            ...

        def _retrieve_object(self, storage_key, expected_type=None):
            ...

        def _object_exists(self, storage_key):
            ...

where

.. rst-class:: spaced-list

    * |_store_object| must store a |base_stored_data| object as well as optionally its ancillary data directory, and
      return a unique key assigned to that object.
    * |_retrieve_object| must return the |base_stored_data| object which has been assigned a given key if the object
      exists in the system, as well as the file path to ancillary data directory if it exists.
    * |_object_exists| should return whether any object still exists in the storage system with a given key.

All of these methods will be called under a `reentrant thread lock <https://docs.python.org/2/library/threading.
html#rlock-objects>`_ and may be considered as thread safe.
