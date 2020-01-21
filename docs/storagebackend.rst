Storage Backends
================

A ``StorageBackend`` is an object used to store data generated as part of property calculations, and to retrieve
that data for use in future calculations.

In general, most data stored in a storage backend is stored in two parts:

* A JSON serialized representation of this class (or a subclass), which contains lightweight information such as the
  state and composition of a system. Larger pieces of data, such as coordinates, trajectories or system objects, should
  be referenced as a file name.
* A directory like structure (either directly a directory, or some NetCDF like compressed archive) of ancillary files
  which do not easily lend themselves to be serialized within a JSON object, whose files are referenced by their file
  name by the data object.

The ancillary directory-like structure is not required if the data may be suitably stored in the data object itself.
See the :doc:`dataclasses` page for more information about the available data classes and their details.

Retrieving Data
---------------

Data may be retrieved from the storage system via two mechanisms:

* using the unique key which was assigned to the data when it was stored, or
* by using :doc:`data queries <dataclasses>` to search for data within the backend.

A data query is an object which exposes a set of search criteria, and is used to match data which has been stored
within a storage backend. A query may be used for example to match all data that was generated for a given
``Substance`` (or a substance which contains a particular component), or for a particular ``ThermodynamicState``.

Implementation
--------------

A ``StorageBackend`` must at minimum implement a structure of::

    class MyStorageBackend(StorageBackend):

        def _store_object(self, object_to_store, storage_key=None, ancillary_data_path=None):
            ...

        def _retrieve_object(self, storage_key, expected_type=None):
            ...

        def _object_exists(self, storage_key):
            ...

where

* ``_store_object`` must store a ``BaseStoredData`` object as well as optionally its ancillary data directory,
  and return a unique key assigned to that object.
* ``_retrieve_object`` must return the ``BaseStoredData`` object which has been assigned a given key if the
  object exists in the system, as well as the ancillary data directory if it exists.
* ``_object_exists`` should check whether any object still exists in the storage system with a given key.

All of these methods will be called under a `reentrant thread lock <https://docs.python.org/2/library/threading.
html#rlock-objects>`_ and may be considered as thread safe.
