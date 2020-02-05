.. |local_file_storage|    replace:: :py:class:`~propertyestimator.storage.LocalFileStorage`

Local File Storage
==================

The |local_file_storage| backend stores and retrieves all data objects to / from the local file system. The root
directory in which all data is to be stored is defined when the object is created::

    storage_backend = LocalFileStorage(root_directory="stored_data")

All data objects will be stored within this directory as JSON files, with file names of the storage key assigned to
that object. If the data object has an associated ancillary data directory, this will be **moved** (not copied) into
the root directory and renamed to the storage key when that object is stored into the system.

An example directory created by a local storage backend will look something similar to::

    - root_directory

        - 1fe615c5cb48429ab77fd71125dec297
            - trajectory.dcd
            - statistics.csv

        - 3e15d19e0e614d0491a1a0bc9a51534e
            - trajectory.dcd
            - statistics.csv

        - 1fe615c5cb48429ab77fd71125dec297.json
        - 3e15d19e0e614d0491a1a0bc9a51534e.json
        - 0f71f2b4a22042d89d6f0882406869b6.json

where here the backend contains two data objects with ancillary data directories, and one without.

When retrieving data which has an ancillary data directory from the backend, the returned directory path will be the
full path to the directory in the root storage directory.
