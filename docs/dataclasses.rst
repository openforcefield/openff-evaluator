Data Classes and Queries
========================

All data which is to be stored within a ``StorageBackend`` must inherit from the ``BaseStoredData`` class. More broadly
there are typically two types of data which are expected to be stored:

* ``HashableStoredData`` - data which is readily hashable and can be quickly queried for in a storage backend.
  The prime examples of such data are ``ForceFieldData``, whose hash can be easily computed from the file
  representation of a force field.

* ``ReplaceableData`` - data which may be replaced in a storage backend, provided that another piece of data of the
  same type but which has a higher information is stored in the backend. An example of this is when storing a piece
  of ``StoredSimulationData`` in the backend which was generated for a particular ``Substance`` and at the same
  ``ThermodynamicState`` as an existing piece of data, but which stores many more uncorrelated configurations.

Every data class **must** be paired with a corresponding data query class which inherits from the ``BaseDataQuery``
class. A data query is a class which exposes a set of criteria for matching pieces of data against, such as the
``Substance`` that a piece of data was gathered for or the particular ``ThermodynamicState`` at which it was
generated.

Force Field Data
----------------


Cached Simulation Data
----------------------

