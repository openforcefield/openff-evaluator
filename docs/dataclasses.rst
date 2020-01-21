.. |storage_backend|        replace:: :py:class:`~propertyestimator.storage.StorageBackend`

.. |base_data|              replace:: :py:class:`~propertyestimator.storage.data.BaseStoredData`
.. |base_query|             replace:: :py:class:`~propertyestimator.storage.query.BaseDataQuery`
.. |hashable_data|          replace:: :py:class:`~propertyestimator.storage.data.HashableStoredData`
.. |replaceable_data|       replace:: :py:class:`~propertyestimator.storage.data.ReplaceableData`

.. |force_field_data|       replace:: :py:class:`~propertyestimator.storage.data.ForceFieldData`
.. |force_field_query|      replace:: :py:class:`~propertyestimator.storage.query.ForceFieldQuery`

.. |simulation_data|        replace:: :py:class:`~propertyestimator.storage.data.StoredSimulationData`

.. |substance|              replace:: :py:class:`~propertyestimator.substances.Substance`
.. |thermodynamic_state|    replace:: :py:class:`~propertyestimator.thermodynamics.ThermodynamicState`
.. |force_field_source|     replace:: :py:class:`~propertyestimator.forcefield.ForceFieldSource`

Data Classes and Queries
========================

All data which is to be stored within a |storage_backend| must inherit from the |base_data| class. More broadly
there are typically two types of data which are expected to be stored:

* |hashable_data| - data which is readily hashable and can be quickly queried for in a storage backend.
  The prime examples of such data are |force_field_data|, whose hash can be easily computed from the file
  representation of a force field.

* |replaceable_data| - data which should be replaced in a storage backend when new data of the same type, but which
  has a higher information content, is stored in the backend. An example of this is when storing a piece
  of |simulation_data| in the backend which was generated for a particular |substance| and at the same
  |thermodynamic_state| as an existing piece of data, but which stores many more uncorrelated configurations.

Every data class **must** be paired with a corresponding data query class which inherits from the |base_query|
class. In addition, each data object must implement a ``to_storage_query`` function which returns the data query
which would uniquely match that data object. The ``to_storage_query`` is used heavily by storage backends when checking
if a piece of data already exists within the backend.

Force Field Data
----------------

The |force_field_data| class is used to |force_field_source| objects within the storage backend. It is a hashable
storage object which allows for rapidly checking whether any calculations have been previously been performed for
a particular force field source.

It has a corresponding |force_field_query| class which can be used to query for particular force field sources within
a storage backend.

Cached Simulation Data
----------------------

