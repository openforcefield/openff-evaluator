.. |reweighting_layer|              replace:: :py:class:`~propertyestimator.layers.reweighting.ReweightingLayer`
.. |reweighting_schema|             replace:: :py:class:`~propertyestimator.layers.reweighting.ReweightingSchema`
.. |workflow_calculation_layer|     replace:: :py:class:`~propertyestimator.layers.workflow.WorkflowCalculationLayer`
.. |workflow_calculation_schema|    replace:: :py:class:`~propertyestimator.layers.workflow.WorkflowCalculationSchema`

.. |storage_queries|                replace:: :py:attr:`~propertyestimator.layers.reweighting.ReweightingSchema.storage_queries`
.. |simulation_data_query|          replace:: :py:class:`~propertyestimator.storage.query.SimulationDataQuery`
.. |substance_attr|                 replace:: :py:attr:`~propertyestimator.storage.query.SimulationDataQuery.substance`

.. |placeholder_value|              replace:: :py:class:`~propertyestimator.attributes.PlaceholderValue`

The MBAR Reweighting Layer
==========================

The |reweighting_layer| is a calculation layer which employs the `Multistate Bennett Acceptance Ratio <http://www.
alchemistry.org/wiki/Multistate_Bennett_Acceptance_Ratio>`_ (MBAR) method to calculate observables at states which have
not been previously simulated, but for which simulations have been previously run at similar states and their data
cached. It inherits the |workflow_calculation_layer| base layer, and primarily makes use of the built-in
:doc:`workflow <workflows>` engine to perform the required calculations.

Because MBAR is a technique which reprocesses exisiting simulation data rather than re-running new simulations, it is
typically several fold faster than the :doc:`simulation layer <simulationlayer>` provided it has cached simulation data
(made accessible via a :doc:`storage backend <storagebackend>`) available. Any properties for which the required data
(see :ref:`reweightinglayer:Calculation Schema`) will be skipped.

Theory
------
Lorem ipsum

Calculation Schema
------------------
The reweighting layer will be provided with one |reweighting_schema| per type of property that it is being requested to
estimate. It builds off of the base |workflow_calculation_schema| schema providing an additional |storage_queries|
attribute.

The |storage_queries| attribute will contain a dictionary of |simulation_data_query| which will be used by the layer to
access the data required for each property from the storage backend. Each key in this dictionary will correspond to the
key of a piece of metadata made available to the property workflows.

Default Metadata
----------------
The reweighting layer makes available the default metadata provided by the :ref:`parent workflow layer
<workflowlayer:Default Metadata>` in addition to any cached data retrieved via the schemas |storage_queries|.

When building the metadata for each property, a copy of the query will be made and any of the supported attributes
(currently only |substance_attr|) whose values are set as |placeholder_value| objects will have their values updated
using values directly from the property. This query will then be passed to the storage backend to retrieve any matching
data.

The matching data will be stored as a list of tuples of the form::

    (object_path, data_directory, force_field_path)

where ``object_path`` is the file path to the stored dataclass, the ``data_directory`` is the file path to the ancillary
data directory and ``force_field_path`` is the file path to the force field parameters which were used to generate the
data originally.

This list of tuples will be made available as metadata under the key that was associated with the query.