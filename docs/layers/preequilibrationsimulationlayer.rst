.. |preequilibrated_simulation_layer|   replace:: :py:class:`~openff.evaluator.layers.preequilibrated_simulation.PreequilibratedSimulationLayer`
.. |preequilibrated_simulation_schema|  replace:: :py:class:`~openff.evaluator.layers.preequilibrated_simulation.PreequilibratedSimulationSchema`
.. |simulation_layer|                   replace:: :py:class:`~openff.evaluator.layers.simulation.SimulationLayer`
.. |workflow_calculation_layer|         replace:: :py:class:`~openff.evaluator.layers.workflow.WorkflowCalculationLayer`
.. |workflow_calculation_schema|    replace:: :py:class:`~openff.evaluator.layers.workflow.WorkflowCalculationSchema`

.. |storage_queries|                replace:: :py:attr:`~openff.evaluator.layers.reweighting.ReweightingSchema.storage_queries`
.. |simulation_data_query|          replace:: :py:class:`~openff.evaluator.storage.query.SimulationDataQuery`
.. |substance_attr|                 replace:: :py:attr:`~openff.evaluator.storage.query.SimulationDataQuery.substance`

.. |placeholder_value|              replace:: :py:class:`~openff.evaluator.attributes.PlaceholderValue`


The Preequilibrated Simulation Layer
====================================

The |preequilibrated_simulation_layer| is a modification of the |simulation_layer| which is designed to be used when
the system of interest has already been equilibrated. It functions very similarly to the |simulation_layer|, but
skips the box-packing step (`build_coordinates`), and instead uses the coordinates from a pre-equilibrated system.

As this layer relies on the existence of pre-equilibrated data, this data must already exist in the storage backend
(most commonly a directory called `stored_data`) and have been computed using an |equilibration_layer|.
Unless otherwise specified in the |storage_queries|, the following attributes must match:

- The number of molecules and overall substance
- thermodynamic state


Calculation Schema
------------------
The preequilibrated simulation layer will be provided with one |preequilibrated_simulation_schema| per type of property that it is being requested to
estimate. It builds off the base |workflow_calculation_schema| schema providing an additional |storage_queries|
attribute.

The |storage_queries| attribute will contain a dictionary of |simulation_data_query| which will be used by the layer to
access the data required for each property from the storage backend. Each key in this dictionary will correspond to the
key of a piece of metadata made available to the property workflows.


Default Metadata
----------------
The preequilibrated simulation layer makes available the default metadata provided by the :ref:`parent workflow layer
<layers/workflowlayer:Default Metadata>` in addition to any cached data retrieved via the schemas |storage_queries|.
