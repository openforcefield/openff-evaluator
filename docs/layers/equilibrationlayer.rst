.. |equilibration_layer|   replace:: :py:class:`~openff.evaluator.layers.equilibration.EquilibrationLayer`
.. |equilibration_schema|   replace:: :py:class:`~openff.evaluator.layers.equilibration.EquilibrationSchema`

.. |preequilibrated_simulation_layer|   replace:: :py:class:`~openff.evaluator.layers.preequilibrated_simulation.PreequilibratedSimulationLayer`
.. |workflow_calculation_schema|    replace:: :py:class:`~openff.evaluator.layers.workflow.WorkflowCalculationSchema`

.. |storage_queries|                replace:: :py:attr:`~openff.evaluator.layers.reweighting.ReweightingSchema.storage_queries`
.. |simulation_data_query|          replace:: :py:class:`~openff.evaluator.storage.query.SimulationDataQuery`
.. |substance_attr|                 replace:: :py:attr:`~openff.evaluator.storage.query.SimulationDataQuery.substance`

.. |placeholder_value|              replace:: :py:class:`~openff.evaluator.attributes.PlaceholderValue`


The Equilibration Layer
====================================

The |equilibration_layer| is a modification of the |simulation_layer| which is designed to only
equilibrate systems. As such, it does not actually estimate properties, but rather is designed to be
computed prior to a |preequilibrated_simulation_layer| until the potential energy of the system
equilibrates within a certain tolerance.



Calculation Schema
------------------

The |equilibration_schema| builds off the |workflow_calculation_schema|.
As it is designed to equilibrate systems, it is recommended to set the ``absolute_tolerance``
attribute to a reasonable value for the potential energy to ensure equilibration;
otherwise, only the default equilibration length will be run.

.. note::

    Only the ``absolute_tolerance`` attribute of the |equilibration_schema| is supported.
    The ``relative_tolerance`` attribute is not supported.

