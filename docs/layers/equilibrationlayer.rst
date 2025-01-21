.. |equilibration_layer|   replace:: :py:class:`~openff.evaluator.layers.equilibration.EquilibrationLayer`
.. |equilibration_schema|   replace:: :py:class:`~openff.evaluator.layers.equilibration.EquilibrationSchema`
.. |equilibration_property|   replace:: :py:class:`~openff.evaluator.layers.equilibration.EquilibrationProperty`
.. |stored_equilibration_data|   replace:: :py:class:`~openff.evaluator.storage.data.StoredEquilibrationData`
.. |equilibration_data_query|   replace:: :py:class:`~openff.evaluator.storage.query.EquilibrationDataQuery`

.. |observable_type|               replace:: :py:class:`~openff.evaluator.utils.observables.ObservableType`
.. |condition_aggregation_behavior| replace:: :py:class:`~openff.evaluator.workflow.attributes.ConditionAggregationBehavior`
.. |preequilibrated_simulation_layer|   replace:: :py:class:`~openff.evaluator.layers.preequilibrated_simulation.PreequilibratedSimulationLayer`
.. |workflow_calculation_schema|    replace:: :py:class:`~openff.evaluator.layers.workflow.WorkflowCalculationSchema`



The Equilibration Layer
=======================

The |equilibration_layer| is a modification of the |simulation_layer| which is designed to only
equilibrate systems. As such, it does not actually estimate properties, but rather is designed to be
computed prior to a |preequilibrated_simulation_layer| until chosen observables of the system
equilibrate within a certain tolerance.

StoredEquilibrationData
-----------------------

The |equilibration_layer| stores the final coordinates of the system in a |stored_equilibration_data| object
and in its ancillary directory.
This can be queried from the storage backend using a |equilibration_data_query| for the
|preequilibrated_simulation_layer|.


Equilibration properties
------------------------

The |equilibration_layer| is designed to equilibrate a system until a set of observables are within
a certain tolerance. The observables that are monitored are defined by setting the ``error_tolerances``
in the |equilibration_schema|. Each error tolerance should be an instance of |equilibration_property|.

The |equilibration_property| class is a simple data class that defines:

* The |observable_type| to monitor (``observable_type``, required)
* The ``absolute_tolerance`` or ``relative_tolerance`` of the observable.
  Both properties are optional, but both cannot be defined at the same time.
  The ``absolute_tolerance`` must be specified in units compatible with the observable.
  The ``relative_tolerance`` must be a float.
* The required ``n_uncorrelated_samples`` (optional). This is the number of uncorrelated samples
  required to sample.



Calculation Schema
------------------

The |equilibration_schema| builds off the |workflow_calculation_schema|. The schema defines:

* The |equilibration_property|s to monitor (``error_tolerances``)
* How to aggregate multiple |equilibration_property|s (``error_aggregration``).
  This should be of type |condition_aggregation_behavior| and can be either ``All`` or ``Any``.
  ``All`` requires all properties to be within tolerance, while ``Any`` requires only one property to be within tolerance.
* The maximum number of iterations to run equilibrations for (``max_iterations``).
  By default, each equilibration is 200 ps long.
* Whether to error on non-convergence, or store the simulated box anyway (``error_on_failure``).
  If ``error_on_failure`` is set to ``False``, the box will be stored even if the equilibration did not converge.

