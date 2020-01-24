.. |protocol|           replace:: :py:class:`~propertyestimator.workflow.Protocol`
.. |protocol_schema|    replace:: :py:class:`~propertyestimator.workflow.schemas.ProtocolSchema`
.. |protocol_graph|     replace:: :py:class:`~propertyestimator.workflow.ProtocolGraph`
.. |protocol_path|      replace:: :py:class:`~propertyestimator.workflow.utils.ProtocolPath`
.. |workflow|           replace:: :py:class:`~propertyestimator.workflow.Workflow`
.. |workflow_schema|    replace:: :py:class:`~propertyestimator.workflow.schemas.WorkflowSchema`
.. |workflow_graph|     replace:: :py:class:`~propertyestimator.workflow.WorkflowGraph`
.. |workflow_result|    replace:: :py:class:`~propertyestimator.workflow.WorkflowResult`

Workflows
=========

The framework offers a lightweight workflow engine for executing graphs of tasks using the available :doc:`calculation
backends <calculationbackend>`. While lightweight, it offers a large amount of extensibility and flexibility, and is
currently used by both the :doc:`simulation <simulationlayer>` and :doc:`reweighting <reweightinglayer>` layers to
perform their required calculations.

Workflows are represented as |workflow| objects by the framework, which are themselves a collection of workflow
|protocol| objects.

A workflow is a collection of steps which, when coupled together, can estimate the values

Replicators
-----------
