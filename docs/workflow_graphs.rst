.. |workflow|           replace:: :py:class:`~propertyestimator.workflow.Workflow`
.. |workflow_graph|     replace:: :py:class:`~propertyestimator.workflow.WorkflowGraph`

Workflow Graphs
===============

A |workflow_graph| is a collection of |workflow| objects which should be executed together. The primary advantage of
executing workflows via the graph object is that the graph will automatically take advantage of the :doc:`protocols
<protocols>` built in redundancy / merging support to collapse duplicate tasks across multiple workflows.