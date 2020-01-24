.. |protocol|           replace:: :py:class:`~propertyestimator.workflow.Protocol`
.. |protocol_schema|    replace:: :py:class:`~propertyestimator.workflow.schemas.ProtocolSchema`
.. |protocol_graph|     replace:: :py:class:`~propertyestimator.workflow.ProtocolGraph`
.. |protocol_path|      replace:: :py:class:`~propertyestimator.workflow.utils.ProtocolPath`
.. |workflow|           replace:: :py:class:`~propertyestimator.workflow.Workflow`
.. |workflow_schema|    replace:: :py:class:`~propertyestimator.workflow.schemas.WorkflowSchema`
.. |workflow_graph|     replace:: :py:class:`~propertyestimator.workflow.WorkflowGraph`
.. |workflow_result|    replace:: :py:class:`~propertyestimator.workflow.WorkflowResult`

.. |openmm_simulation|                       replace:: :py:class:`~propertyestimator.protocols.openmm.OpenMMSimulation`

Protocols
=========

The |protocol| class represents a task to be executed within some larger workflow. The task encoded by a protocol
may be as simple as adding two numbers together or even as complex as performing entire free energy simulations. In
general however a protocol should have a *single* well defined task to perform.

.. figure:: _static/img/protocol.jpg
    :align: center

    A selection of the inputs and outputs of the |openmm_simulation| protocol.

Each protocol exposes a set of required inputs, and the produced outputs. The value of each input may either be set
as a constant, or, may be set as the output of another protocol, such the inputs and outputs of protocols may be
chained together to form complex behaviours.

Implementation
--------------