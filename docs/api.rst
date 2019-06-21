API
===

A set of API documents for this projects classes and modules.

Client Side API
---------------

.. currentmodule:: propertyestimator.client
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PropertyEstimatorClient
    PropertyEstimatorOptions
    PropertyEstimatorSubmission
    PropertyEstimatorResult
    ConnectionOptions

Server Side API
---------------

.. currentmodule:: propertyestimator.server
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PropertyEstimatorServer

Physical Property API
---------------------

.. currentmodule:: propertyestimator
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    properties.PhysicalProperty
    substances.Substance
    substances.Mixture
    thermodynamics.ThermodynamicState

**Built-in Properties**

.. currentmodule:: propertyestimator.properties
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Density
    DielectricConstant
    EnthalpyOfMixing

Calculation Layers API
----------------------

.. currentmodule:: propertyestimator.layers
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PropertyCalculationLayer
    register_calculation_layer

**Built-in Layers**

.. currentmodule:: propertyestimator.layers
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    SurrogateLayer
    ReweightingLayer
    SimulationLayer

Calculation Backends API
------------------------

.. currentmodule:: propertyestimator.backends
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ComputeResources
    PropertyEstimatorBackend

**Built-in Backends**

.. currentmodule:: propertyestimator.backends
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    DaskLocalClusterBackend

Storage Backends API
--------------------

.. currentmodule:: propertyestimator.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PropertyEstimatorStorage
    StoredSimulationData

**Built-in Backends**

.. currentmodule:: propertyestimator.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    LocalFileStorage

Workflow API
------------

.. currentmodule:: propertyestimator.workflow
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Workflow
    WorkflowGraph

**Schema**

.. currentmodule:: propertyestimator.workflow.schemas
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    WorkflowSchema
    ProtocolSchema
    ProtocolGroupSchema
    ProtocolReplicator

**Protocol API**

.. currentmodule:: propertyestimator.workflow
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    protocols.BaseProtocol
    utils.ProtocolPath

**Built in Protocols**

.. currentmodule:: propertyestimator.workflow.protocols
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BuildCoordinatesPackmol
    BuildSmirnoffSystem
    RunEnergyMinimisation
    RunOpenMMSimulation
    AveragePropertyProtocol
    AverageTrajectoryProperty
    ExtractUncorrelatedData
    ExtractUncorrelatedTrajectoryData
    AddValues
    SubtractValues
    UnpackStoredSimulationData

**Protocol Groups**

.. currentmodule:: propertyestimator.workflow.groups
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ProtocolGroup
    ConditionalGroup