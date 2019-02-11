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

Server Side API
---------------

.. currentmodule:: propertyestimator.server
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PropertyEstimatorServer
    PropertyEstimatorServerData

Physical Property API
---------------------

.. currentmodule:: propertyestimator.properties
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PhysicalProperty

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

    PropertyEstimatorBackendResources
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

**Built-in Backends**

.. currentmodule:: propertyestimator.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    LocalFileStorage

Workflow Components API
-----------------------

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
    BuildSmirnoffTopology
    RunEnergyMinimisation
    RunOpenMMSimulation
    AveragePropertyProtocol
    AverageTrajectoryProperty
    ExtractUncorrelatedData
    ExtractUncorrelatedTrajectoryData
    AddQuantities
    SubtractQuantities

**Protocol Groups**

.. currentmodule:: propertyestimator.workflow.groups
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ProtocolGroup
    ConditionalGroup