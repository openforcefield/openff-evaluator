Property API
==================================

This an api for....

Primary objects
---------------

.. currentmodule:: propertyestimator.properties
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PhysicalProperty

Built-in Properties
-------------------

.. currentmodule:: propertyestimator.properties
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Density
    DielectricConstant
    EnthalpyOfMixing

Workflow Components
-------------------

Schema

.. currentmodule:: propertyestimator.workflow.schemas
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    WorkflowSchema

Protocol API

.. currentmodule:: propertyestimator.workflow.protocols
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseProtocol
    ProtocolSchema
    ProtocolPath

Built in Protocols

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

Protocol Groups

.. currentmodule:: propertyestimator.workflow.groups
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ProtocolGroup
    ConditionalGroup