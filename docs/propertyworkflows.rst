.. _propertyworkflows ::

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

.. currentmodule:: propertyestimator.properties.density
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Density

.. currentmodule:: propertyestimator.properties.dielectric
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    DielectricConstant

Workflow Components
-------------------

Schema

.. currentmodule:: propertyestimator.estimator.workflow.schema
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    CalculationSchema

Protocol API

.. currentmodule:: propertyestimator.estimator.workflow.protocols
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseProtocol
    ProtocolSchema
    ProtocolPath

Built in Protocols

.. currentmodule:: propertyestimator.estimator.workflow.protocols
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

Protocol Groups

.. currentmodule:: propertyestimator.estimator.workflow.groups
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ProtocolGroup
    ConditionalGroup