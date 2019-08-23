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

**Substance Definition**

.. currentmodule:: propertyestimator.substances
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Substance

**State Definition**

.. currentmodule:: propertyestimator.thermodynamics
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ThermodynamicState

Data Set API
------------

.. currentmodule:: propertyestimator.datasets
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PhysicalPropertyDataSet

**NIST ThermoML Archive**

.. currentmodule:: propertyestimator.datasets
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ThermoMLDataSet
    register_thermoml_property

Calculation Layers API
----------------------

.. currentmodule:: propertyestimator.layers
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PropertyCalculationLayer
    register_calculation_layer

**Built-in Calculation Layers**

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

    PropertyEstimatorBackend
    ComputeResources
    QueueWorkerResources

**Dask Backends**

.. currentmodule:: propertyestimator.backends
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    DaskLocalCluster
    DaskLSFCluster

Storage Backends API
--------------------

.. currentmodule:: propertyestimator.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PropertyEstimatorStorage
    StoredSimulationData

**Built-in Storage Backends**

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

Built-in Workflow Protocols
---------------------------

**Coordinate Generation**

.. currentmodule:: propertyestimator.protocols.coordinates
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BuildCoordinatesPackmol
    SolvateExistingStructure
    BuildDockedCoordinates

**Force Field Assignment**

.. currentmodule:: propertyestimator.protocols.forcefield
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BuildSmirnoffSystem

**Simulation**

.. currentmodule:: propertyestimator.protocols.simulation
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    RunEnergyMinimisation
    RunOpenMMSimulation
    BaseYankProtocol
    LigandReceptorYankProtocol

**Simulation Analysis**

.. currentmodule:: propertyestimator.protocols.analysis
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    AveragePropertyProtocol
    AverageTrajectoryProperty
    ExtractAverageStatistic
    ExtractUncorrelatedData
    ExtractUncorrelatedTrajectoryData
<<<<<<< HEAD
    ExtractUncorrelatedStatisticsData

**Reweighting**

.. currentmodule:: propertyestimator.protocols.reweighting
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

=======
    AddValues
    SubtractValues
>>>>>>> origin/master
    UnpackStoredSimulationData
    ConcatenateTrajectories
    CalculateReducedPotentialOpenMM
    ReweightWithMBARProtocol

**Groups**

.. currentmodule:: propertyestimator.protocols.groups
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ProtocolGroup
    ConditionalGroup


**Miscellaneous**

.. currentmodule:: propertyestimator.protocols.miscellaneous
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    AddQuantities
    SubtractQuantities
    FilterSubstanceByRole
