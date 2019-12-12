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

**Force Field Sources**

.. currentmodule:: propertyestimator.forcefield
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ForceFieldSource
    SmirnoffForceFieldSource
    TLeapForceFieldSource
    LigParGenForceFieldSource

**Gradient Estimation**

.. currentmodule:: propertyestimator.properties
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ParameterGradientKey
    ParameterGradient


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
    PropertyPhase
    Source
    MeasurementSource
    CalculationSource

**Built-in Properties**

.. currentmodule:: propertyestimator.properties
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Density
    ExcessMolarVolume
    DielectricConstant
    EnthalpyOfMixing
    EnthalpyOfVaporization
    HostGuestBindingAffinity

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

**Metadata**

.. currentmodule:: propertyestimator.properties
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PropertyPhase
    Source
    MeasurementSource
    CalculationSource

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
    thermoml_property

Calculation Layers API
----------------------

.. currentmodule:: propertyestimator.layers
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    CalculationLayer
    register_calculation_layer

**Built-in Calculation Layers**

.. currentmodule:: propertyestimator.layers
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

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

    BaseDaskBackend
    DaskLocalCluster
    DaskLSFBackend

Storage Backends API
--------------------

.. currentmodule:: propertyestimator.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    StorageBackend

**Built-in Storage Backends**

.. currentmodule:: propertyestimator.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    LocalFileStorage

**Data Classes**

.. currentmodule:: propertyestimator.storage.dataclasses
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseStoredData
    StoredSimulationData
    StoredDataCollection

Workflow API
------------

.. currentmodule:: propertyestimator.workflow
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Workflow
    WorkflowGraph
    WorkflowOptions
    IWorkflowProperty

**Schema**

.. currentmodule:: propertyestimator.workflow.schemas
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    WorkflowSchema
    ProtocolSchema
    ProtocolGroupSchema
    ProtocolReplicator
    WorkflowOutputToStore
    WorkflowSimulationDataToStore
    WorkflowDataCollectionToStore

**Base Protocol API**

.. currentmodule:: propertyestimator.workflow.protocols
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseProtocol

*Input / Output Utilities*

.. currentmodule:: propertyestimator.workflow.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ReplicatorValue
    ProtocolPath

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
    BuildTLeapSystem

**Simulation**

.. currentmodule:: propertyestimator.protocols.simulation
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    RunEnergyMinimisation
    RunOpenMMSimulation

**YANK Free Energies**

.. currentmodule:: propertyestimator.protocols.yank
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseYankProtocol
    LigandReceptorYankProtocol
    SolvationYankProtocol

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
    ExtractUncorrelatedStatisticsData

**Reweighting**

.. currentmodule:: propertyestimator.protocols.reweighting
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ConcatenateTrajectories
    ConcatenateStatistics
    CalculateReducedPotentialOpenMM
    BaseMBARProtocol
    ReweightStatistics

**Gradients**

.. currentmodule:: propertyestimator.protocols.gradients
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    GradientReducedPotentials
    CentralDifferenceGradient

**Groups**

.. currentmodule:: propertyestimator.protocols.groups
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ProtocolGroup
    ConditionalGroup


**Storage**

.. currentmodule:: propertyestimator.protocols.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/


    UnpackStoredDataCollection
    UnpackStoredSimulationData


**Miscellaneous**

.. currentmodule:: propertyestimator.protocols.miscellaneous
.. autosummary::
    :nosignatures:
    :toctree: api/generated/


    AddValues
    SubtractValues
    MultiplyValue
    DivideValue
    FilterSubstanceByRole
    BaseWeightByMoleFraction
    WeightByMoleFraction

Workflow Construction Utilities
-------------------------------

.. currentmodule:: propertyestimator.protocols.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseReweightingProtocols
    BaseSimulationProtocols
    generate_base_reweighting_protocols
    generate_base_simulation_protocols
    generate_gradient_protocol_group

Attribute Utilities
-------------------

.. currentmodule:: propertyestimator.attributes
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    AttributeClass
    Attribute
    InputAttribute
    OutputAttribute
    MergeBehaviour
    InequalityMergeBehaviour
    PlaceholderInput
