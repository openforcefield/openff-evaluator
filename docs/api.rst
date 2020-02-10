API
===

Documentation for each of the classes contained within the evaluator framework.

Client Side API
---------------

.. currentmodule:: evaluator.client
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    EvaluatorClient
    BatchMode
    ConnectionOptions
    Request
    RequestOptions
    RequestResult

**Exceptions**

.. currentmodule:: evaluator.utils.exceptions
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    EvaluatorException

Server Side API
---------------

.. currentmodule:: evaluator.server
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    EvaluatorServer
    Batch

Physical Property API
---------------------

.. currentmodule:: evaluator.datasets
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PhysicalProperty
    PropertyPhase
    Source
    CalculationSource
    MeasurementSource

**Built-in Properties**

.. currentmodule:: evaluator.properties
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Density
    ExcessMolarVolume
    DielectricConstant
    EnthalpyOfMixing
    EnthalpyOfVaporization
    SolvationFreeEnergy
    HostGuestBindingAffinity

**Substance Definition**

.. currentmodule:: evaluator.substances
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Substance
    Component
    Amount
    ExactAmount
    MoleFraction

**State Definition**

.. currentmodule:: evaluator.thermodynamics
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ThermodynamicState

Data Set API
------------

.. currentmodule:: evaluator.datasets
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PhysicalPropertyDataSet

**NIST ThermoML Archive**

.. currentmodule:: evaluator.datasets.thermoml
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ThermoMLDataSet
    register_thermoml_property
    thermoml_property

Force Field API
---------------

.. currentmodule:: evaluator.forcefield
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ForceFieldSource
    SmirnoffForceFieldSource
    TLeapForceFieldSource
    LigParGenForceFieldSource

**Gradient Estimation**

.. currentmodule:: evaluator.forcefield
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ParameterGradientKey
    ParameterGradient

Calculation Layers API
----------------------

.. currentmodule:: evaluator.layers
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    CalculationLayer
    CalculationLayerResult
    CalculationLayerSchema
    calculation_layer
    register_calculation_layer
    register_calculation_schema

**Built-in Calculation Layers**

.. currentmodule:: evaluator.layers.workflow
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    WorkflowCalculationLayer
    WorkflowCalculationSchema

.. currentmodule:: evaluator.layers.simulation
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    SimulationLayer
    SimulationSchema

.. currentmodule:: evaluator.layers.reweighting
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ReweightingLayer
    ReweightingSchema
    default_storage_query

Calculation Backends API
------------------------

.. currentmodule:: evaluator.backends
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    CalculationBackend
    ComputeResources

**Dask Backends**

.. currentmodule:: evaluator.backends.dask
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseDaskBackend
    BaseDaskJobQueueBackend
    DaskLocalCluster
    DaskLSFBackend
    DaskPBSBackend
    QueueWorkerResources

Storage API
-----------

.. currentmodule:: evaluator.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    StorageBackend

**Built-in Storage Backends**

.. currentmodule:: evaluator.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    LocalFileStorage

**Data Classes**

.. currentmodule:: evaluator.storage.data
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseStoredData
    HashableStoredData
    ForceFieldData
    ReplaceableData
    StoredSimulationData

**Data Queries**

.. currentmodule:: evaluator.storage.query
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseDataQuery
    SubstanceQuery
    ForceFieldQuery
    SimulationDataQuery

**Attributes**

.. currentmodule:: evaluator.storage.attributes
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    FilePath
    StorageAttribute
    QueryAttribute


Workflow API
------------

.. currentmodule:: evaluator.workflow
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Workflow
    WorkflowException
    WorkflowGraph
    WorkflowResult
    Protocol
    ProtocolGraph
    ProtocolGroup
    workflow_protocol
    register_workflow_protocol

**Schemas**

.. currentmodule:: evaluator.workflow.schemas
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ProtocolSchema
    ProtocolGroupSchema
    ProtocolReplicator
    WorkflowSchema

**Attributes**

.. currentmodule:: evaluator.workflow.attributes
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseMergeBehaviour
    MergeBehaviour
    InequalityMergeBehaviour
    InputAttribute
    OutputAttribute

*Placeholder Values*

.. currentmodule:: evaluator.workflow.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ReplicatorValue
    ProtocolPath

Built-in Workflow Protocols
---------------------------

**Analysis**

.. currentmodule:: evaluator.protocols.analysis
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    AveragePropertyProtocol
    AverageTrajectoryProperty
    ExtractAverageStatistic
    ExtractUncorrelatedData
    ExtractUncorrelatedTrajectoryData
    ExtractUncorrelatedStatisticsData

**Coordinate Generation**

.. currentmodule:: evaluator.protocols.coordinates
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BuildCoordinatesPackmol
    SolvateExistingStructure
    BuildDockedCoordinates

**Force Field Assignment**

.. currentmodule:: evaluator.protocols.forcefield
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseBuildSystem
    BuildSmirnoffSystem
    BuildLigParGenSystem
    BuildTLeapSystem

**Gradients**

.. currentmodule:: evaluator.protocols.gradients
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseGradientPotentials
    CentralDifferenceGradient

**Groups**

.. currentmodule:: evaluator.protocols.groups
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ConditionalGroup

**Miscellaneous**

.. currentmodule:: evaluator.protocols.miscellaneous
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    AddValues
    SubtractValues
    MultiplyValue
    DivideValue
    WeightByMoleFraction
    FilterSubstanceByRole

**OpenMM**

.. currentmodule:: evaluator.protocols.openmm
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    OpenMMEnergyMinimisation
    OpenMMSimulation
    OpenMMReducedPotentials
    OpenMMGradientPotentials

**Reweighting**

.. currentmodule:: evaluator.protocols.reweighting
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ConcatenateTrajectories
    ConcatenateStatistics
    BaseReducedPotentials
    BaseMBARProtocol
    ReweightStatistics

**Simulation**

.. currentmodule:: evaluator.protocols.simulation
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseEnergyMinimisation
    BaseSimulation

**Storage**

.. currentmodule:: evaluator.protocols.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    UnpackStoredSimulationData

**YANK Free Energies**

.. currentmodule:: evaluator.protocols.yank
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseYankProtocol
    LigandReceptorYankProtocol
    SolvationYankProtocol

Workflow Construction Utilities
-------------------------------

.. currentmodule:: evaluator.protocols.utils
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

.. currentmodule:: evaluator.attributes
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Attribute
    AttributeClass
    UNDEFINED
    PlaceholderValue

Plug-in Utilities
-----------------

**Plug-ins**

.. currentmodule:: evaluator.plugins
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    register_default_plugins
    register_external_plugins