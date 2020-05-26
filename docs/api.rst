API
===

Documentation for each of the classes contained within the `openff.evaluator` framework.

Client Side API
---------------

.. currentmodule:: openff.evaluator.client
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

.. currentmodule:: openff.evaluator.utils.exceptions
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    EvaluatorException

Server Side API
---------------

.. currentmodule:: openff.evaluator.server
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    EvaluatorServer
    Batch

Physical Property API
---------------------

.. currentmodule:: openff.evaluator.datasets
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PhysicalProperty
    PropertyPhase
    Source
    CalculationSource
    MeasurementSource

**Built-in Properties**

.. currentmodule:: openff.evaluator.properties
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

.. currentmodule:: openff.evaluator.substances
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Substance
    Component
    Amount
    ExactAmount
    MoleFraction

**State Definition**

.. currentmodule:: openff.evaluator.thermodynamics
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ThermodynamicState

Data Set API
------------

.. currentmodule:: openff.evaluator.datasets
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PhysicalPropertyDataSet

**NIST ThermoML Archive**

.. currentmodule:: openff.evaluator.datasets.thermoml
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ThermoMLDataSet
    register_thermoml_property
    thermoml_property

Force Field API
---------------

.. currentmodule:: openff.evaluator.forcefield
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ForceFieldSource
    SmirnoffForceFieldSource
    TLeapForceFieldSource
    LigParGenForceFieldSource

**Gradient Estimation**

.. currentmodule:: openff.evaluator.forcefield
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ParameterGradientKey
    ParameterGradient

Calculation Layers API
----------------------

.. currentmodule:: openff.evaluator.layers
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

.. currentmodule:: openff.evaluator.layers.workflow
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    WorkflowCalculationLayer
    WorkflowCalculationSchema

.. currentmodule:: openff.evaluator.layers.simulation
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    SimulationLayer
    SimulationSchema

.. currentmodule:: openff.evaluator.layers.reweighting
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ReweightingLayer
    ReweightingSchema
    default_storage_query

Calculation Backends API
------------------------

.. currentmodule:: openff.evaluator.backends
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    CalculationBackend
    ComputeResources
    QueueWorkerResources

**Dask Backends**

.. currentmodule:: openff.evaluator.backends.dask
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseDaskBackend
    BaseDaskJobQueueBackend
    DaskLocalCluster
    DaskLSFBackend
    DaskPBSBackend

Storage API
-----------

.. currentmodule:: openff.evaluator.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    StorageBackend

**Built-in Storage Backends**

.. currentmodule:: openff.evaluator.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    LocalFileStorage

**Data Classes**

.. currentmodule:: openff.evaluator.storage.data
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseStoredData
    HashableStoredData
    ForceFieldData
    ReplaceableData
    StoredSimulationData

**Data Queries**

.. currentmodule:: openff.evaluator.storage.query
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseDataQuery
    SubstanceQuery
    ForceFieldQuery
    SimulationDataQuery

**Attributes**

.. currentmodule:: openff.evaluator.storage.attributes
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    FilePath
    StorageAttribute
    QueryAttribute


Workflow API
------------

.. currentmodule:: openff.evaluator.workflow
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

.. currentmodule:: openff.evaluator.workflow.schemas
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ProtocolSchema
    ProtocolGroupSchema
    ProtocolReplicator
    WorkflowSchema

**Attributes**

.. currentmodule:: openff.evaluator.workflow.attributes
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseMergeBehaviour
    MergeBehaviour
    InequalityMergeBehaviour
    InputAttribute
    OutputAttribute

*Placeholder Values*

.. currentmodule:: openff.evaluator.workflow.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ReplicatorValue
    ProtocolPath

Built-in Workflow Protocols
---------------------------

**Analysis**

.. currentmodule:: openff.evaluator.protocols.analysis
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

.. currentmodule:: openff.evaluator.protocols.coordinates
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BuildCoordinatesPackmol
    SolvateExistingStructure
    BuildDockedCoordinates

**Force Field Assignment**

.. currentmodule:: openff.evaluator.protocols.forcefield
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseBuildSystem
    BuildSmirnoffSystem
    BuildLigParGenSystem
    BuildTLeapSystem

**Gradients**

.. currentmodule:: openff.evaluator.protocols.gradients
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseGradientPotentials
    CentralDifferenceGradient

**pAPRika**

.. currentmodule:: propertyestimator.protocols.paprika
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BasePaprikaProtocol
    OpenMMPaprikaProtocol
    AmberPaprikaProtocol

**Groups**

.. currentmodule:: openff.evaluator.protocols.groups
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ConditionalGroup

**Miscellaneous**

.. currentmodule:: openff.evaluator.protocols.miscellaneous
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

.. currentmodule:: openff.evaluator.protocols.openmm
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    OpenMMEnergyMinimisation
    OpenMMSimulation
    OpenMMReducedPotentials
    OpenMMGradientPotentials

**Reweighting**

.. currentmodule:: openff.evaluator.protocols.reweighting
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ConcatenateTrajectories
    ConcatenateStatistics
    BaseReducedPotentials
    BaseMBARProtocol
    ReweightStatistics

**Simulation**

.. currentmodule:: openff.evaluator.protocols.simulation
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseEnergyMinimisation
    BaseSimulation

**Storage**

.. currentmodule:: openff.evaluator.protocols.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    UnpackStoredSimulationData

**YANK Free Energies**

.. currentmodule:: openff.evaluator.protocols.yank
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseYankProtocol
    LigandReceptorYankProtocol
    SolvationYankProtocol

Workflow Construction Utilities
-------------------------------

.. currentmodule:: openff.evaluator.protocols.utils
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

.. currentmodule:: openff.evaluator.attributes
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

.. currentmodule:: openff.evaluator.plugins
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    register_default_plugins
    register_external_plugins