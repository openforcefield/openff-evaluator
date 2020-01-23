API
===

Documentation for each of the classes contained within the evaluator framework.

Client Side API
---------------

.. currentmodule:: propertyestimator.client
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ConnectionOptions
    EvaluatorClient
    Request
    RequestOptions
    RequestResult

**Exceptions**

.. currentmodule:: propertyestimator.utils.exceptions
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    EvaluatorException

Server Side API
---------------

.. currentmodule:: propertyestimator.server
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    EvaluatorServer

Physical Property API
---------------------

.. currentmodule:: propertyestimator.datasets
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PhysicalProperty
    PropertyPhase
    Source
    CalculationSource
    MeasurementSource

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
    SolvationFreeEnergy
    HostGuestBindingAffinity

**Substance Definition**

.. currentmodule:: propertyestimator.substances
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Substance
    Component
    Amount
    ExactAmount
    MoleFraction

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

.. currentmodule:: propertyestimator.datasets.thermoml
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ThermoMLDataSet
    register_thermoml_property
    thermoml_property

Force Field API
---------------

.. currentmodule:: propertyestimator.forcefield
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ForceFieldSource
    SmirnoffForceFieldSource
    TLeapForceFieldSource
    LigParGenForceFieldSource

**Gradient Estimation**

.. currentmodule:: propertyestimator.forcefield
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ParameterGradientKey
    ParameterGradient

Calculation Layers API
----------------------

.. currentmodule:: propertyestimator.layers
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    CalculationLayer
    CalculationLayerResult
    CalculationLayerSchema
    calculation_layer
    register_calculation_layer
    register_calculation_schema
    registered_calculation_layers
    registered_calculation_schemas

**Built-in Calculation Layers**

.. currentmodule:: propertyestimator.layers.workflow
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    WorkflowCalculationLayer
    WorkflowCalculationSchema

.. currentmodule:: propertyestimator.layers.simulation
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    SimulationLayer
    SimulationSchema

.. currentmodule:: propertyestimator.layers.reweighting
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ReweightingLayer
    ReweightingSchema
    default_storage_query

Calculation Backends API
------------------------

.. currentmodule:: propertyestimator.backends
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    CalculationBackend
    ComputeResources

**Dask Backends**

.. currentmodule:: propertyestimator.backends.dask
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

.. currentmodule:: propertyestimator.storage.data
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseStoredData
    HashableStoredData
    ForceFieldData
    ReplaceableData
    StoredSimulationData

**Data Queries**

.. currentmodule:: propertyestimator.storage.query
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseDataQuery
    SubstanceQuery
    ForceFieldQuery
    SimulationDataQuery

**Attributes**

.. currentmodule:: propertyestimator.storage.attributes
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    FilePath
    StorageAttribute
    QueryAttribute


Workflow API
------------

.. currentmodule:: propertyestimator.workflow
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
    registered_workflow_protocols

**Schemas**

.. currentmodule:: propertyestimator.workflow.schemas
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ProtocolSchema
    ProtocolGroupSchema
    ProtocolReplicator
    WorkflowSchema

**Attributes**

.. currentmodule:: propertyestimator.workflow.attributes
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseMergeBehaviour
    MergeBehaviour
    InequalityMergeBehaviour
    InputAttribute
    OutputAttribute

*Placeholder Values*

.. currentmodule:: propertyestimator.workflow.utils
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ReplicatorValue
    ProtocolPath

Built-in Workflow Protocols
---------------------------

**Analysis**

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

    BaseBuildSystem
    BuildSmirnoffSystem
    BuildLigParGenSystem
    BuildTLeapSystem

**Gradients**

.. currentmodule:: propertyestimator.protocols.gradients
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseGradientPotentials
    CentralDifferenceGradient

**Groups**

.. currentmodule:: propertyestimator.protocols.groups
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ConditionalGroup

**Miscellaneous**

.. currentmodule:: propertyestimator.protocols.miscellaneous
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

.. currentmodule:: propertyestimator.protocols.openmm
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    OpenMMEnergyMinimisation
    OpenMMSimulation
    OpenMMReducedPotentials
    OpenMMGradientPotentials

**Reweighting**

.. currentmodule:: propertyestimator.protocols.reweighting
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ConcatenateTrajectories
    ConcatenateStatistics
    BaseReducedPotentials
    BaseMBARProtocol
    ReweightStatistics

**Simulation**

.. currentmodule:: propertyestimator.protocols.simulation
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseEnergyMinimisation
    BaseSimulation

**Storage**

.. currentmodule:: propertyestimator.protocols.storage
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    UnpackStoredSimulationData

**YANK Free Energies**

.. currentmodule:: propertyestimator.protocols.yank
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseYankProtocol
    LigandReceptorYankProtocol
    SolvationYankProtocol

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

    Attribute
    AttributeClass
    UNDEFINED
    PlaceholderValue

Plug-in Utilities
-----------------

**Plug-ins**

.. currentmodule:: propertyestimator.plugins
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    register_default_plugins
    register_external_plugins