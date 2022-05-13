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

**Taproom**

.. currentmodule:: openff.evaluator.datasets.taproom
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    TaproomDataSet
    TaproomSource

**Data Set Curation**

.. currentmodule:: openff.evaluator.datasets.curation.components
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    CurationComponent
    CurationComponentSchema

.. currentmodule:: openff.evaluator.datasets.curation.workflow
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    CurationWorkflow
    CurationWorkflowSchema

*Filtering*

.. currentmodule:: openff.evaluator.datasets.curation.components.filtering
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    FilterDuplicatesSchema
    FilterDuplicates
    FilterByTemperatureSchema
    FilterByTemperature
    FilterByPressureSchema
    FilterByPressure
    FilterByMoleFractionSchema
    FilterByMoleFraction
    FilterByRacemicSchema
    FilterByRacemic
    FilterByElementsSchema
    FilterByElements
    FilterByPropertyTypesSchema
    FilterByPropertyTypes
    FilterByStereochemistrySchema
    FilterByStereochemistry
    FilterByChargedSchema
    FilterByCharged
    FilterByIonicLiquidSchema
    FilterByIonicLiquid
    FilterBySmilesSchema
    FilterBySmiles
    FilterBySmirksSchema
    FilterBySmirks
    FilterByNComponentsSchema
    FilterByNComponents
    FilterBySubstancesSchema
    FilterBySubstances
    FilterByEnvironmentsSchema
    FilterByEnvironments

*FreeSolv*

.. currentmodule:: openff.evaluator.datasets.curation.components.freesolv
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ImportFreeSolvSchema
    ImportFreeSolv

*ThermoML*

.. currentmodule:: openff.evaluator.datasets.curation.components.thermoml
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ImportThermoMLDataSchema
    ImportThermoMLData

*Data Point Selection*

.. currentmodule:: openff.evaluator.datasets.curation.components.selection
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    SelectSubstancesSchema
    SelectSubstances
    SelectDataPointsSchema
    SelectDataPoints
    State
    TargetState
    FingerPrintType

*Data Conversion*

.. currentmodule:: openff.evaluator.datasets.curation.components.conversion
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ConvertExcessDensityDataSchema
    ConvertExcessDensityData

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
    BaseSimulationData
    StoredSimulationData
    StoredFreeEnergyData

**Data Queries**

.. currentmodule:: openff.evaluator.storage.query
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    BaseDataQuery
    SubstanceQuery
    ForceFieldQuery
    BaseSimulationDataQuery
    SimulationDataQuery
    FreeEnergyDataQuery

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

    BaseMergeBehavior
    MergeBehavior
    InequalityMergeBehavior
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

    BaseAverageObservable
    AverageObservable
    AverageDielectricConstant
    AverageFreeEnergies
    ComputeDipoleMoments
    BaseDecorrelateProtocol
    DecorrelateTrajectory
    DecorrelateObservables

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

    ZeroGradients

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
    DummyProtocol

**OpenMM**

.. currentmodule:: openff.evaluator.protocols.openmm
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    OpenMMEnergyMinimisation
    OpenMMSimulation
    OpenMMEvaluateEnergies

**Paprika**

.. currentmodule:: openff.evaluator.protocols.paprika.coordinates
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    PreparePullCoordinates
    PrepareReleaseCoordinates
    AddDummyAtoms

.. currentmodule:: openff.evaluator.protocols.paprika.restraints
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    GenerateAttachRestraints
    GeneratePullRestraints
    GenerateReleaseRestraints
    ApplyRestraints

.. currentmodule:: openff.evaluator.protocols.paprika.analysis
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    AnalyzeAPRPhase
    ComputeSymmetryCorrection
    ComputeReferenceWork

**Reweighting**

.. currentmodule:: openff.evaluator.protocols.reweighting
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    ConcatenateTrajectories
    ConcatenateObservables
    BaseEvaluateEnergies
    BaseMBARProtocol
    ReweightObservable
    ReweightDielectricConstant

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

    SimulationProtocols
    ReweightingProtocols
    generate_base_reweighting_protocols
    generate_reweighting_protocols
    generate_simulation_protocols

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

Observable Utilities
--------------------

.. currentmodule:: openff.evaluator.utils.observables
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    Observable
    ObservableArray
    ObservableType
    ObservableFrame
    bootstrap

Plug-in Utilities
-----------------

**Plug-ins**

.. currentmodule:: openff.evaluator.plugins
.. autosummary::
    :nosignatures:
    :toctree: api/generated/

    register_default_plugins
    register_external_plugins