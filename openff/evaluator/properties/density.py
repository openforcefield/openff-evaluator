"""
A collection of density physical property definitions.
"""

import copy

from openff.units import unit

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.datasets import PhysicalProperty, PropertyPhase
from openff.evaluator.datasets.thermoml import thermoml_property
from openff.evaluator.layers import register_calculation_schema
from openff.evaluator.layers.equilibration import (
    EquilibrationProperty,
    EquilibrationSchema,
)
from openff.evaluator.layers.preequilibrated_simulation import (
    PreequilibratedSimulationSchema,
)
from openff.evaluator.layers.reweighting import ReweightingLayer, ReweightingSchema
from openff.evaluator.layers.simulation import SimulationLayer, SimulationSchema
from openff.evaluator.properties.properties import EstimableExcessProperty
from openff.evaluator.protocols import analysis
from openff.evaluator.protocols.utils import (
    generate_equilibration_protocols,
    generate_preequilibrated_simulation_protocols,
    generate_reweighting_protocols,
    generate_simulation_protocols,
)
from openff.evaluator.utils.observables import ObservableType
from openff.evaluator.workflow.attributes import ConditionAggregationBehavior
from openff.evaluator.workflow.schemas import WorkflowSchema
from openff.evaluator.workflow.utils import ProtocolPath


@thermoml_property("Mass density, kg/m3", supported_phases=PropertyPhase.Liquid)
class Density(PhysicalProperty):
    """A class representation of a density property"""

    @classmethod
    def default_unit(cls):
        return unit.gram / unit.millilitre

    @classmethod
    def default_equilibration_schema(
        cls,
        n_molecules: int = 1000,
        error_tolerances: list[EquilibrationProperty] = [],
        condition_aggregation_behavior: ConditionAggregationBehavior = ConditionAggregationBehavior.All,
        error_on_failure: bool = True,
        max_iterations: int = 100,
    ) -> EquilibrationSchema:
        """Returns the default equilibration schema to use when equilibrating
        liquid boxes.

        Parameters
        ----------
        n_molecules: int
            The number of molecules to use in the simulation.
        error_tolerances: list[EquilibrationProperty]
            The error tolerances to estimate the property to within.
        condition_aggregation_behavior: ConditionAggregationBehavior
            How to aggregate errors -- any vs all.
        error_on_failure: bool
            Whether to raise an error if the convergence conditions are not met.
        max_iterations: int
            The maximum number of iterations to run the equilibration for.
            Each iteration is 200 ps long by default.

        Returns
        -------
        EquilibrationSchema
            The schema to follow when equilibrating boxes for this property.
        """

        calculation_schema = EquilibrationSchema()
        calculation_schema.error_tolerances = copy.deepcopy(error_tolerances)
        calculation_schema.error_aggregration = copy.deepcopy(
            condition_aggregation_behavior
        )
        calculation_schema.error_on_failure = error_on_failure
        calculation_schema.max_iterations = max_iterations
        calculation_schema.number_of_molecules = n_molecules

        # Define the protocols which will run the simulation itself.
        protocols, value_source, output_to_store = generate_equilibration_protocols(
            n_molecules=n_molecules,
            error_tolerances=calculation_schema.error_tolerances,
            condition_aggregation_behavior=calculation_schema.error_aggregration,
            error_on_failure=calculation_schema.error_on_failure,
            max_iterations=calculation_schema.max_iterations,
        )

        # Build the workflow schema.
        schema = WorkflowSchema()

        schema.protocol_schemas = [
            protocols.check_existing_data.schema,
            protocols.build_coordinates.schema,
            protocols.assign_parameters.schema,
            protocols.energy_minimisation.schema,
            # protocols.equilibration_simulation.schema,
            protocols.converge_uncertainty.schema,
        ]

        schema.outputs_to_store = {"full_system": output_to_store}
        schema.final_value_source = value_source

        calculation_schema.workflow_schema = schema
        return calculation_schema

    @classmethod
    def default_preequilibrated_simulation_schema(
        cls,
        absolute_tolerance=UNDEFINED,
        relative_tolerance=UNDEFINED,
        n_molecules=1000,
        equilibration_error_tolerances: list[EquilibrationProperty] = [],
        equilibration_error_aggregration: ConditionAggregationBehavior = ConditionAggregationBehavior.All,
        equilibration_error_on_failure: bool = False,
        equilibration_max_iterations: int = 100,
        n_uncorrelated_samples: int = 200,
        max_iterations: int = 100,
        error_on_failure: bool = False,
    ) -> PreequilibratedSimulationSchema:

        assert absolute_tolerance == UNDEFINED or relative_tolerance == UNDEFINED

        use_target_uncertainty = (
            absolute_tolerance != UNDEFINED or relative_tolerance != UNDEFINED
        )

        calculation_schema = PreequilibratedSimulationSchema()
        calculation_schema.absolute_tolerance = absolute_tolerance
        calculation_schema.relative_tolerance = relative_tolerance
        calculation_schema.number_of_molecules = n_molecules
        calculation_schema.n_uncorrelated_samples = n_uncorrelated_samples
        calculation_schema.equilibration_error_aggregration = (
            equilibration_error_aggregration
        )
        calculation_schema.equilibration_error_on_failure = (
            equilibration_error_on_failure
        )
        calculation_schema.equilibration_error_tolerances = (
            equilibration_error_tolerances
        )
        calculation_schema.equilibration_max_iterations = equilibration_max_iterations
        calculation_schema.max_iterations = max_iterations
        calculation_schema.error_on_failure = error_on_failure

        protocols, value_source, output_to_store = (
            generate_preequilibrated_simulation_protocols(
                analysis.AverageObservable("average_density"),
                use_target_uncertainty,
                n_molecules=n_molecules,
                equilibration_error_tolerances=equilibration_error_tolerances,
                equilibration_error_aggregration=equilibration_error_aggregration,
                equilibration_error_on_failure=equilibration_error_on_failure,
                equilibration_max_iterations=equilibration_max_iterations,
                n_uncorrelated_samples=n_uncorrelated_samples,
                max_iterations=max_iterations,
                error_on_failure=error_on_failure,
            )
        )
        protocols.analysis_protocol.observable = ProtocolPath(
            f"observables[{ObservableType.Density.value}]",
            protocols.production_simulation.id,
        )

        schema = WorkflowSchema()
        schema.protocol_schemas = [
            protocols.unpack_stored_data.schema,
            protocols.assign_parameters.schema,
            protocols.energy_minimisation.schema,
            protocols.converge_equilibration.schema,
            protocols.converge_uncertainty.schema,
            protocols.decorrelate_trajectory.schema,
            protocols.decorrelate_observables.schema,
        ]
        schema.outputs_to_store = {"full_system": output_to_store}
        schema.final_value_source = value_source
        calculation_schema.workflow_schema = schema
        return calculation_schema

    @staticmethod
    def default_simulation_schema(
        absolute_tolerance=UNDEFINED, relative_tolerance=UNDEFINED, n_molecules=1000
    ) -> SimulationSchema:
        """Returns the default calculation schema to use when estimating
        this class of property from direct simulations.

        Parameters
        ----------
        absolute_tolerance: openff.evaluator.unit.Quantity, optional
            The absolute tolerance to estimate the property to within.
        relative_tolerance: float
            The tolerance (as a fraction of the properties reported
            uncertainty) to estimate the property to within.
        n_molecules: int
            The number of molecules to use in the simulation.

        Returns
        -------
        SimulationSchema
            The schema to follow when estimating this property.
        """
        assert absolute_tolerance == UNDEFINED or relative_tolerance == UNDEFINED

        calculation_schema = SimulationSchema()
        calculation_schema.absolute_tolerance = absolute_tolerance
        calculation_schema.relative_tolerance = relative_tolerance

        use_target_uncertainty = (
            absolute_tolerance != UNDEFINED or relative_tolerance != UNDEFINED
        )

        # Define the protocols which will run the simulation itself.
        protocols, value_source, output_to_store = generate_simulation_protocols(
            analysis.AverageObservable("average_density"),
            use_target_uncertainty,
            n_molecules=n_molecules,
        )
        # Specify that the average density should be estimated.
        protocols.analysis_protocol.observable = ProtocolPath(
            f"observables[{ObservableType.Density.value}]",
            protocols.production_simulation.id,
        )

        # Build the workflow schema.
        schema = WorkflowSchema()

        schema.protocol_schemas = [
            protocols.build_coordinates.schema,
            protocols.assign_parameters.schema,
            protocols.energy_minimisation.schema,
            protocols.equilibration_simulation.schema,
            protocols.converge_uncertainty.schema,
            protocols.decorrelate_trajectory.schema,
            protocols.decorrelate_observables.schema,
        ]

        schema.outputs_to_store = {"full_system": output_to_store}
        schema.final_value_source = value_source

        calculation_schema.workflow_schema = schema
        return calculation_schema

    @staticmethod
    def default_reweighting_schema(
        absolute_tolerance=UNDEFINED,
        relative_tolerance=UNDEFINED,
        n_effective_samples=50,
    ) -> ReweightingSchema:
        """Returns the default calculation schema to use when estimating
        this property by reweighting existing data.

        Parameters
        ----------
        absolute_tolerance: openff.evaluator.unit.Quantity, optional
            The absolute tolerance to estimate the property to within.
        relative_tolerance: float
            The tolerance (as a fraction of the properties reported
            uncertainty) to estimate the property to within.
        n_effective_samples: int
            The minimum number of effective samples to require when
            reweighting the cached simulation data.

        Returns
        -------
        ReweightingSchema
            The schema to follow when estimating this property.
        """
        assert absolute_tolerance == UNDEFINED or relative_tolerance == UNDEFINED

        calculation_schema = ReweightingSchema()
        calculation_schema.absolute_tolerance = absolute_tolerance
        calculation_schema.relative_tolerance = relative_tolerance

        protocols, data_replicator = generate_reweighting_protocols(
            ObservableType.Density
        )
        protocols.reweight_observable.required_effective_samples = n_effective_samples

        schema = WorkflowSchema()
        schema.protocol_schemas = [x.schema for x in protocols]
        schema.protocol_replicators = [data_replicator]

        schema.final_value_source = ProtocolPath(
            "value", protocols.reweight_observable.id
        )

        calculation_schema.workflow_schema = schema
        return calculation_schema


@thermoml_property("Excess molar volume, m3/mol", supported_phases=PropertyPhase.Liquid)
class ExcessMolarVolume(EstimableExcessProperty):
    """A class representation of an excess molar volume property"""

    @classmethod
    def default_unit(cls):
        return unit.centimeter**3 / unit.mole

    @classmethod
    def _observable_type(cls) -> ObservableType:
        return ObservableType.Volume


# Register the properties via the plugin system.
register_calculation_schema(Density, SimulationLayer, Density.default_simulation_schema)
register_calculation_schema(
    Density, ReweightingLayer, Density.default_reweighting_schema
)
register_calculation_schema(
    ExcessMolarVolume,
    SimulationLayer,
    ExcessMolarVolume.default_simulation_schema,
)
register_calculation_schema(
    ExcessMolarVolume,
    ReweightingLayer,
    ExcessMolarVolume.default_reweighting_schema,
)
