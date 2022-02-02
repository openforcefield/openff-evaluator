"""
A collection of density physical property definitions.
"""

from openff.units import unit

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.datasets import PhysicalProperty, PropertyPhase
from openff.evaluator.datasets.thermoml import thermoml_property
from openff.evaluator.layers import register_calculation_schema
from openff.evaluator.layers.reweighting import ReweightingLayer, ReweightingSchema
from openff.evaluator.layers.simulation import SimulationLayer, SimulationSchema
from openff.evaluator.properties.properties import EstimableExcessProperty
from openff.evaluator.protocols import analysis
from openff.evaluator.protocols.utils import (
    generate_reweighting_protocols,
    generate_simulation_protocols,
)
from openff.evaluator.utils.observables import ObservableType
from openff.evaluator.workflow.schemas import WorkflowSchema
from openff.evaluator.workflow.utils import ProtocolPath


@thermoml_property("Mass density, kg/m3", supported_phases=PropertyPhase.Liquid)
class Density(PhysicalProperty):
    """A class representation of a density property"""

    @classmethod
    def default_unit(cls):
        return unit.gram / unit.millilitre

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
        return unit.centimeter ** 3 / unit.mole

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
