"""
A collection of dielectric physical property definitions.
"""
from openff.units import unit

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.datasets import PhysicalProperty, PropertyPhase
from openff.evaluator.datasets.thermoml import thermoml_property
from openff.evaluator.layers import register_calculation_schema
from openff.evaluator.layers.reweighting import ReweightingLayer, ReweightingSchema
from openff.evaluator.layers.simulation import SimulationLayer, SimulationSchema
from openff.evaluator.protocols.analysis import (
    AverageDielectricConstant,
    ComputeDipoleMoments,
    DecorrelateObservables,
)
from openff.evaluator.protocols.reweighting import (
    ConcatenateObservables,
    ReweightDielectricConstant,
)
from openff.evaluator.protocols.utils import (
    generate_base_reweighting_protocols,
    generate_simulation_protocols,
)
from openff.evaluator.utils.observables import ObservableType
from openff.evaluator.workflow import WorkflowSchema
from openff.evaluator.workflow.utils import ProtocolPath


@thermoml_property(
    "Relative permittivity at zero frequency",
    supported_phases=PropertyPhase.Liquid,
)
class DielectricConstant(PhysicalProperty):
    """A class representation of a dielectric property"""

    @classmethod
    def default_unit(cls):
        return unit.dimensionless

    @staticmethod
    def default_simulation_schema(
        absolute_tolerance=UNDEFINED, relative_tolerance=UNDEFINED, n_molecules=1000
    ):
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
            AverageDielectricConstant("average_dielectric"),
            use_target_uncertainty,
            n_molecules=n_molecules,
        )

        # Add a protocol to compute the dipole moments and pass these to
        # the analysis protocol.
        compute_dipoles = ComputeDipoleMoments("compute_dipoles")
        compute_dipoles.parameterized_system = ProtocolPath(
            "parameterized_system", protocols.assign_parameters.id
        )
        compute_dipoles.trajectory_path = ProtocolPath(
            "trajectory_file_path", protocols.production_simulation.id
        )
        compute_dipoles.gradient_parameters = ProtocolPath(
            "parameter_gradient_keys", "global"
        )
        protocols.converge_uncertainty.add_protocols(compute_dipoles)

        protocols.analysis_protocol.volumes = ProtocolPath(
            f"observables[{ObservableType.Volume.value}]",
            protocols.production_simulation.id,
        )
        protocols.analysis_protocol.dipole_moments = ProtocolPath(
            "dipole_moments",
            compute_dipoles.id,
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
    ):
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

        protocols, data_replicator = generate_base_reweighting_protocols(
            statistical_inefficiency=AverageDielectricConstant(
                "average_dielectric_$(data_replicator)"
            ),
            reweight_observable=ReweightDielectricConstant("reweight_dielectric"),
        )
        protocols.zero_gradients.input_observables = ProtocolPath(
            "output_observables[Volume]",
            protocols.join_observables.id,
        )
        protocols.statistical_inefficiency.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )
        protocols.reweight_observable.required_effective_samples = n_effective_samples

        # We don't need to perform bootstrapping as this protocol is only used to
        # calculate the statistical inefficiency and equilibration time. The
        # re-weighting protocol will instead compute the bootstrapped uncertainties.
        protocols.statistical_inefficiency.bootstrap_iterations = 1

        # Set up a protocol to re-evaluate the dipole moments at the target state
        # and concatenate the into a single array.
        compute_dipoles = ComputeDipoleMoments("compute_dipoles_$(data_replicator)")
        compute_dipoles.parameterized_system = ProtocolPath(
            "parameterized_system", protocols.build_target_system.id
        )
        compute_dipoles.trajectory_path = ProtocolPath(
            "trajectory_file_path", protocols.unpack_stored_data.id
        )
        compute_dipoles.gradient_parameters = ProtocolPath(
            "parameter_gradient_keys", "global"
        )
        join_dipoles = ConcatenateObservables("join_dipoles")
        join_dipoles.input_observables = ProtocolPath(
            "dipole_moments",
            compute_dipoles.id,
        )

        # Point the dielectric protocols to the volumes and dipole moments.
        protocols.statistical_inefficiency.volumes = ProtocolPath(
            "observables[Volume]", protocols.unpack_stored_data.id
        )
        protocols.statistical_inefficiency.dipole_moments = ProtocolPath(
            "dipole_moments", compute_dipoles.id
        )

        # Make sure to decorrelate the dipole moments.
        decorrelate_dipoles = DecorrelateObservables("decorrelate_dipoles")
        decorrelate_dipoles.time_series_statistics = ProtocolPath(
            "time_series_statistics", protocols.statistical_inefficiency.id
        )
        decorrelate_dipoles.input_observables = ProtocolPath(
            "output_observables", join_dipoles.id
        )

        protocols.reweight_observable.dipole_moments = ProtocolPath(
            "output_observables", decorrelate_dipoles.id
        )
        protocols.reweight_observable.volumes = ProtocolPath(
            "output_observables", protocols.decorrelate_observable.id
        )
        protocols.reweight_observable.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )

        schema = WorkflowSchema()
        schema.protocol_schemas = [
            *(x.schema for x in protocols),
            compute_dipoles.schema,
            join_dipoles.schema,
            decorrelate_dipoles.schema,
        ]
        schema.protocol_replicators = [data_replicator]
        schema.final_value_source = ProtocolPath(
            "value", protocols.reweight_observable.id
        )

        calculation_schema.workflow_schema = schema
        return calculation_schema


# Register the properties via the plugin system.
register_calculation_schema(
    DielectricConstant, SimulationLayer, DielectricConstant.default_simulation_schema
)
register_calculation_schema(
    DielectricConstant, ReweightingLayer, DielectricConstant.default_reweighting_schema
)
