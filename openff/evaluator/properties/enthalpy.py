"""
A collection of enthalpy physical property definitions.
"""

from openff.units import unit

from openff.evaluator.attributes import UNDEFINED, PlaceholderValue
from openff.evaluator.datasets import PhysicalProperty, PropertyPhase
from openff.evaluator.datasets.thermoml import thermoml_property
from openff.evaluator.layers import register_calculation_schema
from openff.evaluator.layers.equilibration import EquilibrationLayer
from openff.evaluator.layers.preequilibrated_simulation import (
    PreequilibratedSimulationLayer,
)
from openff.evaluator.layers.reweighting import ReweightingLayer, ReweightingSchema
from openff.evaluator.layers.simulation import SimulationLayer, SimulationSchema
from openff.evaluator.properties.properties import EstimableExcessProperty
from openff.evaluator.protocols import analysis, groups, miscellaneous
from openff.evaluator.protocols.utils import (
    generate_reweighting_protocols,
    generate_simulation_protocols,
)
from openff.evaluator.storage.query import SimulationDataQuery
from openff.evaluator.thermodynamics import Ensemble
from openff.evaluator.utils.observables import ObservableType
from openff.evaluator.workflow.schemas import WorkflowSchema
from openff.evaluator.workflow.utils import ProtocolPath


@thermoml_property(
    "Excess molar enthalpy (molar enthalpy of mixing), kJ/mol",
    supported_phases=PropertyPhase.Liquid,
)
class EnthalpyOfMixing(EstimableExcessProperty):
    """A class representation of an enthalpy of mixing property"""

    @classmethod
    def default_unit(cls):
        return unit.kilojoule / unit.mole

    @classmethod
    def _observable_type(cls) -> ObservableType:
        return ObservableType.Enthalpy

    @classmethod
    def default_reweighting_schema(
        cls,
        absolute_tolerance: unit.Quantity = UNDEFINED,
        relative_tolerance: float = UNDEFINED,
        n_effective_samples: int = 50,
    ) -> ReweightingSchema:
        calculation_schema = super(EnthalpyOfMixing, cls)._default_reweighting_schema(
            ObservableType.ReducedPotential,
            absolute_tolerance,
            relative_tolerance,
            n_effective_samples,
        )

        # Divide the excess reduced potential by beta to get an approximation
        # of the excess enthalpy.
        excess_enthalpy_of_mixing = miscellaneous.MultiplyValue(
            "excess_enthalpy_of_mixing"
        )
        excess_enthalpy_of_mixing.value = (
            calculation_schema.workflow_schema.final_value_source
        )
        excess_enthalpy_of_mixing.multiplier = ProtocolPath(
            "thermodynamic_state.inverse_beta", "global"
        )

        # Update the workflow schema.
        calculation_schema.workflow_schema.protocol_schemas.append(
            excess_enthalpy_of_mixing.schema
        )
        calculation_schema.workflow_schema.final_value_source = ProtocolPath(
            "result", excess_enthalpy_of_mixing.id
        )

        return calculation_schema


@thermoml_property(
    "Molar enthalpy of vaporization or sublimation, kJ/mol",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class EnthalpyOfVaporization(PhysicalProperty):
    """A class representation of an enthalpy of vaporization property"""

    @classmethod
    def default_unit(cls):
        return unit.kilojoule / unit.mole

    @staticmethod
    def _default_reweighting_storage_query():
        """Returns the default storage queries to use when
        retrieving cached simulation data to reweight.

        This will include one query for the liquid data (with the
        key `"liquid_data"`) and one for the gas data (with the key
        `"gas_data"`).

        Returns
        -------
        dict of str and SimulationDataQuery
            The dictionary of queries.
        """

        liquid_data_query = SimulationDataQuery()
        liquid_data_query.substance = PlaceholderValue()
        liquid_data_query.property_phase = PropertyPhase.Liquid

        gas_data_query = SimulationDataQuery()
        gas_data_query.substance = PlaceholderValue()
        gas_data_query.property_phase = PropertyPhase.Gas
        gas_data_query.number_of_molecules = 1

        return {
            "liquid_data": liquid_data_query,
            "gas_data": gas_data_query,
        }

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

        # Define a custom conditional group which will ensure both the liquid and
        # gas enthalpies are estimated to within the specified uncertainty tolerance.
        converge_uncertainty = groups.ConditionalGroup("conditional_group")
        converge_uncertainty.max_iterations = 100

        # Define the protocols to perform the simulation in the liquid phase.
        average_liquid_energy = analysis.AverageObservable("average_liquid_potential")
        average_liquid_energy.divisor = n_molecules
        (
            liquid_protocols,
            liquid_value_source,
            liquid_output_to_store,
        ) = generate_simulation_protocols(
            average_liquid_energy,
            use_target_uncertainty,
            "_liquid",
            converge_uncertainty,
            n_molecules=n_molecules,
        )
        liquid_output_to_store.property_phase = PropertyPhase.Liquid

        liquid_protocols.analysis_protocol.observable = ProtocolPath(
            f"observables[{ObservableType.PotentialEnergy.value}]",
            liquid_protocols.production_simulation.id,
        )

        # Define the protocols to perform the simulation in the gas phase.
        average_gas_energy = analysis.AverageObservable("average_gas_potential")
        (
            gas_protocols,
            gas_value_source,
            gas_output_to_store,
        ) = generate_simulation_protocols(
            average_gas_energy,
            use_target_uncertainty,
            "_gas",
            converge_uncertainty,
            n_molecules=1,
        )
        gas_output_to_store.property_phase = PropertyPhase.Gas

        gas_protocols.analysis_protocol.observable = ProtocolPath(
            f"observables[{ObservableType.PotentialEnergy.value}]",
            gas_protocols.production_simulation.id,
        )

        # Specify that for the gas phase only a single molecule in vacuum should be
        # created.
        gas_protocols.build_coordinates.max_molecules = 1
        gas_protocols.build_coordinates.mass_density = (
            0.01 * unit.gram / unit.milliliter
        )

        # Run the gas phase simulations in the NVT ensemble without PBC
        gas_protocols.energy_minimisation.enable_pbc = False
        gas_protocols.equilibration_simulation.ensemble = Ensemble.NVT
        gas_protocols.equilibration_simulation.enable_pbc = False
        gas_protocols.production_simulation.ensemble = Ensemble.NVT
        gas_protocols.production_simulation.enable_pbc = False
        gas_protocols.production_simulation.steps_per_iteration = 15000000
        gas_protocols.production_simulation.output_frequency = 5000
        gas_protocols.production_simulation.checkpoint_frequency = 100

        # Due to a bizarre issue where the OMM Reference platform is
        # the fastest at computing properties of a single molecule
        # in vacuum, we enforce those inputs which will force the
        # gas calculations to run on the Reference platform.
        gas_protocols.equilibration_simulation.high_precision = True
        gas_protocols.equilibration_simulation.allow_gpu_platforms = False
        gas_protocols.production_simulation.high_precision = True
        gas_protocols.production_simulation.allow_gpu_platforms = False

        # Combine the values to estimate the final energy of vaporization
        energy_of_vaporization = miscellaneous.SubtractValues("energy_of_vaporization")
        energy_of_vaporization.value_b = ProtocolPath("value", average_gas_energy.id)
        energy_of_vaporization.value_a = ProtocolPath("value", average_liquid_energy.id)

        ideal_volume = miscellaneous.MultiplyValue("ideal_volume")
        ideal_volume.value = 1.0 * unit.molar_gas_constant
        ideal_volume.multiplier = ProtocolPath(
            "thermodynamic_state.temperature", "global"
        )

        enthalpy_of_vaporization = miscellaneous.AddValues("enthalpy_of_vaporization")
        enthalpy_of_vaporization.values = [
            ProtocolPath("result", energy_of_vaporization.id),
            ProtocolPath("result", ideal_volume.id),
        ]

        # Add the extra protocols and conditions to the custom conditional group.
        converge_uncertainty.add_protocols(
            energy_of_vaporization, ideal_volume, enthalpy_of_vaporization
        )

        if use_target_uncertainty:
            condition = groups.ConditionalGroup.Condition()
            condition.type = groups.ConditionalGroup.Condition.Type.LessThan

            condition.left_hand_value = ProtocolPath(
                "result.error",
                converge_uncertainty.id,
                enthalpy_of_vaporization.id,
            )
            condition.right_hand_value = ProtocolPath("target_uncertainty", "global")

            gas_protocols.production_simulation.total_number_of_iterations = (
                ProtocolPath("current_iteration", converge_uncertainty.id)
            )
            liquid_protocols.production_simulation.total_number_of_iterations = (
                ProtocolPath("current_iteration", converge_uncertainty.id)
            )

            converge_uncertainty.add_condition(condition)

        # Build the workflow schema.
        schema = WorkflowSchema()

        schema.protocol_schemas = [
            liquid_protocols.build_coordinates.schema,
            liquid_protocols.assign_parameters.schema,
            liquid_protocols.energy_minimisation.schema,
            liquid_protocols.equilibration_simulation.schema,
            liquid_protocols.decorrelate_trajectory.schema,
            liquid_protocols.decorrelate_observables.schema,
            gas_protocols.build_coordinates.schema,
            gas_protocols.assign_parameters.schema,
            gas_protocols.energy_minimisation.schema,
            gas_protocols.equilibration_simulation.schema,
            gas_protocols.decorrelate_trajectory.schema,
            gas_protocols.decorrelate_observables.schema,
            converge_uncertainty.schema,
        ]

        schema.outputs_to_store = {
            "liquid_data": liquid_output_to_store,
            "gas_data": gas_output_to_store,
        }

        schema.final_value_source = ProtocolPath(
            "result", converge_uncertainty.id, enthalpy_of_vaporization.id
        )

        calculation_schema.workflow_schema = schema
        return calculation_schema

    @classmethod
    def default_reweighting_schema(
        cls,
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

        # Set up the storage queries
        calculation_schema.storage_queries = cls._default_reweighting_storage_query()

        # Set up a protocol to extract the liquid phase energy from the existing data.
        liquid_protocols, liquid_replicator = generate_reweighting_protocols(
            ObservableType.PotentialEnergy,
            id_suffix="_liquid",
            replicator_id="liquid_data_replicator",
        )
        liquid_replicator.template_values = ProtocolPath("liquid_data", "global")
        liquid_protocols.reweight_observable.required_effective_samples = (
            n_effective_samples
        )

        # Dive the potential by the number of liquid phase molecules from the first
        # piece of cached data.
        divide_by_liquid_molecules = miscellaneous.DivideValue(
            "divide_by_liquid_molecules"
        )
        divide_by_liquid_molecules.value = ProtocolPath(
            "value", liquid_protocols.reweight_observable.id
        )
        divide_by_liquid_molecules.divisor = ProtocolPath(
            "total_number_of_molecules",
            liquid_protocols.unpack_stored_data.id.replace(
                liquid_replicator.placeholder_id, "0"
            ),
        )

        # Set up a protocol to extract the gas phase energy from the existing data.
        gas_protocols, gas_replicator = generate_reweighting_protocols(
            ObservableType.PotentialEnergy,
            id_suffix="_gas",
            replicator_id="gas_data_replicator",
        )
        gas_replicator.template_values = ProtocolPath("gas_data", "global")
        gas_protocols.reweight_observable.required_effective_samples = (
            n_effective_samples
        )

        # Turn of PBC for the gas phase.
        gas_protocols.evaluate_reference_potential.enable_pbc = False
        gas_protocols.evaluate_target_potential.enable_pbc = False

        # Combine the values to estimate the final enthalpy of vaporization
        energy_of_vaporization = miscellaneous.SubtractValues("energy_of_vaporization")
        energy_of_vaporization.value_b = ProtocolPath(
            "value", gas_protocols.reweight_observable.id
        )
        energy_of_vaporization.value_a = ProtocolPath(
            "result", divide_by_liquid_molecules.id
        )

        ideal_volume = miscellaneous.MultiplyValue("ideal_volume")
        ideal_volume.value = 1.0 * unit.molar_gas_constant
        ideal_volume.multiplier = ProtocolPath(
            "thermodynamic_state.temperature", "global"
        )

        enthalpy_of_vaporization = miscellaneous.AddValues("enthalpy_of_vaporization")
        enthalpy_of_vaporization.values = [
            ProtocolPath("result", energy_of_vaporization.id),
            ProtocolPath("result", ideal_volume.id),
        ]

        # Build the workflow schema.
        schema = WorkflowSchema()
        schema.protocol_schemas = [
            *(x.schema for x in liquid_protocols if x is not None),
            *(x.schema for x in gas_protocols if x is not None),
            divide_by_liquid_molecules.schema,
            energy_of_vaporization.schema,
            ideal_volume.schema,
            enthalpy_of_vaporization.schema,
        ]
        schema.protocol_replicators = [liquid_replicator, gas_replicator]
        schema.final_value_source = ProtocolPath("result", enthalpy_of_vaporization.id)

        calculation_schema.workflow_schema = schema
        return calculation_schema


# Register the properties via the plugin system.
register_calculation_schema(
    EnthalpyOfMixing, SimulationLayer, EnthalpyOfMixing.default_simulation_schema
)
register_calculation_schema(
    EnthalpyOfMixing, ReweightingLayer, EnthalpyOfMixing.default_reweighting_schema
)
register_calculation_schema(
    EnthalpyOfMixing, EquilibrationLayer, EnthalpyOfMixing.default_equilibration_schema
)
register_calculation_schema(
    EnthalpyOfMixing,
    PreequilibratedSimulationLayer,
    EnthalpyOfMixing.default_preequilibrated_simulation_schema,
)
register_calculation_schema(
    EnthalpyOfVaporization,
    SimulationLayer,
    EnthalpyOfVaporization.default_simulation_schema,
)
register_calculation_schema(
    EnthalpyOfVaporization,
    ReweightingLayer,
    EnthalpyOfVaporization.default_reweighting_schema,
)
