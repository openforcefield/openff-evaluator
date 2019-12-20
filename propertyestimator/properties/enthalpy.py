"""
A collection of enthalpy physical property definitions.
"""
import copy
from collections import namedtuple

from propertyestimator import unit
from propertyestimator.attributes import UNDEFINED, PlaceholderValue
from propertyestimator.datasets import PhysicalProperty, PropertyPhase
from propertyestimator.datasets.thermoml.plugins import thermoml_property
from propertyestimator.layers import register_calculation_schema
from propertyestimator.layers.reweighting import ReweightingLayer, ReweightingSchema
from propertyestimator.layers.simulation import SimulationLayer, SimulationSchema
from propertyestimator.protocols import analysis, groups, miscellaneous, reweighting
from propertyestimator.protocols.utils import (
    generate_base_reweighting_protocols,
    generate_base_simulation_protocols,
    generate_gradient_protocol_group,
)
from propertyestimator.storage.query import SimulationDataQuery, SubstanceQuery
from propertyestimator.thermodynamics import Ensemble
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import ObservableType
from propertyestimator.workflow import WorkflowOptions
from propertyestimator.workflow.schemas import ProtocolReplicator, WorkflowSchema
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue


@thermoml_property(
    "Excess molar enthalpy (molar enthalpy of mixing), kJ/mol",
    supported_phases=PropertyPhase.Liquid,
)
class EnthalpyOfMixing(PhysicalProperty):
    """A class representation of an enthalpy of mixing property"""

    EnthalpyWorkflow = namedtuple(
        "EnthalpySchema",
        "build_coordinates "
        "assign_topology "
        "energy_minimisation "
        "npt_equilibration "
        "converge_uncertainty "
        "subsample_trajectory "
        "subsample_statistics ",
    )

    @staticmethod
    def _get_simulation_protocols(
        id_suffix,
        gradient_replicator_id,
        replicator_id=None,
        weight_by_mole_fraction=False,
        component_substance_reference=None,
        full_substance_reference=None,
        options=None,
        n_molecules=1000,
    ):

        """Returns the set of protocols which when combined in a workflow
        will yield the enthalpy of a substance.

        Parameters
        ----------
        id_suffix: str
            A suffix to append to the id of each of the returned protocols.
        gradient_replicator_id: str
            The id of the replicator which will clone those protocols which will
            estimate the gradient of the enthalpy with respect to a given parameter.
        replicator_id: str, optional
            The id of the replicator which will be used to clone these protocols.
            This will be appended to the id of each of the returned protocols if
            set.
        weight_by_mole_fraction: bool
            If true, an extra protocol will be added to weight the calculated
            enthalpy by the mole fraction of the component.
        component_substance_reference: ProtocolPath or PlaceholderValue, optional
            An optional protocol path (or replicator reference) to the component substance
            whose enthalpy is being estimated.
        full_substance_reference: ProtocolPath or PlaceholderValue, optional
            An optional protocol path (or replicator reference) to the full substance
            whose enthalpy of mixing is being estimated. This cannot be `None` if
            `weight_by_mole_fraction` is `True`.
        options: WorkflowOptions
            The options to use when setting up the workflows.
        n_molecules: int
            The number of molecules to use in the simulation.

        Returns
        -------
        BaseSimulationProtocols
            The protocols used to estimate the enthalpy of a substance.
        ProtocolPath
            A reference to the estimated enthalpy.
        WorkflowSimulationDataToStore
            An object which describes the default data from a simulation to store,
            such as the uncorrelated statistics and configurations.
        ProtocolGroup
            The group of protocols which will calculate the gradient of the reduced potential
            with respect to a given property.
        ProtocolReplicator
            The protocol which will replicate the gradient group for every gradient to
            estimate.
        ProtocolPath
            A reference to the value of the gradient.
        """

        if replicator_id is not None:
            id_suffix = f"{id_suffix}_$({replicator_id})"

        if component_substance_reference is None:
            component_substance_reference = ProtocolPath("substance", "global")

        if weight_by_mole_fraction is True and full_substance_reference is None:

            raise ValueError(
                "The full substance reference must be set when weighting by"
                "the mole fraction"
            )

        # Define the protocol which will extract the average enthalpy from
        # the results of a simulation.
        extract_enthalpy = analysis.ExtractAverageStatistic(
            f"extract_enthalpy{id_suffix}"
        )
        extract_enthalpy.statistics_type = ObservableType.Enthalpy

        # Define the protocols which will run the simulation itself.
        (
            simulation_protocols,
            value_source,
            output_to_store,
        ) = generate_base_simulation_protocols(
            extract_enthalpy, options, id_suffix, n_molecules=n_molecules
        )

        number_of_molecules = ProtocolPath(
            "output_number_of_molecules", simulation_protocols.build_coordinates.id
        )
        built_substance = ProtocolPath(
            "output_substance", simulation_protocols.build_coordinates.id
        )

        # Divide the enthalpy by the number of molecules in the system
        extract_enthalpy.divisor = number_of_molecules

        # Use the correct substance.
        simulation_protocols.build_coordinates.substance = component_substance_reference
        simulation_protocols.assign_parameters.substance = built_substance
        output_to_store.substance = built_substance

        conditional_group = simulation_protocols.converge_uncertainty

        if weight_by_mole_fraction:

            # The component workflows need an extra step to multiply their enthalpies by their
            # relative mole fraction.
            weight_by_mole_fraction = miscellaneous.WeightByMoleFraction(
                f"weight_by_mole_fraction{id_suffix}"
            )
            weight_by_mole_fraction.value = ProtocolPath("value", extract_enthalpy.id)
            weight_by_mole_fraction.full_substance = full_substance_reference
            weight_by_mole_fraction.component = component_substance_reference

            conditional_group.add_protocols(weight_by_mole_fraction)

            value_source = ProtocolPath(
                "weighted_value", conditional_group.id, weight_by_mole_fraction.id
            )

        if options.convergence_mode != WorkflowOptions.ConvergenceMode.NoChecks:

            # Make sure the convergence criteria is set to use the per component
            # uncertainty target.
            conditional_group.conditions[0].right_hand_value = ProtocolPath(
                "per_component_uncertainty", "global"
            )

            if weight_by_mole_fraction:
                # Make sure the weighted uncertainty is being used in the conditional comparison.
                conditional_group.conditions[0].left_hand_value = ProtocolPath(
                    "weighted_value.uncertainty",
                    conditional_group.id,
                    weight_by_mole_fraction.id,
                )

        # Set up the gradient calculations
        coordinate_source = ProtocolPath(
            "output_coordinate_file", simulation_protocols.equilibration_simulation.id
        )
        trajectory_source = ProtocolPath(
            "trajectory_file_path",
            simulation_protocols.converge_uncertainty.id,
            simulation_protocols.production_simulation.id,
        )
        statistics_source = ProtocolPath(
            "statistics_file_path",
            simulation_protocols.converge_uncertainty.id,
            simulation_protocols.production_simulation.id,
        )

        reweight_enthalpy_template = reweighting.ReweightStatistics("")
        reweight_enthalpy_template.statistics_type = ObservableType.PotentialEnergy
        reweight_enthalpy_template.statistics_paths = statistics_source
        reweight_enthalpy_template.reference_reduced_potentials = statistics_source

        (
            gradient_group,
            gradient_replicator,
            gradient_source,
        ) = generate_gradient_protocol_group(
            reweight_enthalpy_template,
            ProtocolPath("force_field_path", "global"),
            coordinate_source,
            trajectory_source,
            statistics_source,
            replicator_id=gradient_replicator_id,
            substance_source=built_substance,
            id_suffix=id_suffix,
        )

        # Remove the group id from the path.
        gradient_source.pop_next_in_path()

        if weight_by_mole_fraction:

            # The component workflows need an extra step to multiply their gradients by their
            # relative mole fraction.
            weight_gradient = miscellaneous.WeightByMoleFraction(
                f"weight_gradient_by_mole_fraction{id_suffix}"
            )
            weight_gradient.value = gradient_source
            weight_gradient.full_substance = full_substance_reference
            weight_gradient.component = component_substance_reference

            gradient_group.add_protocols(weight_gradient)
            gradient_source = ProtocolPath("weighted_value", weight_gradient.id)

        scale_gradient = miscellaneous.DivideValue(f"scale_gradient{id_suffix}")
        scale_gradient.value = gradient_source
        scale_gradient.divisor = number_of_molecules

        gradient_group.add_protocols(scale_gradient)
        gradient_source = ProtocolPath("result", gradient_group.id, scale_gradient.id)

        return (
            simulation_protocols,
            value_source,
            output_to_store,
            gradient_group,
            gradient_replicator,
            gradient_source,
        )

    @staticmethod
    def _get_reweighting_protocols(
        id_suffix,
        gradient_replicator_id,
        data_replicator_id,
        replicator_id=None,
        weight_by_mole_fraction=False,
        substance_reference=None,
    ):

        """Returns the set of protocols which when combined in a workflow
        will yield the enthalpy of a substance by reweighting cached data.

        Parameters
        ----------
        id_suffix: str
            A suffix to append to the id of each of the returned protocols.
        gradient_replicator_id: str
            The id of the replicator which will clone those protocols which will
            estimate the gradient of the enthalpy with respect to a given parameter.
        data_replicator_id: str
            The id of the replicator which will be used to clone these protocols
            for each cached simulation data.
        replicator_id: str, optional
            The optional id of the replicator which will be used to clone these
            protocols, e.g. for each component in the system.
        weight_by_mole_fraction: bool
            If true, an extra protocol will be added to weight the calculated
            enthalpy by the mole fraction of the component.
        substance_reference: ProtocolPath or PlaceholderValue, optional
            An optional protocol path (or replicator reference) to the substance
            whose enthalpy is being estimated.

        Returns
        -------
        BaseReweightingProtocols
            The protocols used to estimate the enthalpy of a substance.
        ProtocolPath
            A reference to the estimated enthalpy.
        ProtocolReplicator
            The replicator which will replicate each protocol for each
            cached simulation datum.
        ProtocolGroup
            The group of protocols which will calculate the gradient of the reduced potential
            with respect to a given property.
        ProtocolPath
            A reference to the value of the gradient.
        """

        if replicator_id is not None:
            id_suffix = f"{id_suffix}_$({replicator_id})"

        full_id_suffix = id_suffix

        if data_replicator_id is not None:
            full_id_suffix = f"{id_suffix}_$({data_replicator_id})"

        if substance_reference is None:
            substance_reference = ProtocolPath("substance", "global")

        extract_enthalpy = analysis.ExtractAverageStatistic(
            f"extract_enthalpy{full_id_suffix}"
        )
        extract_enthalpy.statistics_type = ObservableType.Enthalpy
        reweight_enthalpy = reweighting.ReweightStatistics(
            f"reweight_enthalpy{id_suffix}"
        )
        reweight_enthalpy.statistics_type = ObservableType.Enthalpy

        (protocols, data_replicator) = generate_base_reweighting_protocols(
            analysis_protocol=extract_enthalpy,
            mbar_protocol=reweight_enthalpy,
            replicator_id=data_replicator_id,
            id_suffix=id_suffix,
        )

        # Make sure we use the reduced internal potential when re-weighting enthalpies
        protocols.reduced_reference_potential.use_internal_energy = True
        protocols.reduced_target_potential.use_internal_energy = True

        # Make sure to use the correct substance.
        protocols.build_target_system.substance = substance_reference

        value_source = ProtocolPath("value", protocols.mbar_protocol.id)

        # Set up the protocols which will be responsible for adding together
        # the component enthalpies, and subtracting these from the full system enthalpy.
        weight_enthalpy = None

        if weight_by_mole_fraction is True:

            weight_enthalpy = miscellaneous.WeightByMoleFraction(
                f"weight_enthalpy{id_suffix}"
            )
            weight_enthalpy.value = ProtocolPath("value", protocols.mbar_protocol.id)
            weight_enthalpy.full_substance = ProtocolPath("substance", "global")
            weight_enthalpy.component = substance_reference

            value_source = ProtocolPath("weighted_value", weight_enthalpy.id)

        # Divide by the component enthalpies by the number of molecules in the system
        number_of_molecules = ProtocolPath(
            "total_number_of_molecules",
            protocols.unpack_stored_data.id.replace(f"$({data_replicator_id})", "0"),
        )

        divide_by_molecules = miscellaneous.DivideValue(
            f"divide_by_molecules{id_suffix}"
        )
        divide_by_molecules.value = value_source
        divide_by_molecules.divisor = number_of_molecules

        value_source = ProtocolPath("result", divide_by_molecules.id)

        # Set up the gradient calculations.
        reweight_potential_template = copy.deepcopy(reweight_enthalpy)
        reweight_potential_template.statistics_type = ObservableType.PotentialEnergy

        coordinate_path = ProtocolPath(
            "output_coordinate_path", protocols.concatenate_trajectories.id
        )
        trajectory_path = ProtocolPath(
            "output_trajectory_path", protocols.concatenate_trajectories.id
        )
        statistics_path = ProtocolPath(
            "statistics_file_path", protocols.reduced_target_potential.id
        )

        gradient_group, _, gradient_source = generate_gradient_protocol_group(
            reweight_potential_template,
            ProtocolPath("force_field_path", "global"),
            coordinate_path,
            trajectory_path,
            statistics_path,
            replicator_id=gradient_replicator_id,
            id_suffix=id_suffix,
            substance_source=substance_reference,
            effective_sample_indices=ProtocolPath(
                "effective_sample_indices", protocols.mbar_protocol.id
            ),
        )

        # Remove the group id from the path.
        gradient_source.pop_next_in_path()

        if weight_by_mole_fraction is True:

            # The component workflows need an extra step to multiply their gradients by their
            # relative mole fraction.
            weight_gradient = miscellaneous.WeightByMoleFraction(
                f"weight_gradient_$({gradient_replicator_id})_"
                f"by_mole_fraction{id_suffix}"
            )
            weight_gradient.value = gradient_source
            weight_gradient.full_substance = ProtocolPath("substance", "global")
            weight_gradient.component = substance_reference

            gradient_group.add_protocols(weight_gradient)
            gradient_source = ProtocolPath("weighted_value", weight_gradient.id)

        scale_gradient = miscellaneous.DivideValue(
            f"scale_gradient_$({gradient_replicator_id}){id_suffix}"
        )
        scale_gradient.value = gradient_source
        scale_gradient.divisor = number_of_molecules

        gradient_group.add_protocols(scale_gradient)
        gradient_source = ProtocolPath("result", gradient_group.id, scale_gradient.id)

        all_protocols = (*protocols, divide_by_molecules)

        if weight_enthalpy is not None:
            all_protocols = (*all_protocols, weight_enthalpy)

        return (
            all_protocols,
            value_source,
            data_replicator,
            gradient_group,
            gradient_source,
        )

    @staticmethod
    def _default_reweighting_storage_query():
        """Returns the default storage queries to use when
        retrieving cached simulation data to reweight.

        This will include one query (with the key `"full_system_data"`)
        to return data for the full mixture system, and another query
        (with the key `"component_data"`) which will include data for
        each pure component in the system.

        Returns
        -------
        dict of str and SimulationDataQuery
            The dictionary of queries.
        """

        mixture_data_query = SimulationDataQuery()
        mixture_data_query.substance = PlaceholderValue()
        mixture_data_query.property_phase = PropertyPhase.Liquid

        # Set up a query which will return the data of each
        # individual component in the system.
        component_query = SubstanceQuery()
        component_query.components_only = True

        component_data_query = SimulationDataQuery()
        component_data_query.property_phase = PropertyPhase.Liquid
        component_data_query.substance = PlaceholderValue()
        component_data_query.substance_query = component_query

        return {
            "full_system_data": mixture_data_query,
            "component_data": component_data_query,
        }

    @staticmethod
    def default_simulation_schema(
        absolute_tolerance=UNDEFINED, relative_tolerance=UNDEFINED, n_molecules=1000
    ):
        """Returns the default calculation schema to use when estimating
        this class of property from direct simulations.

        Parameters
        ----------
        absolute_tolerance: unit.Quantity, optional
            The absolute tolerance to estimate
            the property to within.
        relative_tolerance: float
            The tolerance (as a fraction of the properties
            reported uncertainty) to estimate the
            property to within.
        n_molecules: int
            The number of molecules to use in the simulation.

        Returns
        -------
        SimulationSchema
            The schema to follow when estimating this property.
        """

        calculation_schema = SimulationSchema()
        calculation_schema.absolute_uncertainty = absolute_tolerance
        calculation_schema.relative_uncertainty_fraction = relative_tolerance

        # Define the id of the replicator which will clone the gradient protocols
        # for each gradient key to be estimated.
        gradient_replicator_id = "gradient_replicator"

        # Set up a workflow to calculate the enthalpy of the full, mixed system.
        (
            full_system_protocols,
            full_system_enthalpy,
            full_output,
            full_system_gradient_group,
            full_system_gradient_replicator,
            full_system_gradient,
        ) = EnthalpyOfMixing._get_simulation_protocols(
            "_full",
            gradient_replicator_id,
            options=calculation_schema.workflow_options,
            n_molecules=n_molecules,
        )

        # Set up a general workflow for calculating the enthalpy of one of the system components.
        component_replicator_id = "component_replicator"
        component_substance = ReplicatorValue(component_replicator_id)

        # Make sure to weight by the mole fractions of the actual full system as these may be slightly
        # different to the mole fractions of the measure property due to rounding.
        full_substance = ProtocolPath(
            "output_substance", full_system_protocols.build_coordinates.id
        )

        (
            component_protocols,
            component_enthalpies,
            component_output,
            component_gradient_group,
            component_gradient_replicator,
            component_gradient,
        ) = EnthalpyOfMixing._get_simulation_protocols(
            "_component",
            gradient_replicator_id,
            replicator_id=component_replicator_id,
            weight_by_mole_fraction=True,
            component_substance_reference=component_substance,
            full_substance_reference=full_substance,
            options=calculation_schema.workflow_options,
            n_molecules=n_molecules,
        )

        # Finally, set up the protocols which will be responsible for adding together
        # the component enthalpies, and subtracting these from the mixed system enthalpy.
        add_component_enthalpies = miscellaneous.AddValues("add_component_enthalpies")
        add_component_enthalpies.values = component_enthalpies

        calculate_enthalpy_of_mixing = miscellaneous.SubtractValues(
            "calculate_enthalpy_of_mixing"
        )
        calculate_enthalpy_of_mixing.value_b = full_system_enthalpy
        calculate_enthalpy_of_mixing.value_a = ProtocolPath(
            "result", add_component_enthalpies.id
        )

        # Create the replicator object which defines how the pure component
        # enthalpy estimation protocols will be replicated for each component.
        component_replicator = ProtocolReplicator(replicator_id=component_replicator_id)
        component_replicator.template_values = ProtocolPath("components", "global")

        # Combine the gradients.
        add_component_gradients = miscellaneous.AddValues(
            f"add_component_gradients" f"_$({gradient_replicator_id})"
        )
        add_component_gradients.values = component_gradient

        combine_gradients = miscellaneous.SubtractValues(
            f"combine_gradients_$({gradient_replicator_id})"
        )
        combine_gradients.value_b = full_system_gradient
        combine_gradients.value_a = ProtocolPath("result", add_component_gradients.id)

        # Combine the gradient replicators.
        gradient_replicator = ProtocolReplicator(replicator_id=gradient_replicator_id)
        gradient_replicator.template_values = ProtocolPath(
            "parameter_gradient_keys", "global"
        )

        # Build the final workflow schema
        schema = WorkflowSchema(property_type=EnthalpyOfMixing.__name__)
        schema.id = "{}{}".format(EnthalpyOfMixing.__name__, "Schema")

        schema.protocols = {
            component_protocols.build_coordinates.id: component_protocols.build_coordinates.schema,
            component_protocols.assign_parameters.id: component_protocols.assign_parameters.schema,
            component_protocols.energy_minimisation.id: component_protocols.energy_minimisation.schema,
            component_protocols.equilibration_simulation.id: component_protocols.equilibration_simulation.schema,
            component_protocols.converge_uncertainty.id: component_protocols.converge_uncertainty.schema,
            full_system_protocols.build_coordinates.id: full_system_protocols.build_coordinates.schema,
            full_system_protocols.assign_parameters.id: full_system_protocols.assign_parameters.schema,
            full_system_protocols.energy_minimisation.id: full_system_protocols.energy_minimisation.schema,
            full_system_protocols.equilibration_simulation.id: full_system_protocols.equilibration_simulation.schema,
            full_system_protocols.converge_uncertainty.id: full_system_protocols.converge_uncertainty.schema,
            component_protocols.extract_uncorrelated_trajectory.id: component_protocols.extract_uncorrelated_trajectory.schema,
            component_protocols.extract_uncorrelated_statistics.id: component_protocols.extract_uncorrelated_statistics.schema,
            full_system_protocols.extract_uncorrelated_trajectory.id: full_system_protocols.extract_uncorrelated_trajectory.schema,
            full_system_protocols.extract_uncorrelated_statistics.id: full_system_protocols.extract_uncorrelated_statistics.schema,
            add_component_enthalpies.id: add_component_enthalpies.schema,
            calculate_enthalpy_of_mixing.id: calculate_enthalpy_of_mixing.schema,
            component_gradient_group.id: component_gradient_group.schema,
            full_system_gradient_group.id: full_system_gradient_group.schema,
            add_component_gradients.id: add_component_gradients.schema,
            combine_gradients.id: combine_gradients.schema,
        }

        schema.replicators = [gradient_replicator, component_replicator]

        # Finally, tell the schemas where to look for its final values.
        schema.gradients_sources = [ProtocolPath("result", combine_gradients.id)]
        schema.final_value_source = ProtocolPath(
            "result", calculate_enthalpy_of_mixing.id
        )

        schema.outputs_to_store = {
            "full_system": full_output,
            f"component_$({component_replicator_id})": component_output,
        }

        calculation_schema.workflow_schema = schema
        return calculation_schema

    @staticmethod
    def default_reweighting_schema(existing_schema=None):
        """Returns the default calculation schema to use when estimating
        this property by reweighting existing data.

        Parameters
        ----------
        existing_schema: ReweightingSchema, optional
            An existing schema whose settings to use. If set,
            the schema's `workflow_schema` will be overwritten
            by this method.

        Returns
        -------
        ReweightingSchema
            The schema to follow when estimating this property.
        """

        calculation_schema = ReweightingSchema()

        if existing_schema is not None:

            assert isinstance(existing_schema, ReweightingSchema)
            calculation_schema = copy.deepcopy(existing_schema)

        # Set up the storage queries
        calculation_schema.storage_queries = (
            EnthalpyOfMixing._default_reweighting_storage_query()
        )

        # Set up a replicator that will re-run the component reweighting workflow for each
        # component in the system.
        component_replicator = ProtocolReplicator(replicator_id="component_replicator")
        component_replicator.template_values = ProtocolPath("components", "global")

        gradient_replicator = ProtocolReplicator("gradient")
        gradient_replicator.template_values = ProtocolPath(
            "parameter_gradient_keys", "global"
        )

        # Set up the protocols which will reweight data for the full system.
        full_data_replicator_id = "full_data_replicator"

        (
            full_protocols,
            full_enthalpy,
            full_data_replicator,
            full_gradient_group,
            full_gradient_source,
        ) = EnthalpyOfMixing._get_reweighting_protocols(
            "_full", gradient_replicator.id, full_data_replicator_id
        )

        # Set up the protocols which will reweight data for each component.
        component_data_replicator_id = (
            f"component_{component_replicator.placeholder_id}_data_replicator"
        )

        (
            component_protocols,
            component_enthalpies,
            component_data_replicator,
            component_gradient_group,
            component_gradient_source,
        ) = EnthalpyOfMixing._get_reweighting_protocols(
            "_component",
            gradient_replicator.id,
            component_data_replicator_id,
            replicator_id=component_replicator.id,
            weight_by_mole_fraction=True,
            substance_reference=ReplicatorValue(component_replicator.id),
        )

        # Make sure the replicator is only replicating over component data.
        component_data_replicator.template_values = ProtocolPath(
            f"component_data[$({component_replicator.id})]", "global"
        )

        add_component_potentials = miscellaneous.AddValues("add_component_potentials")
        add_component_potentials.values = component_enthalpies

        calculate_excess_enthalpy = miscellaneous.SubtractValues(
            "calculate_excess_potential"
        )
        calculate_excess_enthalpy.value_b = full_enthalpy
        calculate_excess_enthalpy.value_a = ProtocolPath(
            "result", add_component_potentials.id
        )

        # Combine the gradients.
        add_component_gradients = miscellaneous.AddValues(
            f"add_component_gradients" f"_{gradient_replicator.placeholder_id}"
        )
        add_component_gradients.values = component_gradient_source

        combine_gradients = miscellaneous.SubtractValues(
            f"combine_gradients_{gradient_replicator.placeholder_id}"
        )
        combine_gradients.value_b = full_gradient_source
        combine_gradients.value_a = ProtocolPath("result", add_component_gradients.id)

        # Build the final workflow schema.
        schema = WorkflowSchema(property_type=EnthalpyOfMixing.__name__)
        schema.id = "{}{}".format(EnthalpyOfMixing.__name__, "Schema")

        schema.protocols = dict()

        schema.protocols.update(
            {protocol.id: protocol.schema for protocol in full_protocols}
        )
        schema.protocols.update(
            {protocol.id: protocol.schema for protocol in component_protocols}
        )

        schema.protocols[add_component_potentials.id] = add_component_potentials.schema
        schema.protocols[
            calculate_excess_enthalpy.id
        ] = calculate_excess_enthalpy.schema

        schema.protocols[full_gradient_group.id] = full_gradient_group.schema
        schema.protocols[component_gradient_group.id] = component_gradient_group.schema
        schema.protocols[add_component_gradients.id] = add_component_gradients.schema
        schema.protocols[combine_gradients.id] = combine_gradients.schema

        schema.replicators = [
            full_data_replicator,
            component_replicator,
            component_data_replicator,
            gradient_replicator,
        ]

        schema.gradients_sources = [ProtocolPath("result", combine_gradients.id)]
        schema.final_value_source = ProtocolPath("result", calculate_excess_enthalpy.id)

        calculation_schema.workflow_schema = schema
        return calculation_schema


@thermoml_property(
    "Molar enthalpy of vaporization or sublimation, kJ/mol",
    supported_phases=PropertyPhase.Liquid | PropertyPhase.Gas,
)
class EnthalpyOfVaporization(PhysicalProperty):
    """A class representation of an enthalpy of vaporization property"""

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
        absolute_tolerance=UNDEFINED,
        relative_tolerance=UNDEFINED,
        n_molecules=1000,
    ):
        """Returns the default calculation schema to use when estimating
        this class of property from direct simulations.

        Parameters
        ----------
        absolute_tolerance: unit.Quantity, optional
            The absolute tolerance to estimate
            the property to within.
        relative_tolerance: float
            The tolerance (as a fraction of the properties
            reported uncertainty) to estimate the
            property to within.
        n_molecules: int
            The number of molecules to use in the simulation.

        Returns
        -------
        SimulationSchema
            The schema to follow when estimating this property.
        """

        calculation_schema = SimulationSchema()
        calculation_schema.absolute_uncertainty = absolute_tolerance
        calculation_schema.relative_uncertainty_fraction = relative_tolerance

        # Define a custom conditional group.
        converge_uncertainty = groups.ConditionalGroup(f"converge_uncertainty")
        converge_uncertainty.max_iterations = 100

        # Define the protocols to perform the simulation in the liquid phase.
        extract_liquid_energy = analysis.ExtractAverageStatistic(
            "extract_liquid_energy"
        )
        extract_liquid_energy.statistics_type = ObservableType.PotentialEnergy
        extract_liquid_energy.divisor = n_molecules

        (
            liquid_protocols,
            liquid_value_source,
            liquid_output_to_store,
        ) = generate_base_simulation_protocols(
            extract_liquid_energy,
            calculation_schema.workflow_options,
            "_liquid",
            converge_uncertainty,
        )

        # Make sure the number of molecules in the liquid is consistent.
        liquid_protocols.build_coordinates.max_molecules = n_molecules
        liquid_output_to_store.property_phase = PropertyPhase.Liquid

        # Define the protocols to perform the simulation in the gas phase.
        extract_gas_energy = analysis.ExtractAverageStatistic("extract_gas_energy")
        extract_gas_energy.statistics_type = ObservableType.PotentialEnergy

        (
            gas_protocols,
            gas_value_source,
            gas_output_to_store,
        ) = generate_base_simulation_protocols(
            extract_gas_energy,
            calculation_schema.workflow_options,
            "_gas",
            converge_uncertainty,
        )

        # Create only a single molecule in vacuum
        gas_protocols.build_coordinates.max_molecules = 1
        gas_output_to_store.property_phase = PropertyPhase.Gas

        # Run the gas phase simulations in the NVT ensemble
        gas_protocols.energy_minimisation.enable_pbc = False
        gas_protocols.equilibration_simulation.ensemble = Ensemble.NVT
        gas_protocols.equilibration_simulation.enable_pbc = False
        gas_protocols.production_simulation.ensemble = Ensemble.NVT
        gas_protocols.production_simulation.steps_per_iteration = 15000000
        gas_protocols.production_simulation.output_frequency = 5000
        gas_protocols.production_simulation.checkpoint_frequency = 100
        gas_protocols.production_simulation.enable_pbc = False

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
        energy_of_vaporization.value_b = ProtocolPath("value", extract_gas_energy.id)
        energy_of_vaporization.value_a = ProtocolPath("value", extract_liquid_energy.id)

        ideal_volume = miscellaneous.MultiplyValue("ideal_volume")
        ideal_volume.value = EstimatedQuantity(
            1.0 * unit.molar_gas_constant,
            0.0 * unit.joule / unit.mole / unit.kelvin,
            "Universal Constant",
        )
        ideal_volume.multiplier = ProtocolPath(
            "thermodynamic_state.temperature", "global"
        )

        enthalpy_of_vaporization = miscellaneous.AddValues("enthalpy_of_vaporization")
        enthalpy_of_vaporization.values = [
            ProtocolPath("result", energy_of_vaporization.id),
            ProtocolPath("result", ideal_volume.id),
        ]

        # Add the extra protocols and conditions to the custom group.
        converge_uncertainty.add_protocols(
            energy_of_vaporization, ideal_volume, enthalpy_of_vaporization
        )

        if (
            calculation_schema.workflow_options.convergence_mode
            != WorkflowOptions.ConvergenceMode.NoChecks
        ):

            condition = groups.ConditionalGroup.Condition()
            condition.condition_type = groups.ConditionalGroup.ConditionType.LessThan

            condition.left_hand_value = ProtocolPath(
                "result.uncertainty",
                converge_uncertainty.id,
                enthalpy_of_vaporization.id,
            )
            condition.right_hand_value = ProtocolPath("target_uncertainty", "global")

            gas_protocols.production_simulation.total_number_of_iterations = ProtocolPath(
                "current_iteration", converge_uncertainty.id
            )

            liquid_protocols.production_simulation.total_number_of_iterations = ProtocolPath(
                "current_iteration", converge_uncertainty.id
            )

            converge_uncertainty.add_condition(condition)

        # Set up the liquid gradient calculations
        liquid_coordinate_source = ProtocolPath(
            "output_coordinate_file", liquid_protocols.equilibration_simulation.id
        )
        liquid_trajectory_source = ProtocolPath(
            "trajectory_file_path",
            liquid_protocols.converge_uncertainty.id,
            liquid_protocols.production_simulation.id,
        )
        liquid_statistics_source = ProtocolPath(
            "statistics_file_path",
            liquid_protocols.converge_uncertainty.id,
            liquid_protocols.production_simulation.id,
        )

        reweight_potential_template = reweighting.ReweightStatistics("")
        reweight_potential_template.statistics_type = ObservableType.PotentialEnergy
        reweight_potential_template.reference_reduced_potentials = (
            liquid_statistics_source
        )

        (
            liquid_gradient_group,
            liquid_gradient_replicator,
            liquid_gradient_source,
        ) = generate_gradient_protocol_group(
            reweight_potential_template,
            ProtocolPath("force_field_path", "global"),
            liquid_coordinate_source,
            liquid_trajectory_source,
            liquid_statistics_source,
            id_suffix="_liquid",
        )

        # Set up the gas gradient calculations
        gas_coordinate_source = ProtocolPath(
            "output_coordinate_file", gas_protocols.equilibration_simulation.id
        )
        gas_trajectory_source = ProtocolPath(
            "trajectory_file_path",
            gas_protocols.converge_uncertainty.id,
            gas_protocols.production_simulation.id,
        )
        gas_statistics_source = ProtocolPath(
            "statistics_file_path",
            gas_protocols.converge_uncertainty.id,
            gas_protocols.production_simulation.id,
        )

        reweight_potential_template.reference_reduced_potentials = gas_statistics_source

        (
            gas_gradient_group,
            gas_gradient_replicator,
            gas_gradient_source,
        ) = generate_gradient_protocol_group(
            reweight_potential_template,
            ProtocolPath("force_field_path", "global"),
            gas_coordinate_source,
            gas_trajectory_source,
            gas_statistics_source,
            id_suffix="_gas",
            enable_pbc=False,
        )

        # Combine the gradients.
        scale_liquid_gradient = miscellaneous.DivideValue(
            "scale_liquid_gradient_$(repl)"
        )
        scale_liquid_gradient.value = liquid_gradient_source
        scale_liquid_gradient.divisor = n_molecules

        combine_gradients = miscellaneous.SubtractValues("combine_gradients_$(repl)")
        combine_gradients.value_b = gas_gradient_source
        combine_gradients.value_a = ProtocolPath("result", scale_liquid_gradient.id)

        # Combine the gradient replicators.
        gradient_replicator = ProtocolReplicator(
            replicator_id=liquid_gradient_replicator.id
        )
        gradient_replicator.template_values = ProtocolPath(
            "parameter_gradient_keys", "global"
        )

        # Build the workflow schema.
        schema = WorkflowSchema(property_type=EnthalpyOfVaporization.__name__)
        schema.id = "{}{}".format(EnthalpyOfVaporization.__name__, "Schema")

        schema.protocols = {
            liquid_protocols.build_coordinates.id: liquid_protocols.build_coordinates.schema,
            liquid_protocols.assign_parameters.id: liquid_protocols.assign_parameters.schema,
            liquid_protocols.energy_minimisation.id: liquid_protocols.energy_minimisation.schema,
            liquid_protocols.equilibration_simulation.id: liquid_protocols.equilibration_simulation.schema,
            gas_protocols.build_coordinates.id: gas_protocols.build_coordinates.schema,
            gas_protocols.assign_parameters.id: gas_protocols.assign_parameters.schema,
            gas_protocols.energy_minimisation.id: gas_protocols.energy_minimisation.schema,
            gas_protocols.equilibration_simulation.id: gas_protocols.equilibration_simulation.schema,
            converge_uncertainty.id: converge_uncertainty.schema,
            liquid_protocols.extract_uncorrelated_trajectory.id: liquid_protocols.extract_uncorrelated_trajectory.schema,
            liquid_protocols.extract_uncorrelated_statistics.id: liquid_protocols.extract_uncorrelated_statistics.schema,
            gas_protocols.extract_uncorrelated_trajectory.id: gas_protocols.extract_uncorrelated_trajectory.schema,
            gas_protocols.extract_uncorrelated_statistics.id: gas_protocols.extract_uncorrelated_statistics.schema,
            liquid_gradient_group.id: liquid_gradient_group.schema,
            gas_gradient_group.id: gas_gradient_group.schema,
            scale_liquid_gradient.id: scale_liquid_gradient.schema,
            combine_gradients.id: combine_gradients.schema,
        }

        schema.replicators = [gradient_replicator]

        schema.outputs_to_store = {
            "liquid_data": liquid_output_to_store,
            "gas_data": gas_output_to_store,
        }

        schema.gradients_sources = [ProtocolPath("result", combine_gradients.id)]
        schema.final_value_source = ProtocolPath(
            "result", converge_uncertainty.id, enthalpy_of_vaporization.id
        )

        calculation_schema.workflow_schema = schema
        return calculation_schema

    @staticmethod
    def default_reweighting_schema(existing_schema=None):
        """Returns the default calculation schema to use when estimating
        this property by reweighting existing data.

        Parameters
        ----------
        existing_schema: ReweightingSchema, optional
            An existing schema whose settings to use. If set,
            the schema's `workflow_schema` will be overwritten
            by this method.

        Returns
        -------
        ReweightingSchema
            The schema to follow when estimating this property.
        """

        calculation_schema = ReweightingSchema()

        if existing_schema is not None:

            assert isinstance(existing_schema, ReweightingSchema)
            calculation_schema = copy.deepcopy(existing_schema)

        # Set up the storage queries
        calculation_schema.storage_queries = (
            EnthalpyOfVaporization._default_reweighting_storage_query()
        )

        # Set up the data replicators
        liquid_data_replicator = ProtocolReplicator("liquid_data_replicator")
        liquid_data_replicator.template_values = ProtocolPath("liquid_data", "global")

        gas_data_replicator = ProtocolReplicator("gas_data_replicator")
        gas_data_replicator.template_values = ProtocolPath("gas_data", "global")

        # Set up the gradient replicator
        gradient_replicator = ProtocolReplicator("gradient_replicator")
        gradient_replicator.template_values = ProtocolPath(
            "parameter_gradient_keys", "global"
        )

        # Set up a protocol to extract the liquid phase energy from the existing data.
        extract_liquid_energy = analysis.ExtractAverageStatistic(
            f"extract_liquid_energy_" f"{liquid_data_replicator.placeholder_id}"
        )
        extract_liquid_energy.statistics_type = ObservableType.PotentialEnergy

        reweight_liquid_energy = reweighting.ReweightStatistics(
            "reweight_liquid_energy"
        )
        reweight_liquid_energy.statistics_type = ObservableType.PotentialEnergy

        liquid_protocols, _ = generate_base_reweighting_protocols(
            extract_liquid_energy,
            reweight_liquid_energy,
            id_suffix="_liquid",
            replicator_id=liquid_data_replicator.id,
        )

        # Extract the number of liquid phase molecules from the first data collection.
        number_of_liquid_molecules = ProtocolPath(
            "total_number_of_molecules",
            liquid_protocols.unpack_stored_data.id.replace(
                liquid_data_replicator.placeholder_id, "0"
            ),
        )

        divide_by_liquid_molecules = miscellaneous.DivideValue(
            "divide_by_liquid_molecules"
        )
        divide_by_liquid_molecules.value = ProtocolPath(
            "value", liquid_protocols.mbar_protocol.id
        )
        divide_by_liquid_molecules.divisor = number_of_liquid_molecules

        # Set up a protocol to extract the gas phase energy from the existing data.
        extract_gas_energy = analysis.ExtractAverageStatistic(
            "extract_gas_energy_" f"{gas_data_replicator.placeholder_id}"
        )
        extract_gas_energy.statistics_type = ObservableType.PotentialEnergy

        reweight_gas_energy = reweighting.ReweightStatistics("reweight_gas_energy")
        reweight_gas_energy.statistics_type = ObservableType.PotentialEnergy

        gas_protocols, _ = generate_base_reweighting_protocols(
            extract_gas_energy,
            reweight_gas_energy,
            id_suffix="_gas",
            replicator_id=gas_data_replicator.id,
        )

        # Turn of PBC for the gas phase.
        gas_protocols.reduced_reference_potential.enable_pbc = False
        gas_protocols.reduced_target_potential.enable_pbc = False

        # Combine the values to estimate the final enthalpy of vaporization
        energy_of_vaporization = miscellaneous.SubtractValues("energy_of_vaporization")
        energy_of_vaporization.value_b = ProtocolPath(
            "value", gas_protocols.mbar_protocol.id
        )
        energy_of_vaporization.value_a = ProtocolPath(
            "result", divide_by_liquid_molecules.id
        )

        ideal_volume = miscellaneous.MultiplyValue("ideal_volume")
        ideal_volume.value = EstimatedQuantity(
            1.0 * unit.molar_gas_constant,
            0.0 * unit.joule / unit.mole / unit.kelvin,
            "Universal Constant",
        )
        ideal_volume.multiplier = ProtocolPath(
            "thermodynamic_state.temperature", "global"
        )

        enthalpy_of_vaporization = miscellaneous.AddValues("enthalpy_of_vaporization")
        enthalpy_of_vaporization.values = [
            ProtocolPath("result", energy_of_vaporization.id),
            ProtocolPath("result", ideal_volume.id),
        ]

        # Set up the liquid phase gradient calculations
        reweight_potential_template = copy.deepcopy(reweight_liquid_energy)

        liquid_coordinate_path = ProtocolPath(
            "output_coordinate_path", liquid_protocols.concatenate_trajectories.id
        )
        liquid_trajectory_path = ProtocolPath(
            "output_trajectory_path", liquid_protocols.concatenate_trajectories.id
        )
        liquid_statistics_path = ProtocolPath(
            "statistics_file_path", liquid_protocols.reduced_target_potential.id
        )

        (
            liquid_gradient_group,
            _,
            liquid_gradient_source,
        ) = generate_gradient_protocol_group(
            reweight_potential_template,
            ProtocolPath("force_field_path", "global"),
            liquid_coordinate_path,
            liquid_trajectory_path,
            liquid_statistics_path,
            replicator_id=gradient_replicator.id,
            id_suffix="_liquid",
            effective_sample_indices=ProtocolPath(
                "effective_sample_indices", liquid_protocols.mbar_protocol.id
            ),
        )

        # Set up the gas phase gradient calculations
        reweight_potential_template = copy.deepcopy(reweight_gas_energy)

        gas_coordinate_path = ProtocolPath(
            "output_coordinate_path", gas_protocols.concatenate_trajectories.id
        )
        gas_trajectory_path = ProtocolPath(
            "output_trajectory_path", gas_protocols.concatenate_trajectories.id
        )
        gas_statistics_path = ProtocolPath(
            "statistics_file_path", gas_protocols.reduced_target_potential.id
        )

        gas_gradient_group, _, gas_gradient_source = generate_gradient_protocol_group(
            reweight_potential_template,
            ProtocolPath("force_field_path", "global"),
            gas_coordinate_path,
            gas_trajectory_path,
            gas_statistics_path,
            replicator_id=gradient_replicator.id,
            id_suffix="_gas",
            enable_pbc=False,
            effective_sample_indices=ProtocolPath(
                "effective_sample_indices", gas_protocols.mbar_protocol.id
            ),
        )

        # Combine the gradients.
        divide_liquid_gradient = miscellaneous.DivideValue(
            f"divide_liquid_gradient_" f"{gradient_replicator.placeholder_id}"
        )
        divide_liquid_gradient.value = liquid_gradient_source
        divide_liquid_gradient.divisor = number_of_liquid_molecules

        combine_gradients = miscellaneous.SubtractValues(
            f"combine_gradients_{gradient_replicator.placeholder_id}"
        )
        combine_gradients.value_b = gas_gradient_source
        combine_gradients.value_a = ProtocolPath("result", divide_liquid_gradient.id)

        # Build the workflow schema.
        schema = WorkflowSchema(property_type=EnthalpyOfVaporization.__name__)
        schema.id = "{}{}".format(EnthalpyOfVaporization.__name__, "Schema")

        schema.protocols.update(
            {protocol.id: protocol.schema for protocol in liquid_protocols}
        )
        schema.protocols.update(
            {protocol.id: protocol.schema for protocol in gas_protocols}
        )

        schema.protocols[
            divide_by_liquid_molecules.id
        ] = divide_by_liquid_molecules.schema
        schema.protocols[energy_of_vaporization.id] = energy_of_vaporization.schema
        schema.protocols[ideal_volume.id] = ideal_volume.schema
        schema.protocols[enthalpy_of_vaporization.id] = enthalpy_of_vaporization.schema

        schema.protocols[liquid_gradient_group.id] = liquid_gradient_group.schema
        schema.protocols[gas_gradient_group.id] = gas_gradient_group.schema
        schema.protocols[divide_liquid_gradient.id] = divide_liquid_gradient.schema
        schema.protocols[combine_gradients.id] = combine_gradients.schema

        schema.replicators = [
            liquid_data_replicator,
            gas_data_replicator,
            gradient_replicator,
        ]

        schema.gradients_sources = [ProtocolPath("result", combine_gradients.id)]
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
    EnthalpyOfVaporization,
    SimulationLayer,
    EnthalpyOfVaporization.default_simulation_schema,
)
register_calculation_schema(
    EnthalpyOfVaporization,
    ReweightingLayer,
    EnthalpyOfVaporization.default_reweighting_schema,
)
