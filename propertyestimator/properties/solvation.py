"""
A collection of physical property definitions relating to
solvation free energies.
"""
from propertyestimator import unit
from propertyestimator.properties import PhysicalProperty
from propertyestimator.properties.plugins import register_estimable_property
from propertyestimator.protocols import (
    coordinates,
    forcefield,
    groups,
    miscellaneous,
    simulation,
    yank,
)
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import Ensemble
from propertyestimator.workflow import WorkflowOptions
from propertyestimator.workflow.schemas import WorkflowSchema
from propertyestimator.workflow.utils import ProtocolPath


@register_estimable_property()
class SolvationFreeEnergy(PhysicalProperty):
    """A class representation of a solvation free energy property."""

    @staticmethod
    def get_default_workflow_schema(calculation_layer, options=None):

        if calculation_layer == "SimulationLayer":
            # Currently reweighting is not supported.
            return SolvationFreeEnergy.get_default_simulation_workflow_schema(options)

        return None

    @staticmethod
    def get_default_simulation_workflow_schema(options=None):
        """Returns the default workflow to use when estimating this property
        from direct simulations.

        Parameters
        ----------
        options: WorkflowOptions
            The default options to use when setting up the estimation workflow.

        Returns
        -------
        WorkflowSchema
            The schema to follow when estimating this property.
        """

        # Setup the fully solvated systems.
        build_full_coordinates = coordinates.BuildCoordinatesPackmol(
            "build_solvated_coordinates"
        )
        build_full_coordinates.substance = ProtocolPath("substance", "global")
        build_full_coordinates.max_molecules = 2000

        assign_full_parameters = forcefield.BuildSmirnoffSystem(
            f"assign_solvated_parameters"
        )
        assign_full_parameters.force_field_path = ProtocolPath(
            "force_field_path", "global"
        )
        assign_full_parameters.substance = ProtocolPath("substance", "global")
        assign_full_parameters.coordinate_file_path = ProtocolPath(
            "coordinate_file_path", build_full_coordinates.id
        )

        # Perform a quick minimisation of the full system to give
        # YANK a better starting point for its minimisation.
        energy_minimisation = simulation.RunEnergyMinimisation("energy_minimisation")
        energy_minimisation.system_path = ProtocolPath(
            "system_path", assign_full_parameters.id
        )
        energy_minimisation.input_coordinate_file = ProtocolPath(
            "coordinate_file_path", build_full_coordinates.id
        )

        equilibration_simulation = simulation.RunOpenMMSimulation(
            "equilibration_simulation"
        )
        equilibration_simulation.ensemble = Ensemble.NPT
        equilibration_simulation.steps_per_iteration = 100000
        equilibration_simulation.output_frequency = 10000
        equilibration_simulation.timestep = 2.0 * unit.femtosecond
        equilibration_simulation.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )
        equilibration_simulation.system_path = ProtocolPath(
            "system_path", assign_full_parameters.id
        )
        equilibration_simulation.input_coordinate_file = ProtocolPath(
            "output_coordinate_file", energy_minimisation.id
        )

        # Create a substance which only contains the solute (e.g. for the
        # vacuum phase simulations).
        filter_solvent = miscellaneous.FilterSubstanceByRole("filter_solvent")
        filter_solvent.input_substance = ProtocolPath("substance", "global")
        filter_solvent.component_role = Substance.ComponentRole.Solvent

        filter_solute = miscellaneous.FilterSubstanceByRole("filter_solute")
        filter_solute.input_substance = ProtocolPath("substance", "global")
        filter_solute.component_role = Substance.ComponentRole.Solute

        # Setup the solute in vacuum system.
        build_vacuum_coordinates = coordinates.BuildCoordinatesPackmol(
            "build_vacuum_coordinates"
        )
        build_vacuum_coordinates.substance = ProtocolPath(
            "filtered_substance", filter_solute.id
        )
        build_vacuum_coordinates.max_molecules = 1

        assign_vacuum_parameters = forcefield.BuildSmirnoffSystem(f"assign_parameters")
        assign_vacuum_parameters.force_field_path = ProtocolPath(
            "force_field_path", "global"
        )
        assign_vacuum_parameters.substance = ProtocolPath(
            "filtered_substance", filter_solute.id
        )
        assign_vacuum_parameters.coordinate_file_path = ProtocolPath(
            "coordinate_file_path", build_vacuum_coordinates.id
        )

        # Set up the protocol to run yank.
        run_yank = yank.SolvationYankProtocol("run_solvation_yank")
        run_yank.solute = ProtocolPath("filtered_substance", filter_solute.id)
        run_yank.solvent_1 = ProtocolPath("filtered_substance", filter_solvent.id)
        run_yank.solvent_2 = Substance()
        run_yank.thermodynamic_state = ProtocolPath("thermodynamic_state", "global")
        run_yank.steps_per_iteration = 500
        run_yank.checkpoint_interval = 50
        run_yank.solvent_1_coordinates = ProtocolPath(
            "output_coordinate_file", equilibration_simulation.id
        )
        run_yank.solvent_1_system = ProtocolPath(
            "system_path", assign_full_parameters.id
        )
        run_yank.solvent_2_coordinates = ProtocolPath(
            "coordinate_file_path", build_vacuum_coordinates.id
        )
        run_yank.solvent_2_system = ProtocolPath(
            "system_path", assign_vacuum_parameters.id
        )

        # Set up the group which will run yank until the free energy has been determined to within
        # a given uncertainty
        conditional_group = groups.ConditionalGroup(f"conditional_group")
        conditional_group.max_iterations = 20

        if options.convergence_mode != WorkflowOptions.ConvergenceMode.NoChecks:

            condition = groups.ConditionalGroup.Condition()
            condition.condition_type = groups.ConditionalGroup.ConditionType.LessThan
            condition.right_hand_value = ProtocolPath("target_uncertainty", "global")
            condition.left_hand_value = ProtocolPath(
                "estimated_free_energy.uncertainty", conditional_group.id, run_yank.id
            )

            conditional_group.add_condition(condition)

        # Define the total number of iterations that yank should run for.
        total_iterations = miscellaneous.MultiplyValue("total_iterations")
        total_iterations.value = 2000
        total_iterations.multiplier = ProtocolPath(
            "current_iteration", conditional_group.id
        )

        # Make sure the simulations gets extended after each iteration.
        run_yank.number_of_iterations = ProtocolPath("result", total_iterations.id)

        conditional_group.add_protocols(total_iterations, run_yank)

        # Define the full workflow schema.
        schema = WorkflowSchema(property_type=SolvationFreeEnergy.__name__)
        schema.id = "{}{}".format(SolvationFreeEnergy.__name__, "Schema")

        schema.protocols = {
            build_full_coordinates.id: build_full_coordinates.schema,
            assign_full_parameters.id: assign_full_parameters.schema,
            energy_minimisation.id: energy_minimisation.schema,
            equilibration_simulation.id: equilibration_simulation.schema,
            filter_solvent.id: filter_solvent.schema,
            filter_solute.id: filter_solute.schema,
            build_vacuum_coordinates.id: build_vacuum_coordinates.schema,
            assign_vacuum_parameters.id: assign_vacuum_parameters.schema,
            conditional_group.id: conditional_group.schema,
        }

        schema.final_value_source = ProtocolPath(
            "estimated_free_energy", conditional_group.id, run_yank.id
        )
        return schema
