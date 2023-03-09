"""
A collection of physical property definitions relating to
solvation free energies.
"""
from openff.units import unit

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.datasets import PhysicalProperty
from openff.evaluator.layers import register_calculation_schema
from openff.evaluator.layers.simulation import SimulationLayer, SimulationSchema
from openff.evaluator.protocols import (
    coordinates,
    forcefield,
    groups,
    miscellaneous,
    openmm,
    yank,
)
from openff.evaluator.substances import Component, Substance
from openff.evaluator.thermodynamics import Ensemble
from openff.evaluator.workflow import WorkflowSchema
from openff.evaluator.workflow.utils import ProtocolPath


class SolvationFreeEnergy(PhysicalProperty):
    """A class representation of a solvation free energy property."""

    @classmethod
    def default_unit(cls):
        return unit.kilojoule / unit.mole

    @staticmethod
    def default_simulation_schema(
        absolute_tolerance=UNDEFINED, relative_tolerance=UNDEFINED, n_molecules=2000
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

        # Setup the fully solvated systems.
        build_full_coordinates = coordinates.BuildCoordinatesPackmol(
            "build_solvated_coordinates"
        )
        build_full_coordinates.substance = ProtocolPath("substance", "global")
        build_full_coordinates.max_molecules = n_molecules

        assign_full_parameters = forcefield.BaseBuildSystem(
            "assign_solvated_parameters"
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
        energy_minimisation = openmm.OpenMMEnergyMinimisation("energy_minimisation")
        energy_minimisation.parameterized_system = ProtocolPath(
            "parameterized_system", assign_full_parameters.id
        )
        energy_minimisation.input_coordinate_file = ProtocolPath(
            "coordinate_file_path", build_full_coordinates.id
        )

        equilibration_simulation = openmm.OpenMMSimulation("equilibration_simulation")
        equilibration_simulation.ensemble = Ensemble.NPT
        equilibration_simulation.steps_per_iteration = 100000
        equilibration_simulation.output_frequency = 10000
        equilibration_simulation.timestep = 2.0 * unit.femtosecond
        equilibration_simulation.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )
        equilibration_simulation.parameterized_system = ProtocolPath(
            "parameterized_system", assign_full_parameters.id
        )
        equilibration_simulation.input_coordinate_file = ProtocolPath(
            "output_coordinate_file", energy_minimisation.id
        )

        # Create a substance which only contains the solute (e.g. for the
        # vacuum phase simulations).
        filter_solvent = miscellaneous.FilterSubstanceByRole("filter_solvent")
        filter_solvent.input_substance = ProtocolPath("substance", "global")
        filter_solvent.component_roles = [Component.Role.Solvent]

        filter_solute = miscellaneous.FilterSubstanceByRole("filter_solute")
        filter_solute.input_substance = ProtocolPath("substance", "global")
        filter_solute.component_roles = [Component.Role.Solute]

        # Setup the solute in vacuum system.
        build_vacuum_coordinates = coordinates.BuildCoordinatesPackmol(
            "build_vacuum_coordinates"
        )
        build_vacuum_coordinates.substance = ProtocolPath(
            "filtered_substance", filter_solute.id
        )
        build_vacuum_coordinates.max_molecules = 1

        assign_vacuum_parameters = forcefield.BaseBuildSystem("assign_parameters")
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
        run_yank.checkpoint_interval = 1
        run_yank.solution_1_coordinates = ProtocolPath(
            "output_coordinate_file", equilibration_simulation.id
        )
        run_yank.solution_1_system = ProtocolPath(
            "parameterized_system", assign_full_parameters.id
        )
        run_yank.solution_2_coordinates = ProtocolPath(
            "coordinate_file_path", build_vacuum_coordinates.id
        )
        run_yank.solution_2_system = ProtocolPath(
            "parameterized_system", assign_vacuum_parameters.id
        )
        run_yank.gradient_parameters = ProtocolPath("parameter_gradient_keys", "global")

        # Set up the group which will run yank until the free energy has been determined
        # to within a given uncertainty
        conditional_group = groups.ConditionalGroup("conditional_group")
        conditional_group.max_iterations = 20

        if use_target_uncertainty:
            condition = groups.ConditionalGroup.Condition()
            condition.type = groups.ConditionalGroup.Condition.Type.LessThan
            condition.right_hand_value = ProtocolPath("target_uncertainty", "global")
            condition.left_hand_value = ProtocolPath(
                "free_energy_difference.error", conditional_group.id, run_yank.id
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
        schema = WorkflowSchema()

        schema.protocol_schemas = [
            build_full_coordinates.schema,
            assign_full_parameters.schema,
            energy_minimisation.schema,
            equilibration_simulation.schema,
            filter_solvent.schema,
            filter_solute.schema,
            build_vacuum_coordinates.schema,
            assign_vacuum_parameters.schema,
            conditional_group.schema,
        ]

        schema.final_value_source = ProtocolPath(
            "free_energy_difference", conditional_group.id, run_yank.id
        )

        calculation_schema.workflow_schema = schema
        return calculation_schema


# Register the properties via the plugin system.
register_calculation_schema(
    SolvationFreeEnergy, SimulationLayer, SolvationFreeEnergy.default_simulation_schema
)
