"""
A collection of density physical property definitions.
"""

from propertyestimator.properties import PhysicalProperty
from propertyestimator.properties.plugins import register_estimable_property
from propertyestimator.protocols import coordinates, forcefield, simulation, miscellaneous
from propertyestimator.substances import Substance
from propertyestimator.workflow.schemas import WorkflowSchema
from propertyestimator.workflow.utils import ProtocolPath


@register_estimable_property()
class HostGuestBindingAffinity(PhysicalProperty):
    """A class representation of a host-guest binding affinity property"""

    @property
    def multi_component_property(self):
        """Returns whether this property is dependant on properties of the
        full mixed substance, or whether it is also dependant on the properties
        of the individual components also.
        """
        return False

    @staticmethod
    def get_default_workflow_schema(calculation_layer, options=None):

        if calculation_layer == 'SimulationLayer':
            return HostGuestBindingAffinity.get_default_simulation_workflow_schema(options)

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

        schema = WorkflowSchema(property_type=HostGuestBindingAffinity.__name__)
        schema.id = '{}{}'.format(HostGuestBindingAffinity.__name__, 'Schema')

        # Initial coordinate and topology setup.
        filter_ligand = miscellaneous.FilterSubstanceByRole('filter_ligand')
        filter_ligand.input_substance = ProtocolPath('substance', 'global')

        filter_ligand.component_role = Substance.ComponentRole.Ligand
        # We only support substances with a single guest ligand.
        filter_ligand.expected_components = 1

        schema.protocols[filter_ligand.id] = filter_ligand.schema

        # Construct the protocols which will (for now) take as input a set of host coordinates,
        # and generate a set of charges for them.
        filter_receptor = miscellaneous.FilterSubstanceByRole('filter_receptor')
        filter_receptor.input_substance = ProtocolPath('substance', 'global')

        filter_receptor.component_role = Substance.ComponentRole.Receptor
        # We only support substances with a single host receptor.
        filter_receptor.expected_components = 1

        schema.protocols[filter_receptor.id] = filter_receptor.schema

        # Perform docking to position the guest within the host.
        perform_docking = coordinates.BuildDockedCoordinates('perform_docking')

        perform_docking.ligand_substance = ProtocolPath('filtered_substance', filter_ligand.id)
        perform_docking.receptor_coordinate_file = ProtocolPath('receptor_mol2', 'global')

        schema.protocols[perform_docking.id] = perform_docking.schema

        # Solvate the docked structure using packmol
        filter_solvent = miscellaneous.FilterSubstanceByRole('filter_solvent')
        filter_solvent.input_substance = ProtocolPath('substance', 'global')
        filter_solvent.component_role = Substance.ComponentRole.Solvent

        schema.protocols[filter_solvent.id] = filter_solvent.schema

        solvate_complex = coordinates.SolvateExistingStructure('solvate_complex')
        solvate_complex.max_molecules = 1000

        solvate_complex.substance = ProtocolPath('filtered_substance', filter_solvent.id)
        solvate_complex.solute_coordinate_file = ProtocolPath('docked_complex_coordinate_path', perform_docking.id)

        schema.protocols[solvate_complex.id] = solvate_complex.schema

        # Assign force field parameters to the solvated complex system.
        build_solvated_complex_system = forcefield.BuildSmirnoffSystem('build_solvated_complex_system')

        build_solvated_complex_system.force_field_path = ProtocolPath('force_field_path', 'global')

        build_solvated_complex_system.coordinate_file_path = ProtocolPath('coordinate_file_path', solvate_complex.id)
        build_solvated_complex_system.substance = ProtocolPath('substance', 'global')

        build_solvated_complex_system.charged_molecule_paths = [ProtocolPath('receptor_mol2', 'global')]

        schema.protocols[build_solvated_complex_system.id] = build_solvated_complex_system.schema

        # Solvate the ligand using packmol
        solvate_ligand = coordinates.SolvateExistingStructure('solvate_ligand')
        solvate_ligand.max_molecules = 1000

        solvate_ligand.substance = ProtocolPath('filtered_substance', filter_solvent.id)
        solvate_ligand.solute_coordinate_file = ProtocolPath('docked_ligand_coordinate_path', perform_docking.id)

        schema.protocols[solvate_ligand.id] = solvate_ligand.schema

        # Assign force field parameters to the solvated ligand system.
        build_solvated_ligand_system = forcefield.BuildSmirnoffSystem('build_solvated_ligand_system')

        build_solvated_ligand_system.force_field_path = ProtocolPath('force_field_path', 'global')

        build_solvated_ligand_system.coordinate_file_path = ProtocolPath('coordinate_file_path', solvate_ligand.id)
        build_solvated_ligand_system.substance = ProtocolPath('substance', 'global')

        schema.protocols[build_solvated_ligand_system.id] = build_solvated_ligand_system.schema

        # Employ YANK to estimate the binding free energy.
        yank_protocol = simulation.LigandReceptorYankProtocol('yank_protocol')

        yank_protocol.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')

        yank_protocol.number_of_iterations = 2000
        yank_protocol.steps_per_iteration = 500
        yank_protocol.checkpoint_interval = 10

        yank_protocol.verbose = True

        yank_protocol.force_field_path = ProtocolPath('force_field_path', 'global')

        yank_protocol.ligand_residue_name = ProtocolPath('ligand_residue_name', perform_docking.id)

        yank_protocol.solvated_ligand_coordinates = ProtocolPath('coordinate_file_path', solvate_ligand.id)
        yank_protocol.solvated_ligand_system = ProtocolPath('system_path', build_solvated_ligand_system.id)

        yank_protocol.solvated_complex_coordinates = ProtocolPath('coordinate_file_path', solvate_complex.id)
        yank_protocol.solvated_complex_system = ProtocolPath('system_path', build_solvated_complex_system.id)

        schema.protocols[yank_protocol.id] = yank_protocol.schema

        # Define where the final values come from.
        schema.final_value_source = ProtocolPath('estimated_free_energy', yank_protocol.id)

        # output_to_store = WorkflowOutputToStore()
        #
        # output_to_store.trajectory_file_path = ProtocolPath('output_trajectory_path',
        #                                                     extract_uncorrelated_trajectory.id)
        # output_to_store.coordinate_file_path = ProtocolPath('output_coordinate_file',
        #                                                     converge_uncertainty.id, npt_production.id)
        #
        # output_to_store.statistics_file_path = ProtocolPath('output_statistics_path',
        #                                                     extract_uncorrelated_statistics.id)
        #
        # output_to_store.statistical_inefficiency = ProtocolPath('statistical_inefficiency', converge_uncertainty.id,
        #                                                                                     extract_density.id)
        #
        # schema.outputs_to_store = {'full_system': output_to_store}

        return schema
