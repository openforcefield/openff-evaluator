"""
A collection of density physical property definitions.
"""
import copy

from evaluator import unit
from evaluator.datasets import PhysicalProperty
from evaluator.layers.simulation import SimulationSchema
from evaluator.protocols import coordinates, forcefield, miscellaneous, yank
from evaluator.protocols.paprika import OpenMMPaprikaProtocol
from evaluator.substances import Component
from evaluator.workflow.schemas import ProtocolReplicator, WorkflowSchema
from evaluator.workflow.utils import ProtocolPath, ReplicatorValue


class HostGuestBindingAffinity(PhysicalProperty):
    """A class representation of a host-guest binding affinity property"""

    @classmethod
    def default_unit(cls):
        return unit.kilojoule / unit.mole

    @staticmethod
    def default_simulation_schema(existing_schema=None):
        """Returns the default calculation schema to use when estimating
        this class of property from direct simulations using the YANK
        free energy package.

        Parameters
        ----------
        existing_schema: SimulationSchema, optional
            An existing schema whose settings to use. If set,
            the schema's `workflow_schema` will be overwritten
            by this method.

        Returns
        -------
        SimulationSchema
            The schema to follow when estimating this property.
        """

        calculation_schema = SimulationSchema()

        if existing_schema is not None:

            assert isinstance(existing_schema, SimulationSchema)
            calculation_schema = copy.deepcopy(existing_schema)

        schema = WorkflowSchema(property_type=HostGuestBindingAffinity.__name__)
        schema.id = "{}{}".format(HostGuestBindingAffinity.__name__, "Schema")

        # Initial coordinate and topology setup.
        filter_ligand = miscellaneous.FilterSubstanceByRole("filter_ligand")
        filter_ligand.input_substance = ProtocolPath("substance", "global")

        filter_ligand.component_role = Component.Role.Ligand
        # We only support substances with a single guest ligand.
        filter_ligand.expected_components = 1

        schema.protocols[filter_ligand.id] = filter_ligand.schema

        # Construct the protocols which will (for now) take as input a set of host coordinates,
        # and generate a set of charges for them.
        filter_receptor = miscellaneous.FilterSubstanceByRole("filter_receptor")
        filter_receptor.input_substance = ProtocolPath("substance", "global")

        filter_receptor.component_role = Component.Role.Receptor
        # We only support substances with a single host receptor.
        filter_receptor.expected_components = 1

        schema.protocols[filter_receptor.id] = filter_receptor.schema

        # Perform docking to position the guest within the host.
        perform_docking = coordinates.BuildDockedCoordinates("perform_docking")

        perform_docking.ligand_substance = ProtocolPath(
            "filtered_substance", filter_ligand.id
        )
        perform_docking.receptor_coordinate_file = ProtocolPath(
            "receptor_mol2", "global"
        )

        schema.protocols[perform_docking.id] = perform_docking.schema

        # Solvate the docked structure using packmol
        filter_solvent = miscellaneous.FilterSubstanceByRole("filter_solvent")
        filter_solvent.input_substance = ProtocolPath("substance", "global")
        filter_solvent.component_role = Component.Role.Solvent

        schema.protocols[filter_solvent.id] = filter_solvent.schema

        solvate_complex = coordinates.SolvateExistingStructure("solvate_complex")
        solvate_complex.max_molecules = 1000

        solvate_complex.substance = ProtocolPath(
            "filtered_substance", filter_solvent.id
        )
        solvate_complex.solute_coordinate_file = ProtocolPath(
            "docked_complex_coordinate_path", perform_docking.id
        )

        schema.protocols[solvate_complex.id] = solvate_complex.schema

        # Assign force field parameters to the solvated complex system.
        build_solvated_complex_system = forcefield.BaseBuildSystem(
            "build_solvated_complex_system"
        )

        build_solvated_complex_system.force_field_path = ProtocolPath(
            "force_field_path", "global"
        )

        build_solvated_complex_system.coordinate_file_path = ProtocolPath(
            "coordinate_file_path", solvate_complex.id
        )
        build_solvated_complex_system.substance = ProtocolPath("substance", "global")

        build_solvated_complex_system.charged_molecule_paths = [
            ProtocolPath("receptor_mol2", "global")
        ]

        schema.protocols[
            build_solvated_complex_system.id
        ] = build_solvated_complex_system.schema

        # Solvate the ligand using packmol
        solvate_ligand = coordinates.SolvateExistingStructure("solvate_ligand")
        solvate_ligand.max_molecules = 1000

        solvate_ligand.substance = ProtocolPath("filtered_substance", filter_solvent.id)
        solvate_ligand.solute_coordinate_file = ProtocolPath(
            "docked_ligand_coordinate_path", perform_docking.id
        )

        schema.protocols[solvate_ligand.id] = solvate_ligand.schema

        # Assign force field parameters to the solvated ligand system.
        build_solvated_ligand_system = forcefield.BaseBuildSystem(
            "build_solvated_ligand_system"
        )

        build_solvated_ligand_system.force_field_path = ProtocolPath(
            "force_field_path", "global"
        )

        build_solvated_ligand_system.coordinate_file_path = ProtocolPath(
            "coordinate_file_path", solvate_ligand.id
        )
        build_solvated_ligand_system.substance = ProtocolPath("substance", "global")

        schema.protocols[
            build_solvated_ligand_system.id
        ] = build_solvated_ligand_system.schema

        # Employ YANK to estimate the binding free energy.
        yank_protocol = yank.LigandReceptorYankProtocol("yank_protocol")

        yank_protocol.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )

        yank_protocol.number_of_iterations = 2000
        yank_protocol.steps_per_iteration = 500
        yank_protocol.checkpoint_interval = 10

        yank_protocol.verbose = True

        yank_protocol.force_field_path = ProtocolPath("force_field_path", "global")

        yank_protocol.ligand_residue_name = ProtocolPath(
            "ligand_residue_name", perform_docking.id
        )
        yank_protocol.receptor_residue_name = ProtocolPath(
            "receptor_residue_name", perform_docking.id
        )

        yank_protocol.solvated_ligand_coordinates = ProtocolPath(
            "coordinate_file_path", solvate_ligand.id
        )
        yank_protocol.solvated_ligand_system = ProtocolPath(
            "system_path", build_solvated_ligand_system.id
        )

        yank_protocol.solvated_complex_coordinates = ProtocolPath(
            "coordinate_file_path", solvate_complex.id
        )
        yank_protocol.solvated_complex_system = ProtocolPath(
            "system_path", build_solvated_complex_system.id
        )

        schema.protocols[yank_protocol.id] = yank_protocol.schema

        # Define where the final values come from.
        schema.final_value_source = ProtocolPath(
            "estimated_free_energy", yank_protocol.id
        )

        calculation_schema.workflow_schema = schema
        return calculation_schema

    @staticmethod
    def default_paprika_schema(
        n_solvent_molecules=2000,
        n_equilibration_steps=200000,
        n_production_steps=1000000,
    ):
        """Returns the default workflow to use when estimating this property
        using the attach, pull release (APR) method to predict a binding affinity.

        This makes use of the optional `paprika` package.

        Parameters
        ----------
        n_solvent_molecules: int
            The number of solvent molecules to add to the box.
        n_equilibration_steps: int
            The number of equilibration simulations steps to perform.
            Sample generated during this step will be discarded.
        n_production_steps: int
            The number of production simulations steps to perform.
            Sample generated during this step will be used in the final
            free energy calculation.

        Returns
        -------
        SimulationSchema
            The schema to follow when estimating this property.
        """

        # Set up a replicator which will perform the attach-pull calculation for
        # each of the guest orientations
        orientation_replicator = ProtocolReplicator("orientation_replicator")
        orientation_replicator.template_values = ProtocolPath(
            "guest_orientations", "global"
        )

        # Create the protocol which will run the attach pull calculations
        host_guest_protocol = OpenMMPaprikaProtocol(
            f"host_guest_free_energy_{orientation_replicator.placeholder_id}"
        )

        host_guest_protocol.substance = ProtocolPath("substance", "global")
        host_guest_protocol.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )
        host_guest_protocol.force_field_path = ProtocolPath(
            "force_field_path", "global"
        )
        host_guest_protocol.taproom_host_name = ProtocolPath(
            "host_identifier", "global"
        )
        host_guest_protocol.taproom_guest_name = ProtocolPath(
            "guest_identifier", "global"
        )
        host_guest_protocol.taproom_guest_orientation = ReplicatorValue(
            orientation_replicator.id
        )

        host_guest_protocol.number_of_equilibration_steps = n_equilibration_steps
        host_guest_protocol.number_of_production_steps = n_production_steps
        host_guest_protocol.equilibration_output_frequency = 5000
        host_guest_protocol.production_output_frequency = 5000
        host_guest_protocol.number_of_solvent_molecules = n_solvent_molecules

        # Retrieve a subset of the full substance which only contains the
        # host and the solvent.
        filter_host = miscellaneous.FilterSubstanceByRole("filter_host")
        filter_host.input_substance = ProtocolPath("substance", "global")

        filter_host.component_roles = [
            Component.Role.Solute,
            Component.Role.Solvent,
            Component.Role.Receptor,
        ]

        # Create the protocols which will run the release calculations
        host_protocol = OpenMMPaprikaProtocol("host")

        host_protocol.substance = ProtocolPath("filtered_substance", filter_host.id)
        host_protocol.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )
        host_protocol.force_field_path = ProtocolPath("force_field_path", "global")
        host_protocol.taproom_host_name = ProtocolPath("host_identifier", "global")

        host_protocol.number_of_equilibration_steps = n_equilibration_steps
        host_protocol.number_of_production_steps = n_production_steps
        host_protocol.equilibration_output_frequency = 5000
        host_protocol.production_output_frequency = 5000
        host_protocol.number_of_solvent_molecules = n_solvent_molecules

        # Sum together the free energies of the individual orientations
        sum_protocol = miscellaneous.AddValues(
            f"add_per_orientation_free_energies_"
            f"{orientation_replicator.placeholder_id}"
        )
        sum_protocol.values = [
            ProtocolPath("attach_free_energy", host_guest_protocol.id),
            ProtocolPath("pull_free_energy", host_guest_protocol.id),
            ProtocolPath("reference_free_energy", host_guest_protocol.id),
            ProtocolPath("release_free_energy", host_protocol.id),
        ]

        # Finally, combine all of the values together
        combine_values = miscellaneous.AddBindingFreeEnergies("combine_values")
        combine_values.values = ProtocolPath("result", sum_protocol.id)
        combine_values.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )

        schema = WorkflowSchema()

        schema.protocol_schemas = {
            host_guest_protocol.schema,
            filter_host.schema,
            host_protocol.schema,
            sum_protocol.schema,
            combine_values.schema,
        }

        # Define where the final values come from.
        schema.final_value_source = ProtocolPath("result", combine_values.id)
        schema.replicators = [orientation_replicator]

        return schema
