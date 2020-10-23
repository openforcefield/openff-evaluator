"""
A collection of density physical property definitions.
"""
import copy
from typing import Dict, Tuple

from openff.evaluator import unit
from openff.evaluator.datasets import PhysicalProperty
from openff.evaluator.layers import register_calculation_schema
from openff.evaluator.layers.simulation import SimulationLayer, SimulationSchema
from openff.evaluator.protocols import (
    analysis,
    coordinates,
    forcefield,
    miscellaneous,
    openmm,
    yank,
)
from openff.evaluator.protocols.paprika.analysis import (
    AnalyzeAPRPhase,
    ComputeReferenceWork,
    ComputeSymmetryCorrection,
)
from openff.evaluator.protocols.paprika.coordinates import (
    AddDummyAtoms,
    PreparePullCoordinates,
    PrepareReleaseCoordinates,
)
from openff.evaluator.protocols.paprika.restraints import (
    ApplyRestraints,
    GenerateAttachRestraints,
    GeneratePullRestraints,
    GenerateReleaseRestraints,
)
from openff.evaluator.substances import Component
from openff.evaluator.thermodynamics import Ensemble
from openff.evaluator.workflow.schemas import ProtocolReplicator, WorkflowSchema
from openff.evaluator.workflow.utils import ProtocolPath, ReplicatorValue


class HostGuestBindingAffinity(PhysicalProperty):
    """A class representation of a host-guest binding affinity property"""

    @classmethod
    def default_unit(cls):
        return unit.kilojoule / unit.mole

    @staticmethod
    def default_yank_schema(existing_schema=None):
        """Returns the default calculation schema to use when estimating
        this class of property from direct simulations.

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

        filter_ligand.component_roles = [Component.Role.Ligand]
        # We only support substances with a single guest ligand.
        filter_ligand.expected_components = 1

        schema.protocols[filter_ligand.id] = filter_ligand.schema

        # Construct the protocols which will (for now) take as input a set of host coordinates,
        # and generate a set of charges for them.
        filter_receptor = miscellaneous.FilterSubstanceByRole("filter_receptor")
        filter_receptor.input_substance = ProtocolPath("substance", "global")

        filter_receptor.component_roles = [Component.Role.Receptor]
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
        filter_solvent.component_roles = [Component.Role.Solvent]

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
            "parameterized_system", build_solvated_ligand_system.id
        )

        yank_protocol.solvated_complex_coordinates = ProtocolPath(
            "coordinate_file_path", solvate_complex.id
        )
        yank_protocol.solvated_complex_system = ProtocolPath(
            "parameterized_system", build_solvated_complex_system.id
        )

        schema.protocols[yank_protocol.id] = yank_protocol.schema

        # Define where the final values come from.
        schema.final_value_source = ProtocolPath(
            "free_energy_difference", yank_protocol.id
        )

        calculation_schema.workflow_schema = schema
        return calculation_schema

    @classmethod
    def _paprika_default_solvation_protocol(
        cls, n_solvent_molecules: int
    ) -> coordinates.SolvateExistingStructure:
        """Returns the default protocol to use for solvating each window
        of an APR calculation.
        """

        solvation_protocol = coordinates.SolvateExistingStructure("")
        solvation_protocol.max_molecules = n_solvent_molecules
        solvation_protocol.count_exact_amount = False
        solvation_protocol.box_aspect_ratio = [1.0, 1.0, 2.0]
        solvation_protocol.center_solute_in_box = False
        return solvation_protocol

    @classmethod
    def _paprika_default_simulation_protocols(
        cls,
        n_thermalization_steps: int,
        n_equilibration_steps: int,
        n_production_steps: int,
        dt_thermalization: unit.Quantity,
        dt_equilibration: unit.Quantity,
        dt_production: unit.Quantity,
    ) -> Tuple[
        openmm.OpenMMEnergyMinimisation,
        openmm.OpenMMSimulation,
        openmm.OpenMMSimulation,
        openmm.OpenMMSimulation,
    ]:
        """Returns the default set of simulation protocols to use for each window
        of an APR calculation.

        Parameters
        ----------
        n_thermalization_steps
            The number of thermalization simulations steps to perform.
            Sample generated during this step will be discarded.
        n_equilibration_steps
            The number of equilibration simulations steps to perform.
            Sample generated during this step will be discarded.
        n_production_steps
            The number of production simulations steps to perform.
            Sample generated during this step will be used in the final
            free energy calculation.
        dt_thermalization
            The integration timestep during thermalization
        dt_equilibration
            The integration timestep during equilibration
        dt_production
            The integration timestep during production

        Returns
        -------
            A protocol to perform an energy minimization, a thermalization,
            an equilibration, and finally a production simulation.
        """
        energy_minimisation = openmm.OpenMMEnergyMinimisation("")

        thermalization = openmm.OpenMMSimulation("")
        thermalization.steps_per_iteration = n_thermalization_steps
        thermalization.output_frequency = 10000
        thermalization.timestep = dt_thermalization

        equilibration = openmm.OpenMMSimulation("")
        equilibration.steps_per_iteration = n_equilibration_steps
        equilibration.output_frequency = 10000
        equilibration.timestep = dt_equilibration

        production = openmm.OpenMMSimulation("")
        production.steps_per_iteration = n_production_steps
        production.output_frequency = 5000
        production.timestep = dt_production

        return energy_minimisation, thermalization, equilibration, production

    @classmethod
    def _paprika_build_simulation_protocols(
        cls,
        coordinate_path: ProtocolPath,
        parameterized_system: ProtocolPath,
        id_prefix: str,
        id_suffix: str,
        minimization_template: openmm.OpenMMEnergyMinimisation,
        thermalization_template: openmm.OpenMMSimulation,
        equilibration_template: openmm.OpenMMSimulation,
        production_template: openmm.OpenMMSimulation,
    ) -> Tuple[
        openmm.OpenMMEnergyMinimisation,
        openmm.OpenMMSimulation,
        openmm.OpenMMSimulation,
        openmm.OpenMMSimulation,
    ]:

        minimization = copy.deepcopy(minimization_template)
        minimization.id = f"{id_prefix}_energy_minimization_{id_suffix}"
        minimization.input_coordinate_file = coordinate_path
        minimization.parameterized_system = parameterized_system

        thermalization = copy.deepcopy(thermalization_template)
        thermalization.id = f"{id_prefix}_thermalization_{id_suffix}"
        thermalization.input_coordinate_file = ProtocolPath(
            "output_coordinate_file", minimization.id
        )
        thermalization.parameterized_system = parameterized_system
        thermalization.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )

        equilibration = copy.deepcopy(equilibration_template)
        equilibration.id = f"{id_prefix}_equilibration_{id_suffix}"
        equilibration.input_coordinate_file = ProtocolPath(
            "output_coordinate_file", thermalization.id
        )
        equilibration.parameterized_system = parameterized_system
        equilibration.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )

        production = copy.deepcopy(production_template)
        production.id = f"{id_prefix}_production_{id_suffix}"
        production.input_coordinate_file = ProtocolPath(
            "output_coordinate_file", equilibration.id
        )
        production.parameterized_system = parameterized_system
        production.thermodynamic_state = ProtocolPath("thermodynamic_state", "global")

        return minimization, thermalization, equilibration, production

    @classmethod
    def _paprika_build_attach_pull_protocols(
        cls,
        orientation_replicator: ProtocolReplicator,
        restraint_schemas: Dict[str, ProtocolPath],
        solvation_template: coordinates.SolvateExistingStructure,
        minimization_template: openmm.OpenMMEnergyMinimisation,
        thermalization_template: openmm.OpenMMSimulation,
        equilibration_template: openmm.OpenMMSimulation,
        production_template: openmm.OpenMMSimulation,
    ):

        # Define a replicator to set and solvate up the coordinates for each pull window
        orientation_placeholder = orientation_replicator.placeholder_id

        pull_replicator = ProtocolReplicator(
            f"pull_replicator_{orientation_placeholder}"
        )
        pull_replicator.template_values = ProtocolPath("pull_windows_indices", "global")
        pull_replicator_id = (
            f"{pull_replicator.placeholder_id}_" f"{orientation_placeholder}"
        )

        attach_replicator = ProtocolReplicator(
            f"attach_replicator_{orientation_placeholder}"
        )
        attach_replicator.template_values = ProtocolPath(
            "attach_windows_indices", "global"
        )
        attach_replicator_id = (
            f"{attach_replicator.placeholder_id}_" f"{orientation_placeholder}"
        )

        # Filter out only the solvent substance to help with the solvation step.
        filter_solvent = miscellaneous.FilterSubstanceByRole(
            "host-guest-filter_solvent"
        )
        filter_solvent.input_substance = ProtocolPath("substance", "global")
        filter_solvent.component_roles = [Component.Role.Solvent]

        # Define the protocols which will set and solvate up the coordinates for each
        # pull window
        align_coordinates = PreparePullCoordinates(
            f"pull_align_coordinates_{pull_replicator_id}"
        )
        align_coordinates.substance = ProtocolPath("substance", "global")
        align_coordinates.complex_file_path = ProtocolPath(
            f"guest_orientations[{orientation_placeholder}].coordinate_path", "global"
        )
        align_coordinates.guest_orientation_mask = ProtocolPath(
            "guest_orientation_mask", "global"
        )
        align_coordinates.pull_window_index = ReplicatorValue(pull_replicator.id)
        align_coordinates.pull_distance = ProtocolPath("pull_distance", "global")
        align_coordinates.n_pull_windows = ProtocolPath("n_pull_windows", "global")

        solvate_coordinates = copy.deepcopy(solvation_template)
        solvate_coordinates.id = f"pull_solvate_coordinates_{pull_replicator_id}"
        solvate_coordinates.substance = ProtocolPath(
            "filtered_substance", filter_solvent.id
        )
        solvate_coordinates.solute_coordinate_file = ProtocolPath(
            "output_coordinate_path", align_coordinates.id
        )

        # Apply the force field parameters. This only needs to be done once.
        apply_parameters = forcefield.BuildSmirnoffSystem(
            f"pull_apply_parameters_{orientation_placeholder}"
        )
        apply_parameters.force_field_path = ProtocolPath("force_field_path", "global")
        apply_parameters.substance = ProtocolPath("substance", "global")
        apply_parameters.coordinate_file_path = ProtocolPath(
            "coordinate_file_path",
            f"pull_solvate_coordinates_0_{orientation_placeholder}",
        )

        # Add the dummy atoms.
        add_dummy_atoms = AddDummyAtoms(f"pull_add_dummy_atoms_{pull_replicator_id}")
        add_dummy_atoms.substance = ProtocolPath("substance", "global")
        add_dummy_atoms.input_coordinate_path = ProtocolPath(
            "coordinate_file_path", solvate_coordinates.id
        )
        add_dummy_atoms.input_system = ProtocolPath(
            "parameterized_system", apply_parameters.id
        )
        add_dummy_atoms.offset = ProtocolPath("dummy_atom_offset", "global")

        attach_coordinate_path = ProtocolPath(
            "output_coordinate_path",
            f"pull_add_dummy_atoms_0_{orientation_placeholder}",
        )
        attach_system = ProtocolPath(
            "output_system", f"pull_add_dummy_atoms_0_{orientation_placeholder}"
        )

        # Apply the attach restraints
        generate_attach_restraints = GenerateAttachRestraints(
            f"attach_generate_restraints_{orientation_placeholder}"
        )
        generate_attach_restraints.complex_coordinate_path = attach_coordinate_path
        generate_attach_restraints.attach_lambdas = ProtocolPath(
            "attach_lambdas", "global"
        )
        generate_attach_restraints.restraint_schemas = restraint_schemas

        apply_attach_restraints = ApplyRestraints(
            f"attach_apply_restraints_{attach_replicator_id}"
        )
        apply_attach_restraints.restraints_path = ProtocolPath(
            "restraints_path", generate_attach_restraints.id
        )
        apply_attach_restraints.phase = "attach"
        apply_attach_restraints.window_index = ReplicatorValue(attach_replicator.id)
        apply_attach_restraints.input_system = attach_system

        # Apply the pull restraints
        generate_pull_restraints = GeneratePullRestraints(
            f"pull_generate_restraints_{orientation_placeholder}"
        )
        generate_pull_restraints.complex_coordinate_path = attach_coordinate_path
        generate_pull_restraints.attach_lambdas = ProtocolPath(
            "attach_lambdas", "global"
        )
        generate_pull_restraints.n_pull_windows = ProtocolPath(
            "n_pull_windows", "global"
        )
        generate_pull_restraints.restraint_schemas = restraint_schemas

        apply_pull_restraints = ApplyRestraints(
            f"pull_apply_restraints_{pull_replicator_id}"
        )
        apply_pull_restraints.restraints_path = ProtocolPath(
            "restraints_path", generate_pull_restraints.id
        )
        apply_pull_restraints.phase = "pull"
        apply_pull_restraints.window_index = ReplicatorValue(pull_replicator.id)
        apply_pull_restraints.input_system = ProtocolPath(
            "output_system", add_dummy_atoms.id
        )

        # Setup the simulations for the attach and pull phases.
        (
            attach_minimization,
            attach_thermalization,
            attach_equilibration,
            attach_production,
        ) = cls._paprika_build_simulation_protocols(
            attach_coordinate_path,
            ProtocolPath("output_system", apply_attach_restraints.id),
            "attach",
            attach_replicator_id,
            minimization_template,
            thermalization_template,
            equilibration_template,
            production_template,
        )

        (
            pull_minimization,
            pull_thermalization,
            pull_equilibration,
            pull_production,
        ) = cls._paprika_build_simulation_protocols(
            ProtocolPath("output_coordinate_path", add_dummy_atoms.id),
            ProtocolPath("output_system", apply_pull_restraints.id),
            "pull",
            pull_replicator_id,
            minimization_template,
            thermalization_template,
            equilibration_template,
            production_template,
        )

        # Analyze the attach phase.
        attach_free_energy = AnalyzeAPRPhase(
            f"analyze_attach_phase_{orientation_placeholder}"
        )
        attach_free_energy.topology_path = attach_coordinate_path
        attach_free_energy.trajectory_paths = ProtocolPath(
            "trajectory_file_path", attach_production.id
        )
        attach_free_energy.phase = "attach"
        attach_free_energy.restraints_path = ProtocolPath(
            "restraints_path", generate_attach_restraints.id
        )

        # Analyze the pull phase.
        pull_free_energy = AnalyzeAPRPhase(
            f"analyze_pull_phase_{orientation_placeholder}"
        )
        pull_free_energy.topology_path = attach_coordinate_path
        pull_free_energy.trajectory_paths = ProtocolPath(
            "trajectory_file_path", pull_production.id
        )
        pull_free_energy.phase = "pull"
        pull_free_energy.restraints_path = ProtocolPath(
            "restraints_path", generate_pull_restraints.id
        )

        reference_state_work = ComputeReferenceWork(
            f"pull_reference_work_{orientation_placeholder}"
        )
        reference_state_work.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )
        reference_state_work.restraints_path = ProtocolPath(
            "restraints_path", generate_pull_restraints.id
        )

        # Return the full list of protocols which make up the attach and pull parts
        # of a host-guest APR calculation.
        protocols = [
            filter_solvent,
            align_coordinates,
            solvate_coordinates,
            apply_parameters,
            add_dummy_atoms,
            generate_attach_restraints,
            apply_attach_restraints,
            generate_pull_restraints,
            apply_pull_restraints,
            attach_minimization,
            attach_thermalization,
            attach_equilibration,
            attach_production,
            pull_minimization,
            pull_thermalization,
            pull_equilibration,
            pull_production,
            attach_free_energy,
            pull_free_energy,
            reference_state_work,
        ]
        protocol_replicators = [pull_replicator, attach_replicator]

        return (
            protocols,
            protocol_replicators,
            ProtocolPath("result", attach_free_energy.id),
            ProtocolPath("result", pull_free_energy.id),
            ProtocolPath("result", reference_state_work.id),
        )

    @classmethod
    def _paprika_build_release_protocols(
        cls,
        orientation_replicator: ProtocolReplicator,
        restraint_schemas: Dict[str, ProtocolPath],
        solvation_template: coordinates.SolvateExistingStructure,
        minimization_template: openmm.OpenMMEnergyMinimisation,
        thermalization_template: openmm.OpenMMSimulation,
        equilibration_template: openmm.OpenMMSimulation,
        production_template: openmm.OpenMMSimulation,
    ):

        # Define a replicator to set up each release window
        release_replicator = ProtocolReplicator("release_replicator")
        release_replicator.template_values = ProtocolPath(
            "release_windows_indices", "global"
        )

        orientation_placeholder = orientation_replicator.placeholder_id

        release_replicator_id = (
            f"{release_replicator.placeholder_id}_" f"{orientation_placeholder}"
        )

        # Filter out only the solvent substance to help with the solvation step.
        filter_solvent = miscellaneous.FilterSubstanceByRole("host-filter_solvent")
        filter_solvent.input_substance = ProtocolPath("host_substance", "global")
        filter_solvent.component_roles = [Component.Role.Solvent]

        # Construct a set of coordinates for a host molecule correctly
        # aligned to the z-axis.
        align_coordinates = PrepareReleaseCoordinates("release_align_coordinates")
        align_coordinates.substance = ProtocolPath("host_substance", "global")
        align_coordinates.complex_file_path = ProtocolPath(
            "host_coordinate_path", "global"
        )

        solvate_coordinates = copy.deepcopy(solvation_template)
        solvate_coordinates.id = "release_solvate_coordinates"
        solvate_coordinates.substance = ProtocolPath(
            "filtered_substance", filter_solvent.id
        )
        solvate_coordinates.solute_coordinate_file = ProtocolPath(
            "output_coordinate_path", align_coordinates.id
        )

        # Apply the force field parameters. This only needs to be done for one
        # of the windows.
        apply_parameters = forcefield.BaseBuildSystem("release_apply_parameters")
        apply_parameters.force_field_path = ProtocolPath("force_field_path", "global")
        apply_parameters.substance = ProtocolPath("host_substance", "global")
        apply_parameters.coordinate_file_path = ProtocolPath(
            "coordinate_file_path", solvate_coordinates.id
        )

        # Add the dummy atoms.
        add_dummy_atoms = AddDummyAtoms("release_add_dummy_atoms")
        add_dummy_atoms.substance = ProtocolPath("host_substance", "global")
        add_dummy_atoms.input_coordinate_path = ProtocolPath(
            "coordinate_file_path",
            solvate_coordinates.id,
        )
        add_dummy_atoms.input_system = ProtocolPath(
            "parameterized_system", apply_parameters.id
        )
        add_dummy_atoms.offset = ProtocolPath("dummy_atom_offset", "global")

        # Apply the restraints files
        generate_restraints = GenerateReleaseRestraints(
            f"release_generate_restraints_{orientation_placeholder}"
        )
        generate_restraints.host_coordinate_path = ProtocolPath(
            "output_coordinate_path", add_dummy_atoms.id
        )
        generate_restraints.release_lambdas = ProtocolPath("release_lambdas", "global")
        generate_restraints.restraint_schemas = restraint_schemas

        apply_restraints = ApplyRestraints(
            f"release_apply_restraints_{release_replicator_id}"
        )
        apply_restraints.restraints_path = ProtocolPath(
            "restraints_path", generate_restraints.id
        )
        apply_restraints.phase = "release"
        apply_restraints.window_index = ReplicatorValue(release_replicator.id)
        apply_restraints.input_system = ProtocolPath(
            "output_system", add_dummy_atoms.id
        )

        # Setup the simulations for the release phase.
        (
            release_minimization,
            release_thermalization,
            release_equilibration,
            release_production,
        ) = cls._paprika_build_simulation_protocols(
            ProtocolPath("output_coordinate_path", add_dummy_atoms.id),
            ProtocolPath("output_system", apply_restraints.id),
            "release",
            release_replicator_id,
            minimization_template,
            thermalization_template,
            equilibration_template,
            production_template,
        )

        # Analyze the release phase.
        analyze_release_phase = AnalyzeAPRPhase(
            f"analyze_release_phase_{orientation_placeholder}"
        )
        analyze_release_phase.topology_path = ProtocolPath(
            "output_coordinate_path", add_dummy_atoms.id
        )
        analyze_release_phase.trajectory_paths = ProtocolPath(
            "trajectory_file_path", release_production.id
        )
        analyze_release_phase.phase = "release"
        analyze_release_phase.restraints_path = ProtocolPath(
            "restraints_path", generate_restraints.id
        )

        # Return the full list of protocols which make up the release parts
        # of a host-guest APR calculation.
        protocols = [
            filter_solvent,
            align_coordinates,
            solvate_coordinates,
            apply_parameters,
            add_dummy_atoms,
            generate_restraints,
            apply_restraints,
            release_minimization,
            release_thermalization,
            release_equilibration,
            release_production,
            analyze_release_phase,
        ]

        return (
            protocols,
            release_replicator,
            ProtocolPath("result", analyze_release_phase.id),
        )

    @classmethod
    def default_paprika_schema(
        cls,
        existing_schema: SimulationSchema = None,
        n_solvent_molecules: int = 2500,
        n_thermalization_steps: int = 50000,
        n_equilibration_steps: int = 200000,
        n_production_steps: int = 2500000,
        dt_thermalization: unit.Quantity = 1.0 * unit.femtosecond,
        dt_equilibration: unit.Quantity = 2.0 * unit.femtosecond,
        dt_production: unit.Quantity = 2.0 * unit.femtosecond,
        debug: bool = False,
    ):
        """Returns the default calculation schema to use when estimating
        a host-guest binding affinity measurement with an APR calculation
        using the ``paprika`` package.

        Notes
        -----
        * This schema requires additional metadata to be able to estimate
          each metadata. This metadata is automatically generated for properties
          loaded from the ``taproom`` package using the ``TaproomDataSet`` object.

        Parameters
        ----------
        existing_schema: SimulationSchema, optional
            An existing schema whose settings to use. If set,
            the schema's `workflow_schema` will be overwritten
            by this method.
        n_solvent_molecules
            The number of solvent molecules to add to the box.
        n_thermalization_steps
            The number of thermalization simulations steps to perform.
            Sample generated during this step will be discarded.
        n_equilibration_steps
            The number of equilibration simulations steps to perform.
            Sample generated during this step will be discarded.
        n_production_steps
            The number of production simulations steps to perform.
            Sample generated during this step will be used in the final
            free energy calculation.
        dt_thermalization
            The integration timestep during thermalization
        dt_equilibration
            The integration timestep during equilibration
        dt_production
            The integration timestep during production
        debug
            Whether to return a debug schema. This is nearly identical
            to the default schema, albeit with significantly less
            solvent molecules (10), all simulations run in NVT and much
            shorter simulation runs (500 steps). If True, the other input
            arguments will be ignored.

        Returns
        -------
        SimulationSchema
            The schema to follow when estimating this property.
        """

        calculation_schema = SimulationSchema()

        if existing_schema is not None:
            assert isinstance(existing_schema, SimulationSchema)
            calculation_schema = copy.deepcopy(existing_schema)

        # Initialize the protocols which will serve as templates for those
        # used in the actual workflows.
        solvation_template = cls._paprika_default_solvation_protocol(
            n_solvent_molecules=n_solvent_molecules
        )

        (
            minimization_template,
            *simulation_templates,
        ) = cls._paprika_default_simulation_protocols(
            n_thermalization_steps=n_thermalization_steps,
            n_equilibration_steps=n_equilibration_steps,
            n_production_steps=n_production_steps,
            dt_thermalization=dt_thermalization,
            dt_equilibration=dt_equilibration,
            dt_production=dt_production,
        )

        if debug:

            solvation_template.max_molecules = 10
            solvation_template.mass_density = 0.01 * unit.grams / unit.milliliters

            for simulation_template in simulation_templates:

                simulation_template.ensemble = Ensemble.NVT
                simulation_template.steps_per_iteration = 500
                simulation_template.output_frequency = 50

        # Set up a replicator which will perform the attach-pull calculation for
        # each of the guest orientations
        orientation_replicator = ProtocolReplicator("orientation_replicator")
        orientation_replicator.template_values = ProtocolPath(
            "guest_orientations", "global"
        )

        restraint_schemas = {
            "static": ProtocolPath(
                f"guest_orientations[{orientation_replicator.placeholder_id}]."
                f"static_restraints",
                "global",
            ),
            "conformational": ProtocolPath(
                f"guest_orientations[{orientation_replicator.placeholder_id}]."
                f"conformational_restraints",
                "global",
            ),
            "guest": ProtocolPath("guest_restraints", "global"),
            "wall": ProtocolPath("wall_restraints", "global"),
            "symmetry": ProtocolPath("symmetry_restraints", "global"),
        }

        # Build the protocols to compute the attach and pull free energies.
        (
            attach_pull_protocols,
            attach_pull_replicators,
            attach_free_energy,
            pull_free_energy,
            reference_work,
        ) = cls._paprika_build_attach_pull_protocols(
            orientation_replicator,
            restraint_schemas,
            solvation_template,
            minimization_template,
            *simulation_templates,
        )

        # Build the protocols to compute the release free energies.
        (
            release_protocols,
            release_replicator,
            release_free_energy,
        ) = cls._paprika_build_release_protocols(
            orientation_replicator,
            restraint_schemas,
            solvation_template,
            minimization_template,
            *simulation_templates,
        )

        # Compute the symmetry correction.
        symmetry_correction = ComputeSymmetryCorrection("symmetry_correction")
        symmetry_correction.n_microstates = ProtocolPath(
            "n_guest_microstates", "global"
        )
        symmetry_correction.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )

        # Sum together the free energies of the individual orientations
        orientation_free_energy = miscellaneous.AddValues(
            f"orientation_free_energy_{orientation_replicator.placeholder_id}"
        )
        orientation_free_energy.values = [
            attach_free_energy,
            pull_free_energy,
            reference_work,
            release_free_energy,
            ProtocolPath("result", symmetry_correction.id),
        ]

        # Finally, combine all of the values together
        total_free_energy = analysis.AverageFreeEnergies("total_free_energy")
        total_free_energy.values = ProtocolPath("result", orientation_free_energy.id)
        total_free_energy.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )

        calculation_schema.workflow_schema = WorkflowSchema()

        calculation_schema.workflow_schema.protocol_schemas = [
            *(protocol.schema for protocol in attach_pull_protocols),
            *(protocol.schema for protocol in release_protocols),
            symmetry_correction.schema,
            orientation_free_energy.schema,
            total_free_energy.schema,
        ]
        calculation_schema.workflow_schema.protocol_replicators = [
            orientation_replicator,
            *attach_pull_replicators,
            release_replicator,
        ]

        # Define where the final value comes from.
        calculation_schema.workflow_schema.final_value_source = ProtocolPath(
            "result", total_free_energy.id
        )

        return calculation_schema


register_calculation_schema(
    HostGuestBindingAffinity,
    SimulationLayer,
    HostGuestBindingAffinity.default_paprika_schema,
)
