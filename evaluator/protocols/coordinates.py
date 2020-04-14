"""
A collection of protocols for building coordinates for molecular systems.
"""
import logging
from collections import defaultdict
from enum import Enum
from os import path

import numpy as np
import pint
from simtk.openmm import app

from evaluator import unit
from evaluator.attributes import UNDEFINED
from evaluator.substances import Component, ExactAmount, MoleFraction, Substance
from evaluator.utils import packmol
from evaluator.workflow.attributes import InputAttribute, OutputAttribute
from evaluator.workflow.plugins import workflow_protocol
from evaluator.workflow.protocols import Protocol

logger = logging.getLogger(__name__)


@workflow_protocol()
class BuildCoordinatesPackmol(Protocol):
    """Creates a set of 3D coordinates with a specified composition
    using the PACKMOL package.
    """

    max_molecules = InputAttribute(
        docstring="The maximum number of molecules to be added to the system.",
        type_hint=int,
        default_value=1000,
    )
    mass_density = InputAttribute(
        docstring="The target density of the created system.",
        type_hint=pint.Quantity,
        default_value=0.95 * unit.grams / unit.milliliters,
    )

    box_aspect_ratio = InputAttribute(
        docstring="The aspect ratio of the simulation box.",
        type_hint=list,
        default_value=[1.0, 1.0, 1.0],
    )

    substance = InputAttribute(
        docstring="The composition of the system to build.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )

    verbose_packmol = InputAttribute(
        docstring="If True, packmol will print verbose information to the logger",
        type_hint=bool,
        default_value=False,
    )
    retain_packmol_files = InputAttribute(
        docstring="If True, packmol will not delete all of the temporary files "
        "it creates while building the coordinates.",
        type_hint=bool,
        default_value=False,
    )

    output_number_of_molecules = OutputAttribute(
        docstring="The number of molecules in the created system. This "
        "may be less than maximum requested due to rounding of "
        "mole fractions",
        type_hint=int,
    )
    output_substance = OutputAttribute(
        docstring="The substance which was built by packmol. This may differ "
        "from the input substance for system containing two or "
        "more components due to rounding of mole fractions. The "
        "mole fractions provided by this output should always be "
        "used when weighting values by a mole fraction.",
        type_hint=Substance,
    )

    assigned_residue_names = OutputAttribute(
        docstring="The residue names which were assigned to "
        "each of the components. Each key corresponds to a "
        "component identifier.",
        type_hint=dict,
    )

    coordinate_file_path = OutputAttribute(
        docstring="The file path to the created PDB coordinate file.", type_hint=str
    )

    def _build_molecule_arrays(self):
        """Converts the input substance into a list of molecules and a list
        of counts for how many of each there should be as determined by the
        `max_molecules` input and the substances respective mole fractions.

        Returns
        -------
        list of openforcefield.topology.Molecule
            The list of molecules.
        list of int
            The number of each molecule which should be added to the system.
        """
        from openforcefield.topology import Molecule

        molecules = []

        for component in self.substance.components:

            molecule = Molecule.from_smiles(component.smiles)
            molecules.append(molecule)

        # Determine how many molecules of each type will be present in the system.
        molecules_per_component = self.substance.get_molecules_per_component(
            self.max_molecules
        )
        number_of_molecules = [0] * self.substance.number_of_components

        for index, component in enumerate(self.substance.components):
            number_of_molecules[index] = molecules_per_component[component.identifier]

        if sum(number_of_molecules) > self.max_molecules:

            raise ValueError(
                f"The number of molecules to create ({sum(number_of_molecules)}) is "
                f"greater than the maximum number requested ({self.max_molecules})."
            )

        return molecules, number_of_molecules, None

    def _rebuild_substance(self, number_of_molecules):
        """Rebuilds the `Substance` object which this protocol is building
        coordinates for.

        This may not be the same as the input substance due to the finite
        number of molecules to be added causing rounding of mole fractions.

        Parameters
        ----------
        number_of_molecules: list of int
            The number of each component which should be added to the system.

        Returns
        -------
        Substance
            The substance which contains the corrected component amounts.
        """

        new_amounts = defaultdict(list)

        total_number_of_molecules = sum(number_of_molecules)

        # Handle any exact amounts.
        for component in self.substance.components:

            exact_amounts = [
                amount
                for amount in self.substance.get_amounts(component)
                if isinstance(amount, ExactAmount)
            ]

            if len(exact_amounts) == 0:
                continue

            total_number_of_molecules -= exact_amounts[0].value
            new_amounts[component].append(exact_amounts[0])

        # Recompute the mole fractions.
        total_mole_fraction = 0.0
        number_of_new_mole_fractions = 0

        for index, component in enumerate(self.substance.components):

            mole_fractions = [
                amount
                for amount in self.substance.get_amounts(component)
                if isinstance(amount, MoleFraction)
            ]

            if len(mole_fractions) == 0:
                continue

            molecule_count = number_of_molecules[index]

            if component in new_amounts:
                molecule_count -= new_amounts[component][0].value

            new_mole_fraction = molecule_count / total_number_of_molecules
            new_amounts[component].append(MoleFraction(new_mole_fraction))

            total_mole_fraction += new_mole_fraction
            number_of_new_mole_fractions += 1

        if (
            not np.isclose(total_mole_fraction, 1.0)
            and number_of_new_mole_fractions > 0
        ):
            raise ValueError("The new mole fraction does not equal 1.0")

        output_substance = Substance()

        for component, amounts in new_amounts.items():

            for amount in amounts:
                output_substance.add_component(component, amount)

        return output_substance

    def _save_results(self, directory, trajectory):
        """Save the results of running PACKMOL in the working directory

        Parameters
        ----------
        directory: str
            The directory to save the results in.
        trajectory : mdtraj.Trajectory
            The trajectory of the created system.
        """

        self.coordinate_file_path = path.join(directory, "output.pdb")
        trajectory.save_pdb(self.coordinate_file_path)

    def _execute(self, directory, available_resources):

        molecules, number_of_molecules, exception = self._build_molecule_arrays()

        self.output_number_of_molecules = sum(number_of_molecules)
        self.output_substance = self._rebuild_substance(number_of_molecules)

        packmol_directory = path.join(directory, "packmol_files")

        # Create packed box
        trajectory, residue_names = packmol.pack_box(
            molecules=molecules,
            number_of_copies=number_of_molecules,
            mass_density=self.mass_density,
            box_aspect_ratio=self.box_aspect_ratio,
            verbose=self.verbose_packmol,
            working_directory=packmol_directory,
            retain_working_files=self.retain_packmol_files,
        )

        self.assigned_residue_names = dict()

        for component, residue_name in zip(self.substance, residue_names):
            self.assigned_residue_names[component.identifier] = residue_name

        if trajectory is None:
            raise RuntimeError("Packmol failed to complete.")

        self._save_results(directory, trajectory)


@workflow_protocol()
class SolvateExistingStructure(BuildCoordinatesPackmol):
    """Solvates a set of 3D coordinates with a specified solvent
    using the PACKMOL package.
    """

    solute_coordinate_file = InputAttribute(
        docstring="A file path to the solute to solvate.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    center_solute_in_box = InputAttribute(
        docstring="If `True`, the solute to solvate will be centered in the "
        "simulation box.",
        type_hint=bool,
        default_value=True,
    )

    def _execute(self, directory, available_resources):

        molecules, number_of_molecules, exception = self._build_molecule_arrays()

        packmol_directory = path.join(directory, "packmol_files")

        # Create packed box
        trajectory, residue_names = packmol.pack_box(
            molecules=molecules,
            number_of_copies=number_of_molecules,
            structure_to_solvate=self.solute_coordinate_file,
            center_solute=self.center_solute_in_box,
            mass_density=self.mass_density,
            box_aspect_ratio=self.box_aspect_ratio,
            verbose=self.verbose_packmol,
            working_directory=packmol_directory,
            retain_working_files=self.retain_packmol_files,
        )

        if trajectory is None:
            raise RuntimeError("Packmol failed to complete.")

        self.assigned_residue_names = dict()

        for component, residue_name in zip(self.substance, residue_names):
            self.assigned_residue_names[component.identifier] = residue_name

        self._save_results(directory, trajectory)


@workflow_protocol()
class BuildDockedCoordinates(Protocol):
    """Creates a set of coordinates for a ligand bound to some receptor.

    Notes
    -----
    This protocol currently only supports docking with the OpenEye OEDocking
    framework.
    """

    class ActivateSiteLocation(Enum):
        """An enum which describes the methods by which a receptors
        activate site(s) is located."""

        ReceptorCenterOfMass = "ReceptorCenterOfMass"

    ligand_substance = InputAttribute(
        docstring="A substance containing only the ligand to dock.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )
    number_of_ligand_conformers = InputAttribute(
        docstring="The number of conformers to try and dock into the "
        "receptor structure.",
        type_hint=int,
        default_value=100,
    )

    receptor_coordinate_file = InputAttribute(
        docstring="The file path to the MOL2 coordinates of the receptor molecule.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    activate_site_location = InputAttribute(
        docstring="Defines the method by which the activate site is identified.",
        type_hint=ActivateSiteLocation,
        default_value=ActivateSiteLocation.ReceptorCenterOfMass,
    )

    docked_ligand_coordinate_path = OutputAttribute(
        docstring="The file path to the coordinates of the ligand in "
        "it's docked pose, aligned with the initial "
        "`receptor_coordinate_file`.",
        type_hint=str,
    )
    docked_complex_coordinate_path = OutputAttribute(
        docstring="The file path to the docked ligand-receptor complex.", type_hint=str
    )

    ligand_residue_name = OutputAttribute(
        docstring="The residue name assigned to the docked ligand.", type_hint=str
    )
    receptor_residue_name = OutputAttribute(
        docstring="The residue name assigned to the receptor.", type_hint=str
    )

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self.ligand_residue_name = "LIG"
        self.receptor_residue_name = "REC"

    def _create_receptor(self):
        """Create an OpenEye receptor from a mol2 file.

        Returns
        -------
        openeye.oedocking.OEReceptor
            The OpenEye receptor object.
        """
        from openeye import oechem, oedocking

        input_stream = oechem.oemolistream(self.receptor_coordinate_file)

        original_receptor_molecule = oechem.OEGraphMol()
        oechem.OEReadMolecule(input_stream, original_receptor_molecule)

        center_of_mass = oechem.OEFloatArray(3)
        oechem.OEGetCenterOfMass(original_receptor_molecule, center_of_mass)

        receptor = oechem.OEGraphMol()
        oedocking.OEMakeReceptor(
            receptor,
            original_receptor_molecule,
            center_of_mass[0],
            center_of_mass[1],
            center_of_mass[2],
        )

        return receptor

    def _create_ligand(self):
        """Create an OpenEye receptor from a mol2 file.

        Returns
        -------
        openeye.oechem.OEMol
            The OpenEye ligand object with multiple conformers.
        """
        from openforcefield.topology import Molecule

        ligand = Molecule.from_smiles(self.ligand_substance.components[0].smiles)
        ligand.generate_conformers(n_conformers=self.number_of_ligand_conformers)

        # Assign AM1-BCC charges to the ligand just as an initial guess
        # for docking. In future, we may want to get the charge model
        # directly from the force field.
        ligand.compute_partial_charges_am1bcc()

        return ligand.to_openeye()

    def _execute(self, directory, available_resources):

        import mdtraj
        from openeye import oechem, oedocking
        from simtk import unit as simtk_unit

        if (
            len(self.ligand_substance.components) != 1
            or self.ligand_substance.components[0].role != Component.Role.Ligand
        ):

            raise ValueError(
                "The ligand substance must contain a single ligand component."
            )

        logger.info("Initializing the receptor molecule.")
        receptor_molecule = self._create_receptor()

        logger.info("Initializing the ligand molecule.")
        ligand_molecule = self._create_ligand()

        logger.info("Initializing the docking object.")

        # Dock the ligand to the receptor.
        dock = oedocking.OEDock()
        dock.Initialize(receptor_molecule)

        docked_ligand = oechem.OEGraphMol()

        logger.info("Performing the docking.")

        status = dock.DockMultiConformerMolecule(docked_ligand, ligand_molecule)

        if status != oedocking.OEDockingReturnCode_Success:
            raise RuntimeError("The ligand could not be successfully docked",)

        docking_method = oedocking.OEDockMethodGetName(oedocking.OEDockMethod_Default)
        oedocking.OESetSDScore(docked_ligand, dock, docking_method)

        dock.AnnotatePose(docked_ligand)

        self.docked_ligand_coordinate_path = path.join(directory, "ligand.pdb")

        output_stream = oechem.oemolostream(self.docked_ligand_coordinate_path)
        oechem.OEWriteMolecule(output_stream, docked_ligand)
        output_stream.close()

        receptor_pdb_path = path.join(directory, "receptor.pdb")

        output_stream = oechem.oemolostream(receptor_pdb_path)
        oechem.OEWriteMolecule(output_stream, receptor_molecule)
        output_stream.close()

        ligand_trajectory = mdtraj.load(self.docked_ligand_coordinate_path)

        ligand_residue = ligand_trajectory.topology.residue(0)
        ligand_residue.name = self.ligand_residue_name

        # Save the ligand file with the correct residue name.
        ligand_trajectory.save(self.docked_ligand_coordinate_path)

        receptor_trajectory = mdtraj.load(receptor_pdb_path)

        receptor_residue = receptor_trajectory.topology.residue(0)
        receptor_residue.name = self.receptor_residue_name

        # Create a merged ligand-receptor topology.
        complex_topology = ligand_trajectory.topology.copy()

        atom_mapping = {}

        new_residue = complex_topology.add_residue(
            receptor_residue.name, complex_topology.chain(0)
        )

        for receptor_atom in receptor_residue.atoms:

            new_atom = complex_topology.add_atom(
                receptor_atom.name,
                receptor_atom.element,
                new_residue,
                serial=receptor_atom.serial,
            )

            atom_mapping[receptor_atom] = new_atom

        for bond in receptor_trajectory.topology.bonds:

            complex_topology.add_bond(
                atom_mapping[bond[0]],
                atom_mapping[bond[1]],
                type=bond.type,
                order=bond.order,
            )

        complex_positions = []

        complex_positions.extend(
            ligand_trajectory.openmm_positions(0).value_in_unit(simtk_unit.angstrom)
        )
        complex_positions.extend(
            receptor_trajectory.openmm_positions(0).value_in_unit(simtk_unit.angstrom)
        )

        complex_positions *= simtk_unit.angstrom

        self.docked_complex_coordinate_path = path.join(directory, "complex.pdb")

        with open(self.docked_complex_coordinate_path, "w+") as file:
            app.PDBFile.writeFile(complex_topology.to_openmm(), complex_positions, file)
