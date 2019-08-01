"""
A collection of protocols for building coordinates for molecular systems.
"""

import logging
from enum import Enum
from os import path

from simtk import unit
from simtk.openmm import app

from propertyestimator.substances import Substance
from propertyestimator.utils import packmol, create_molecule_from_smiles
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class BuildCoordinatesPackmol(BaseProtocol):
    """Creates a set of 3D coordinates with a specified composition.

    Notes
    -----
    The coordinates are created using packmol.
    """

    @protocol_input(int)
    def max_molecules(self):
        """The maximum number of molecules to be added to the system."""
        pass

    @protocol_input(unit.Quantity)
    def mass_density(self):
        """The target density of the created system."""
        pass

    @protocol_input(list)
    def box_aspect_ratio(self):
        """The aspect ratio of the simulation box. The default is [1.0, 1.0, 1.0],
        i.e a cubic box."""
        return self._box_aspect_ratio

    @protocol_input(Substance)
    def substance(self):
        """The composition of the system to build."""
        pass

    @protocol_input(bool)
    def verbose_packmol(self):
        """If True, packmol will be allowed to log verbose information to the logger,
        and any working packmol files will be retained."""
        pass

    @protocol_input(bool)
    def retain_packmol_files(self):
        """If True, packmol will not delete all of the temporary files it creates
        while building the coordinates."""
        pass

    @protocol_output(int)
    def final_number_of_molecules(self):
        """The file path to the created PDB coordinate file.
        TODO: This is a temporary addition until inputs are made
              available as outputs by default.
        """
        pass

    @protocol_input(str)
    def change_chains(self):
        """ If a string of the format `chain B`, this will set the chain of the file to be solvated."""
        pass

    @protocol_output(str)
    def coordinate_file_path(self):
        """The file path to the created PDB coordinate file."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._substance = None

        self._coordinate_file_path = None
        self._positions = None

        self._max_molecules = 1000
        self._mass_density = 0.95 * unit.grams / unit.milliliters

        self._verbose_packmol = False
        self._retain_packmol_files = False

        self._box_aspect_ratio = [1.0, 1.0, 1.0]

        self._final_number_of_molecules = None
        self._change_chains = "chain B"

    def _build_molecule_arrays(self, directory):
        """Converts the input substance into a list of openeye OEMol's and a list of
        counts for how many of each there should be as determined by the `max_molecules`
        input and the molecules respective mole fractions.

        Parameters
        ----------
        directory: The directory in which this protocols working files are being saved.

        Returns
        -------
        list of openeye.oechem.OEMol
            The list of openeye molecules.
        list of int
            The number of each molecule which should be added to the system.
        PropertyEstimatorException, optional
            None if no exceptions occured, otherwise the exception.
        """

        molecules = []

        for component in self._substance.components:

            molecule = create_molecule_from_smiles(component.smiles)

            if molecule is None:

                return None, None, PropertyEstimatorException(directory=directory,
                                                              message=f'{component} could not be converted '
                                                                      f'to a Molecule')

            molecules.append(molecule)

        # Determine how many molecules of each type will be present in the system.
        molecules_per_component = self._substance.get_molecules_per_component(self._max_molecules)
        number_of_molecules = [0] * self._substance.number_of_components

        for index, component in enumerate(self._substance.components):
            number_of_molecules[index] = molecules_per_component[component.identifier]

        return molecules, number_of_molecules, None

    def _save_results(self, directory, topology, positions):
        """Save the results of running PACKMOL in the working directory

        Parameters
        ----------
        directory: str
            The directory to save the results in.
        topology : simtk.openmm.Topology
            The topology of the created system.
        positions : simtk.unit.Quantity
            A `simtk.unit.Quantity` wrapped `numpy.ndarray` (shape=[natoms,3]) which contains
            the created positions with units compatible with angstroms.
        """

        self._coordinate_file_path = path.join(directory, 'output.pdb')

        with open(self._coordinate_file_path, 'w+') as minimised_file:
            # noinspection PyTypeChecker
            app.PDBFile.writeFile(topology, positions, minimised_file)

        logging.info('Coordinates generated: ' + self._substance.identifier)

    def execute(self, directory, available_resources):

        logging.info(f'Generating coordinates for {self._substance.identifier}: {self.id}')

        if self._substance is None:

            return PropertyEstimatorException(directory=directory,
                                              message='The substance input is non-optional')

        self._final_number_of_molecules = self._max_molecules

        molecules, number_of_molecules, exception = self._build_molecule_arrays(directory)

        if exception is not None:
            return exception

        packmol_directory = path.join(directory, 'packmol_files')

        # Create packed box
        topology, positions = packmol.pack_box(molecules=molecules,
                                               number_of_copies=number_of_molecules,
                                               mass_density=self._mass_density,
                                               box_aspect_ratio=self._box_aspect_ratio,
                                               center_box=False,
                                               verbose=self._verbose_packmol,
                                               working_directory=packmol_directory,
                                               retain_working_files=self._retain_packmol_files)

        if topology is None or positions is None:

            return PropertyEstimatorException(directory=directory,
                                              message='Packmol failed to complete.')

        self._save_results(directory, topology, positions)

        return self._get_output_dictionary()


@register_calculation_protocol()
class SolvateExistingStructure(BuildCoordinatesPackmol):
    """Creates a set of 3D coordinates with a specified composition.

    Notes
    -----
    The coordinates are created using packmol.
    """

    @protocol_input(str)
    def solute_coordinate_file(self):
        """A file path to the solute to solvate."""
        pass

    @protocol_input(bool)
    def center_solute_in_box(self):
        """If `True`, the center of the solute will be moved to the origin."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._solute_coordinate_file = None
        self._center_solute_in_box = True

    def execute(self, directory, available_resources):

        logging.info(f'Generating coordinates for {self._substance.identifier}: {self.id}')

        if self._substance is None:
            return PropertyEstimatorException(directory=directory,
                                              message='The substance input is non-optional')

        if self._solute_coordinate_file is None:
            return PropertyEstimatorException(directory=directory,
                                              message='The solute coordinate file input is non-optional')

        molecules, number_of_molecules, exception = self._build_molecule_arrays(directory)

        if exception is not None:
            return exception

        packmol_directory = path.join(directory, 'packmol_files')

        # Create packed box
        topology, positions = packmol.pack_box(molecules=molecules,
                                               number_of_copies=number_of_molecules,
                                               structure_to_solvate=self._solute_coordinate_file,
                                               mass_density=self._mass_density,
                                               box_aspect_ratio=self._box_aspect_ratio,
                                               center_box=self._center_solute_in_box,
                                               verbose=self._verbose_packmol,
                                               working_directory=packmol_directory,
                                               retain_working_files=self._retain_packmol_files,
                                               change_chains=self._change_chains)

        if topology is None or positions is None:
            return PropertyEstimatorException(directory=directory,
                                              message='Packmol failed to complete.')

        self._save_results(directory, topology, positions)

        return self._get_output_dictionary()


@register_calculation_protocol()
class BuildDockedCoordinates(BaseProtocol):
    """Creates a set of coordinates for a ligand bound to some receptor.

    Notes
    -----
    This protocol currently only supports docking with the OpenEye OEDocking
    framework.
    """
    class ActivateSiteLocation(Enum):
        """An enum which describes the methods by which a receptors
        activate site(s) is located."""
        ReceptorCenterOfMass = 'ReceptorCenterOfMass'

    @protocol_input(Substance)
    def ligand_substance(self):
        """A substance containing only the ligand to dock."""
        pass

    @protocol_input(int)
    def number_of_ligand_conformers(self):
        """The number of conformers to try and dock into the receptor structure."""
        pass

    @protocol_input(str)
    def receptor_coordinate_file(self):
        """The file path to the coordinates of the receptor molecule."""
        pass

    @protocol_input(ActivateSiteLocation)
    def activate_site_location(self):
        """Defines the method by which the activate site is identified. Currently the only available
        option is `ActivateSiteLocation.ReceptorCenterOfMass`"""
        pass

    @protocol_output(str)
    def docked_ligand_coordinate_path(self):
        """The file path to the coordinates of the ligand in it's docked
        pose, aligned with the initial `receptor_coordinate_file`."""
        pass

    @protocol_output(str)
    def docked_complex_coordinate_path(self):
        """The file path to the docked ligand-receptor complex."""
        pass

    @protocol_output(str)
    def ligand_residue_name(self):
        """The residue name assigned to the docked ligand."""
        return self._ligand_residue_name

    @protocol_output(str)
    def receptor_residue_name(self):
        """The residue name assigned to the receptor."""
        return self._receptor_residue_name

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._ligand_substance = None
        self._number_of_ligand_conformers = 100

        self._receptor_coordinate_file = None
        self._activate_site_location = self.ActivateSiteLocation.ReceptorCenterOfMass

        self._docked_ligand_coordinate_path = None
        self._docked_complex_coordinate_path = None

        self._ligand_residue_name = 'LIG'
        self._receptor_residue_name = 'REC'

    def _create_receptor(self):
        """Create an OpenEye receptor from a mol2 file.

        Returns
        -------
        openeye.oedocking.OEReceptor
            The OpenEye receptor object.
        """
        from openeye import oechem, oedocking

        input_stream = oechem.oemolistream(self._receptor_coordinate_file)

        original_receptor_molecule = oechem.OEGraphMol()
        oechem.OEReadMolecule(input_stream, original_receptor_molecule)

        center_of_mass = oechem.OEFloatArray(3)
        oechem.OEGetCenterOfMass(original_receptor_molecule, center_of_mass)

        receptor = oechem.OEGraphMol()
        oedocking.OEMakeReceptor(receptor, original_receptor_molecule,
                                 center_of_mass[0], center_of_mass[1], center_of_mass[2])

        return receptor

    def _create_ligand(self):
        """Create an OpenEye receptor from a mol2 file.

        Returns
        -------
        openeye.oechem.OEMol
            The OpenEye ligand object with multiple conformers.
        """
        from openforcefield.topology import Molecule

        ligand = Molecule.from_smiles(self._ligand_substance.components[0].smiles)
        ligand.generate_conformers(n_conformers=self._number_of_ligand_conformers)

        # Assign AM1-BCC charges to the ligand just as an initial guess
        # for docking. In future, we may want to get the charge model
        # directly from the force field.
        ligand.compute_partial_charges_am1bcc()

        return ligand.to_openeye()

    def execute(self, directory, available_resources):

        if (len(self._ligand_substance.components) != 1 or
            self._ligand_substance.components[0].role != Substance.ComponentRole.Ligand):

            return PropertyEstimatorException(directory=directory,
                                              message='The ligand substance must contain a single ligand component.')

        import mdtraj
        from openeye import oechem, oedocking

        logging.info('Initializing the receptor molecule.')
        receptor_molecule = self._create_receptor()

        logging.info('Initializing the ligand molecule.')
        ligand_molecule = self._create_ligand()

        logging.info('Initializing the docking object.')

        # Dock the ligand to the receptor.
        dock = oedocking.OEDock()
        dock.Initialize(receptor_molecule)

        docked_ligand = oechem.OEGraphMol()

        logging.info('Performing the docking.')

        status = dock.DockMultiConformerMolecule(docked_ligand, ligand_molecule)

        if status != oedocking.OEDockingReturnCode_Success:

            return PropertyEstimatorException(directory=directory,
                                              message='The ligand could not be successfully docked')

        docking_method = oedocking.OEDockMethodGetName(oedocking.OEDockMethod_Default)
        oedocking.OESetSDScore(docked_ligand, dock, docking_method)

        dock.AnnotatePose(docked_ligand)

        self._docked_ligand_coordinate_path = path.join(directory, 'ligand.pdb')

        output_stream = oechem.oemolostream(self._docked_ligand_coordinate_path)
        oechem.OEWriteMolecule(output_stream, docked_ligand)
        output_stream.close()

        receptor_pdb_path = path.join(directory, 'receptor.pdb')

        output_stream = oechem.oemolostream(receptor_pdb_path)
        oechem.OEWriteMolecule(output_stream, receptor_molecule)
        output_stream.close()

        ligand_trajectory = mdtraj.load(self._docked_ligand_coordinate_path)

        ligand_residue = ligand_trajectory.topology.residue(0)
        ligand_residue.name = self._ligand_residue_name

        # Save the ligand file with the correct residue name.
        ligand_trajectory.save(self._docked_ligand_coordinate_path)
        
        receptor_trajectory = mdtraj.load(receptor_pdb_path)

        receptor_residue = receptor_trajectory.topology.residue(0)
        receptor_residue.name = self._receptor_residue_name

        # Create a merged ligand-receptor topology.
        complex_topology = ligand_trajectory.topology.copy()

        atom_mapping = {}

        new_residue = complex_topology.add_residue(receptor_residue.name, complex_topology.chain(0))

        for receptor_atom in receptor_residue.atoms:

            new_atom = complex_topology.add_atom(receptor_atom.name, receptor_atom.element, new_residue,
                                                 serial=receptor_atom.serial)

            atom_mapping[receptor_atom] = new_atom

        for bond in receptor_trajectory.topology.bonds:

            complex_topology.add_bond(atom_mapping[bond[0]], atom_mapping[bond[1]],
                                      type=bond.type, order=bond.order)

        complex_positions = []

        complex_positions.extend(ligand_trajectory.openmm_positions(0).value_in_unit(unit.angstrom))
        complex_positions.extend(receptor_trajectory.openmm_positions(0).value_in_unit(unit.angstrom))

        complex_positions *= unit.angstrom

        self._docked_complex_coordinate_path = path.join(directory, 'complex.pdb')

        with open(self._docked_complex_coordinate_path, 'w+') as file:

            app.PDBFile.writeFile(complex_topology.to_openmm(),
                                  complex_positions, file)

        return self._get_output_dictionary()
