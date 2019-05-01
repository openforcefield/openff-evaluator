"""
A collection of protocols for building coordinates for molecular systems.
"""

import logging
from enum import Enum
from os import path

import numpy as np
from simtk import unit
from simtk.openmm import app

from propertyestimator.substances import Substance
from propertyestimator.utils import packmol, create_molecule_from_smiles
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.workflow.decorators import protocol_input, protocol_output
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


class SupportedFileFormat(Enum):

    PDB = 'pdb'
    MOL2 = 'mol2'


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

    @protocol_input(Substance)
    def substance(self):
        """The composition of the system to build."""
        pass

    @protocol_input(bool)
    def verbose_packmol(self):
        """If True, packmol will be allowed to log verbose information to the logger,
        and any working packmol files will be retained."""
        pass

    @protocol_output(str)
    def coordinate_file_path(self):
        """The file path to the created PDB coordinate file."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        # inputs
        self._substance = None

        # outputs
        self._coordinate_file_path = None
        self._positions = None

        self._max_molecules = 1000
        self._mass_density = 0.95 * unit.grams / unit.milliliters

        self._verbose_packmol = False

    def execute(self, directory, available_resources):

        logging.info(f'Generating coordinates for {self._substance.identifier}: {self.id}')

        if self._substance is None:

            return PropertyEstimatorException(directory=directory,
                                              message='The substance input is non-optional')

        molecules = []

        for component in self._substance.components:

            molecule = create_molecule_from_smiles(component.smiles)

            if molecule is None:

                return PropertyEstimatorException(directory=directory,
                                                  message='{} could not be converted to a Molecule'.format(component))

            molecules.append(molecule)

        # Determine how many molecules of each type will be present in the system.
        number_of_molecules = np.zeros(self._substance.number_of_components, dtype=np.int)

        for index, component in enumerate(self._substance.components):

            mole_fraction = self._substance.get_mole_fraction(component)
            number_of_molecules[index] = int(round(mole_fraction * self._max_molecules))

            if np.isclose(mole_fraction, 0.0):
                number_of_molecules[index] = 1

            if mole_fraction > 0.0 and number_of_molecules[index] == 0:

                message = f'The maximum number of molecules is not high enough to sufficiently ' \
                    f'capture the mole fraction ({mole_fraction}) of {component.identifier}'

                return PropertyEstimatorException(directory=directory, message=message)

        # Create packed box
        topology, positions = packmol.pack_box(molecules=molecules,
                                               n_copies=number_of_molecules,
                                               mass_density=self._mass_density,
                                               verbose=self._verbose_packmol,
                                               working_directory=None,
                                               retain_working_files=False)

        if topology is None or positions is None:

            return PropertyEstimatorException(directory=directory,
                                              message='Packmol failed to complete.')

        self._coordinate_file_path = path.join(directory, 'output.pdb')

        with open(self._coordinate_file_path, 'w+') as minimised_file:
            app.PDBFile.writeFile(topology, positions, minimised_file)

        logging.info('Coordinates generated: ' + self._substance.identifier)

        return self._get_output_dictionary()


# @register_calculation_protocol()
# class BuildCoordinatesForComponent(BaseProtocol):
#     """Creates a set of 3D coordinates for each molecule within a given
#     substance. Optionally, the coordinates will only be generated for
#     those components of a substance that satisfy a certain role, such as
#     only ligands or solutes.
#     """
#     @protocol_input(Substance)
#     def substance(self):
#         """The substance which contains the component of interest."""
#         pass
#
#     @protocol_input(Substance.ComponentRole)
#     def component_role(self):
#         """The substance which contains the component of interest."""
#         pass
#
#     @protocol_input(SupportedFileFormat)
#     def output_format(self):
#         """The output coordinate file format."""
#         pass
#
#     @protocol_output(str)
#     def coordinate_file_path(self):
#         """The file path to the created PDB coordinate file."""
#         pass
#
#     def __init__(self, protocol_id):
#
#         super().__init__(protocol_id)
#
#         self._substance = None
#         self._component_role = Substance.ComponentRole.Undefined
#
#         self._output_format = SupportedFileFormat.MOL2
#
#         self._coordinate_file_path = None
#
#     def execute(self, directory, available_resources):
#
#         from openeye import oechem
#
#         if self._output_format != SupportedFileFormat.MOL2:
#
#             message = 'The BuildCoordinatesForComponent protocol only supports ' \
#                       'generating mol2 files at this time.'
#
#             return PropertyEstimatorException(directory=directory,
#                                               message=message)
#
#         components = [] if self._component_role == Substance.ComponentRole.Undefined else self._substance.components
#
#         if self._component_role != Substance.ComponentRole.Undefined:
#
#             for component in self._substance.components:
#
#                 if component.role != self._component_role:
#                     continue
#
#                 components.append(component)
#
#         if len(components) == 0:
#
#             message = 'The substance does not contain any components which meet' \
#                       ' desired criteria.'
#
#             return PropertyEstimatorException(directory=directory,
#                                               message=message)
#
#         if len(components) > 1:
#
#             message = 'Currently the BuildCoordinatesForComponent protocol ' \
#                       'only supports building coordinates for a single component.'
#
#             return PropertyEstimatorException(directory=directory,
#                                               message=message)
#
#         component = components[0]
#
#         logging.info(f'Generating coordinates for {component.identifier}: {self.id}')
#
#         oe_molecule = create_molecule_from_smiles(component.smiles)
#
#         if oe_molecule is None:
#
#             return PropertyEstimatorException(directory=directory,
#                                               message=f'{component} could not be converted to an OEMol')
#
#         self._coordinate_file_path = path.join(directory, 'output.mol2')
#
#         # Create the file.
#         ofs = oechem.oemolostream(self._coordinate_file_path)
#         ofs.SetFlavor(oechem.OEFormat_MOL2, oechem.OEOFlavor_MOL2_DEFAULT)
#
#         oechem.OEWriteConstMolecule(ofs, oe_molecule)
#         ofs.close()
#
#         logging.info(f'Coordinates generated: {component.identifier}')
#
#         return self._get_output_dictionary()


@register_calculation_protocol()
class BuildDockedCoordinates(BaseProtocol):
    """Creates a set of coordinates for a ligand bound to some receptor.

    Notes
    -----
    This protocol currently only supports docking with the OpenEye OEDocking
    framework.
    """

    @protocol_input(str)
    def ligand_coordinate_file(self):
        """The file path to the coordinates of the ligand molecule."""
        pass

    @protocol_input(str)
    def receptor_coordinate_file(self):
        """The file path to the coordinates of the receptor molecule."""
        pass

    @protocol_output(str)
    def output_coordinate_file_path(self):
        """The file path to the created PDB coordinate file, which contains the coordinates
        of the ligand and the receptor."""
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._ligand_coordinate_file = None
        self._receptor_coordinate_file = None

        self._coordinate_file_path = None

    def _create_receptor(self, binding_site_box):
        """Create an OpenEye receptor from a PDB file.

        Parameters
        ----------
        binding_site_box : list of float
            The minimum and maximum values of the coordinates of the box
            representing the binding site [xmin, ymin, zmin, xmax, ymax, zmax].

        Returns
        -------
        receptor : openeye.oedocking.OEReceptor
            The OpenEye receptor object.

        """
        from openeye import oechem, oedocking

        input_stream = oechem.oemolistream(self._receptor_coordinate_file)

        original_receptor_molecule = oechem.OEGraphMol()
        oechem.OEReadMolecule(input_stream, original_receptor_molecule)

        box = oedocking.OEBox(*binding_site_box)

        receptor = oechem.OEGraphMol()
        oedocking.OEMakeReceptor(receptor, original_receptor_molecule, box)

        return receptor

    @staticmethod
    def _dock_molecule(self, receptor_molecule, ligand_molecule):
        """Run the multi-conformer docker.

        Parameters
        ----------
        receptor_molecule : openeye.oedocking.OEReceptor
            The openeye receptor.
        ligand_molecule : openeye.oechem.OEMol
            The multi-conformer OpenEye molecule to dock.

        Returns
        -------
        openeye.oechem.OEMol
            The docked OpenEye molecule.
        """
        from openeye import oechem, oedocking

        # Generate docked poses.
        dock = oedocking.OEDock()
        dock.Initialize(receptor_molecule)

        docked_oemol = oechem.OEMol()
        status = dock.DockMultiConformerMolecule(docked_oemol, ligand_molecule, 1)

        # Check for errors
        if status != oedocking.OEDockingReturnCode_Success:
            return None

        return docked_oemol

    def execute(self, directory, available_resources):

        logging.info(f'Generating coordinates for {self._substance.identifier}: {self.id}')

        logging.info('Coordinates generated: ' + self._substance.identifier)

        return self._get_output_dictionary()
