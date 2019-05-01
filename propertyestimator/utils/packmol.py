"""
An API for interacting with `packmol <http://m3g.iqm.unicamp.br/packmol/home.shtml>`_.

Notes
-----
Based on the `SolvationToolkit <https://github.com/MobleyLab/SolvationToolkit>`_.
"""

import logging
import os
import random
import shutil
import string
import subprocess
from distutils.spawn import find_executable
from tempfile import mkdtemp

from simtk import openmm
from simtk import unit
from simtk.openmm import app

PACKMOL_PATH = find_executable("packmol") or shutil.which("packmol") or \
               None if 'PACKMOL' not in os.environ else os.environ['PACKMOL']

_HEADER_TEMPLATE = """
# Mixture
tolerance {0:f}
filetype pdb
output {1:s}
"""
# add_amber_ter

_BOX_TEMPLATE = """
structure {0:s}
  number {1:d}
  inside box 0. 0. 0. {2:f} {3:f} {4:f}
end structure
"""

_SOLVATE_TEMPLATE = """
structure {0:s}
  number 1
  fixed 0. 0. 0. 0. 0. 0.
  centerofmass
end structure
"""


def pack_box(molecules,
             number_of_copies,
             structure_to_solvate=None,
             tolerance=2.0,
             box_size=None,
             mass_density=None,
             verbose=False,
             working_directory=None,
             retain_working_files=False):

    """Run packmol to generate a box containing a mixture of molecules.

    Parameters
    ----------
    molecules : list of openeye.oechem.OEMol
        The molecules in the system (with 3D geometries)
    number_of_copies : list of int
        A list of the number of copies of each molecule type, of length
        equal to the length of `molecules`.
    structure_to_solvate: str, optional
        A file path to the PDB coordinates of the structure to be solvated.
    tolerance : float
        The minimum spacing between molecules during packing in angstroms.
    box_size : simtk.unit.Quantity, optional
        The size of the box to generate in units compatible with angstroms. If `None`,
        `mass_density` must be provided.
    mass_density : simtk.unit.Quantity, optional
        Target mass density for final system with units compatible with g/mL. If `None`,
        `box_size` must be provided.
    verbose : bool
        If True, verbose output is written.
    working_directory: str, optional
        The directory in which to generate the temporary working files. If `None`,
        a temporary one will be created.
    retain_working_files: bool
        If True all of the working files, such as individual molecule coordinate
        files, will be retained.

    Returns
    -------
    topology : simtk.openmm.Topology
        Topology of the resulting system
    positions : simtk.unit.Quantity
        A `simtk.unit.Quantity` wrapped `numpy.ndarray` (shape=[natoms,3]) which contains
        the created positions with units compatible with angstroms.
    """

    if box_size is None and mass_density is None:
        raise ValueError('Either a `box_size` or `mass_density` must be specified.')

    # noinspection PyTypeChecker
    if len(molecules) != len(number_of_copies):
        raise ValueError('Length of `molecules` and `number_of_copies` must be identical.')

    temporary_directory = False

    if working_directory is None:

        working_directory = mkdtemp()
        temporary_directory = True

    elif not os.path.isdir(working_directory):
        os.mkdir(working_directory)

    # Create PDB files for all components.
    pdb_filenames = list()
    mdtraj_topologies = []

    for index, molecule in enumerate(molecules):

        tmp_filename = os.path.join(working_directory, f'{index}.pdb')
        pdb_filenames.append(tmp_filename)

        mdtraj_topologies.append(_create_pdb_and_topology(molecule, tmp_filename))

    # Run packmol
    if PACKMOL_PATH is None:
        raise IOError("Packmol not found, cannot run pack_box()")

    output_filename = os.path.join(working_directory, "packmol_output.pdb")

    # Approximate volume to initialize box
    if box_size is None:

        # Estimate box_size from mass density.
        box_size = _approximate_volume_by_density(molecules,
                                                  number_of_copies,
                                                  mass_density)

    unitless_box_angstrom = box_size.value_in_unit(unit.angstrom)

    packmol_input = _HEADER_TEMPLATE.format(tolerance, output_filename)

    for (pdb_filename, molecule, count) in zip(pdb_filenames,
                                               molecules,
                                               number_of_copies):

        packmol_input += _BOX_TEMPLATE.format(pdb_filename,
                                              count,
                                              unitless_box_angstrom,
                                              unitless_box_angstrom,
                                              unitless_box_angstrom)

    if structure_to_solvate is not None:

        if not os.path.isfile(structure_to_solvate):
            raise ValueError(f'The structure to solvate ({structure_to_solvate}) does not exist.')

        packmol_input += _SOLVATE_TEMPLATE.format(structure_to_solvate)

    # Write packmol input
    packmol_filename = os.path.join(working_directory, "packmol_input.txt")

    with open(packmol_filename, 'w') as file_handle:
        file_handle.write(packmol_input)

    packmol_succeeded = False

    with open(packmol_filename) as file_handle:

        result = subprocess.check_output(PACKMOL_PATH,
                                         stdin=file_handle,
                                         stderr=subprocess.STDOUT).decode("utf-8")

        if verbose:
            logging.info(result)

        packmol_succeeded = result.find('Success!') > 0

    if not retain_working_files:

        os.unlink(packmol_filename)

        for filename in pdb_filenames:
            os.unlink(filename)

    if not packmol_succeeded:

        if verbose:
            logging.info("Packmol failed to converge")

        if os.path.isfile(output_filename):
            os.unlink(output_filename)

        if temporary_directory and not retain_working_files:
            shutil.rmtree(working_directory)

        return None, None

    # Append missing connect statements to the end of the
    # output file.
    _append_connect_statements(output_filename,
                               mdtraj_topologies,
                               number_of_copies)

    # Read the resulting PDB file.
    pdb_file = app.PDBFile(output_filename)

    if not retain_working_files:

        os.unlink(output_filename)

        if temporary_directory:
            shutil.rmtree(working_directory)

    # Extract topology and positions
    topology = pdb_file.getTopology()
    positions = pdb_file.getPositions()

    unitless_box_nm = box_size / unit.nanometers

    box_vector_x = openmm.Vec3(unitless_box_nm, 0, 0)
    box_vector_y = openmm.Vec3(0, unitless_box_nm, 0)
    box_vector_z = openmm.Vec3(0, 0, unitless_box_nm)

    # Set the periodic box vectors.
    topology.setPeriodicBoxVectors([box_vector_x, box_vector_y, box_vector_z] * unit.nanometers)

    return topology, positions


def _approximate_volume_by_density(molecules,
                                   n_copies,
                                   mass_density=1.0*unit.grams/unit.milliliters,
                                   box_scaleup_factor=1.1):
    """Generate an approximate box size based on the number and molecular weight of molecules present, and a target
    density for the final solvated mixture. If no density is specified, the target density is assumed to be 1 g/ml.

    Parameters
    ----------
    molecules : list of OEMol
        Molecules in the system (with 3D geometries)
    n_copies : list of int (same length as 'molecules')
        Number of copies of the molecules.
    box_scaleup_factor : float, optional, default = 1.1
        Factor by which the estimated box size is increased
    mass_density : simtk.unit.Quantity with units compatible with grams/milliliters, optional,
                   default = 1.0*grams/milliliters
        Target mass density for final system, if available.

    Returns
    -------
    box_edge : simtk.unit.Quantity with units compatible with angstroms
        The size (edge length) of the box to generate.

    Notes
    -----
    By default, boxes are only modestly large. This approach has not been extensively tested for stability but has been
    used in the Mobley lab for perhaps ~100 different systems without substantial problems.

    """
    from openeye import oechem

    # Load molecules to get molecular weights
    volume = 0.0 * unit.angstrom**3

    for (molecule, number) in zip(molecules, n_copies):

        molecule_mass = oechem.OECalculateMolecularWeight(molecule) * \
                        unit.grams / unit.mole / unit.AVOGADRO_CONSTANT_NA

        molecule_volume = molecule_mass / mass_density

        volume += molecule_volume * number

    # Add 2 angs to help ease PBC issues.
    box_edge = volume**(1.0/3.0) * box_scaleup_factor + 2.0 * unit.angstrom

    return box_edge


def _append_connect_statements(file_name, molecule_topologies, n_copies):

    lines = []

    with open(file_name, 'r') as file:
        lines = file.readlines()

    if lines[len(lines) - 1].find('END') == 0:
        lines.pop()

    atom_counter = 0

    # TODO: Does packmol always give the exact number of mols asked for?
    # In future may be better way to figure this out.
    for (topology, count) in zip(molecule_topologies, n_copies):

        bonds = {}

        for bond in topology.bonds:

            index_a = bond[0].index
            index_b = bond[1].index

            if index_a not in bonds:
                bonds[index_a] = []
            if index_b not in bonds:
                bonds[index_b] = []

            bonds[index_a].append(index_b)
            bonds[index_b].append(index_a)

        for i in range(count):

            for index_a in bonds:

                if len(bonds[index_a]) == 0:
                    continue

                connect_string = 'CONECT' + "%5d" % (index_a + atom_counter + 1)

                for j in range(len(bonds[index_a])):
                    connect_string += "%5d" % (bonds[index_a][j] + atom_counter + 1)

                connect_string += '\n'

                lines.append(connect_string)

            atom_counter += topology.n_atoms

    lines.append('END')

    with open(file_name, 'w') as file:
        file.writelines(lines)


def _create_pdb_and_topology(molecule, file_path):
    """Creates a uniform PDB file and `mdtraj.Topology` from an
    openeye molecule.

    Parameters
    ----------
    molecule: openeye.oechem.OEChem
        The component to create the PDB and topology for.
    file_path: str
        The path pointing to where the PDB file should be created.

    Returns
    -------
    mdtraj.Topology
        The topology of the created PDB file.
    """
    import mdtraj
    from openeye import oechem

    # Write the PDB file
    pdb_flavor = oechem.OEOFlavor_PDB_Default

    ofs = oechem.oemolostream(file_path)
    ofs.SetFlavor(oechem.OEFormat_PDB, pdb_flavor)

    # Fix residue names
    residue_name = ''.join([random.choice(string.ascii_uppercase) for _ in range(3)])

    oechem.OEWriteConstMolecule(ofs, molecule)
    ofs.close()

    with open(file_path, 'rb') as file:
        pdb_contents = file.read().decode().replace('UNL', residue_name)

    with open(file_path, 'wb') as file:
        file.write(pdb_contents.encode())

    oe_pdb = mdtraj.load_pdb(file_path)
    return oe_pdb.topology
