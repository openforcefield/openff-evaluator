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
import tempfile
from distutils.spawn import find_executable
from functools import reduce

import numpy as np
from simtk import openmm

from evaluator import unit
from evaluator.utils.openmm import openmm_quantity_to_pint
from evaluator.utils.utils import temporarily_change_directory

logger = logging.getLogger(__name__)


_PACKMOL_PATH = (
    find_executable("packmol") or shutil.which("packmol") or None
    if "PACKMOL" not in os.environ
    else os.environ["PACKMOL"]
)

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
  fixed {1:f} {2:f} {3:f} 0. 0. 0.
  centerofmass
end structure
"""


def pack_box(
    molecules,
    number_of_copies,
    structure_to_solvate=None,
    tolerance=2.0,
    box_size=None,
    mass_density=None,
    box_aspect_ratio=None,
    verbose=False,
    working_directory=None,
    retain_working_files=False,
):

    """Run packmol to generate a box containing a mixture of molecules.

    Parameters
    ----------
    molecules : list of openforcefield.topology.Molecule
        The molecules in the system.
    number_of_copies : list of int
        A list of the number of copies of each molecule type, of length
        equal to the length of `molecules`.
    structure_to_solvate: str, optional
        A file path to the PDB coordinates of the structure to be solvated.
    tolerance : float
        The minimum spacing between molecules during packing in angstroms.
    box_size : pint.Quantity, optional
        The size of the box to generate in units compatible with angstroms. If `None`,
        `mass_density` must be provided.
    mass_density : pint.Quantity, optional
        Target mass density for final system with units compatible with g/mL. If `None`,
        `box_size` must be provided.
    box_aspect_ratio: list of float, optional
        The aspect ratio of the simulation box, used in conjunction with the `mass_density`
        parameter. If none, an isotropic ratio (i.e. [1.0, 1.0, 1.0]) is used.
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
    positions : pint.Quantity
        A `pint.Quantity` wrapped `numpy.ndarray` (shape=[natoms,3]) which contains
        the created positions with units compatible with angstroms.
    """

    if box_size is None and mass_density is None:
        raise ValueError("Either a `box_size` or `mass_density` must be specified.")

    if box_size is not None and len(box_size) != 3:

        raise ValueError("`box_size` must be a pint.Quantity wrapped list of length 3")

    if mass_density is not None and box_aspect_ratio is None:
        box_aspect_ratio = [1.0, 1.0, 1.0]

    if box_aspect_ratio is not None:

        assert len(box_aspect_ratio) == 3
        assert box_aspect_ratio[0] * box_aspect_ratio[1] * box_aspect_ratio[2] > 0

    # noinspection PyTypeChecker
    if len(molecules) != len(number_of_copies):
        raise ValueError(
            "Length of `molecules` and `number_of_copies` must be identical."
        )

    temporary_directory = False

    if working_directory is None:

        working_directory = tempfile.mkdtemp()
        temporary_directory = True

    elif not os.path.isdir(working_directory):
        os.mkdir(working_directory)

    # Create PDB files for all components.
    pdb_file_paths = []
    pdb_file_names = []

    mdtraj_topologies = []

    # Packmol does not like long file paths, so we need to store just file names
    # and do a temporary cd into the working directory to get around this.
    for index, molecule in enumerate(molecules):

        tmp_file_name = f"{index}.pdb"
        tmp_file_path = os.path.join(working_directory, tmp_file_name)

        pdb_file_paths.append(tmp_file_path)
        pdb_file_names.append(tmp_file_name)

        mdtraj_topologies.append(_create_pdb_and_topology(molecule, tmp_file_path))

    structure_to_solvate_file_name = None

    if structure_to_solvate is not None:

        if not os.path.isfile(structure_to_solvate):
            raise ValueError(
                f"The structure to solvate ({structure_to_solvate}) does not exist."
            )

        structure_to_solvate_file_name = os.path.basename(structure_to_solvate)
        shutil.copyfile(
            structure_to_solvate,
            os.path.join(working_directory, structure_to_solvate_file_name),
        )

    # Run packmol
    if _PACKMOL_PATH is None:
        raise IOError("Packmol not found, cannot run pack_box()")

    output_file_name = "packmol_output.pdb"
    output_file_path = os.path.join(working_directory, output_file_name)

    # Approximate volume to initialize box
    if box_size is None:

        # Estimate box_size from mass density.
        initial_box_length = _approximate_volume_by_density(
            molecules, number_of_copies, mass_density
        )

        initial_box_length_angstrom = initial_box_length.to(unit.angstrom).magnitude

        aspect_ratio_normalizer = (
            box_aspect_ratio[0] * box_aspect_ratio[1] * box_aspect_ratio[2]
        ) ** (1.0 / 3.0)

        box_size = [
            initial_box_length_angstrom * box_aspect_ratio[0],
            initial_box_length_angstrom * box_aspect_ratio[1],
            initial_box_length_angstrom * box_aspect_ratio[2],
        ] * unit.angstrom

        box_size /= aspect_ratio_normalizer

    unitless_box_angstrom = box_size.to(unit.angstrom).magnitude

    packmol_input = _HEADER_TEMPLATE.format(tolerance, output_file_name)

    for (pdb_file_name, molecule, count) in zip(
        pdb_file_names, molecules, number_of_copies
    ):

        packmol_input += _BOX_TEMPLATE.format(
            pdb_file_name,
            count,
            unitless_box_angstrom[0],
            unitless_box_angstrom[1],
            unitless_box_angstrom[2],
        )

    if structure_to_solvate_file_name is not None:

        packmol_input += _SOLVATE_TEMPLATE.format(
            structure_to_solvate_file_name,
            unitless_box_angstrom[0] / 2.0,
            unitless_box_angstrom[1] / 2.0,
            unitless_box_angstrom[2] / 2.0,
        )

    # Write packmol input
    packmol_file_name = "packmol_input.txt"
    packmol_file_path = os.path.join(working_directory, packmol_file_name)

    with open(packmol_file_path, "w") as file_handle:
        file_handle.write(packmol_input)

    with temporarily_change_directory(working_directory):

        with open(packmol_file_name) as file_handle:

            result = subprocess.check_output(
                _PACKMOL_PATH, stdin=file_handle, stderr=subprocess.STDOUT
            ).decode("utf-8")

            if verbose:
                logger.info(result)

            packmol_succeeded = result.find("Success!") > 0

    if not retain_working_files:

        os.unlink(packmol_file_path)

        for file_path in pdb_file_paths:
            os.unlink(file_path)

    if not packmol_succeeded:

        if verbose:
            logger.info("Packmol failed to converge")

        if os.path.isfile(output_file_path):
            os.unlink(output_file_path)

        if temporary_directory and not retain_working_files:
            shutil.rmtree(working_directory)

        return None, None

    # Append missing connect statements to the end of the
    # output file.
    positions, topology = _correct_packmol_output(
        output_file_path, mdtraj_topologies, number_of_copies, structure_to_solvate
    )

    if not retain_working_files:

        os.unlink(output_file_path)

        if temporary_directory:
            shutil.rmtree(working_directory)

    # Add a 2 angstrom buffer to help alleviate PBC issues.
    box_vectors = [
        openmm.Vec3(
            (box_size[0] + 2.0 * unit.angstrom).to(unit.nanometers).magnitude, 0, 0
        ),
        openmm.Vec3(
            0, (box_size[1] + 2.0 * unit.angstrom).to(unit.nanometers).magnitude, 0
        ),
        openmm.Vec3(
            0, 0, (box_size[2] + 2.0 * unit.angstrom).to(unit.nanometers).magnitude
        ),
    ]

    # Set the periodic box vectors.
    from simtk import unit as simtk_unit

    topology.setPeriodicBoxVectors(box_vectors * simtk_unit.nanometers)

    return topology, positions


def _approximate_volume_by_density(
    molecules,
    n_copies,
    mass_density=1.0 * unit.grams / unit.milliliters,
    box_scaleup_factor=1.1,
):
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
    mass_density : pint.Quantity, optional
        The target mass density for final system, if available.
        It should have units compatible with grams/milliliters.

    Returns
    -------
    box_edge : pint.Quantity
        The size (edge length) of the box to generate in units
        compatible with angstroms.

    Notes
    -----
    By default, boxes are only modestly large. This approach has not been
    extensively tested for stability but has been used in the Mobley lab
    for perhaps ~100 different systems without substantial problems.
    """

    # Load molecules to get molecular weights
    volume = 0.0 * unit.angstrom ** 3

    for (molecule, number) in zip(molecules, n_copies):

        molecule_mass = reduce(
            (lambda x, y: x + y), [atom.mass for atom in molecule.atoms]
        )
        molecule_mass = openmm_quantity_to_pint(molecule_mass) / unit.avogadro_constant

        molecule_volume = molecule_mass / mass_density

        volume += molecule_volume * number

    box_edge = volume ** (1.0 / 3.0) * box_scaleup_factor
    return box_edge


def _correct_packmol_output(
    file_path, molecule_topologies, number_of_copies, structure_to_solvate
):
    """Corrects the PDB file output by packmol (i.e adds full connectivity
    information, and extracts the topology and positions.

    Parameters
    ----------
    file_path: str
        The file path to the packmol output file.
    molecule_topologies: list of mdtraj.Topology
        A list of topologies for the molecules which packmol has
        added.
    number_of_copies: list of int
        The total number of each molecule which packmol should have
        created.
    structure_to_solvate: str
        The file path to a preexisting structure which packmol
        has solvated.

    Returns
    -------
    list
        The positions determined by packmol.
    simtk.openmm.app.Topology
        The topology of the created system with full connectivity.
    """

    import mdtraj
    from simtk import unit as simtk_unit

    trajectory = mdtraj.load(file_path)

    atoms_data_frame, _ = trajectory.topology.to_dataframe()

    all_bonds = []
    all_positions = (
        trajectory.openmm_positions(0).value_in_unit(simtk_unit.angstrom)
        * unit.angstrom
    )

    all_topologies = []
    all_copies = []

    all_topologies.extend(molecule_topologies)
    all_copies.extend(number_of_copies)

    if structure_to_solvate is not None:

        solvated_trajectory = mdtraj.load(structure_to_solvate)

        all_topologies.append(solvated_trajectory.topology)
        all_copies.append(1)

    offset = 0

    for (molecule_topology, count) in zip(all_topologies, all_copies):

        _, molecule_bonds = molecule_topology.to_dataframe()

        for i in range(count):

            for bond in molecule_bonds:

                all_bonds.append(
                    [int(bond[0].item()) + offset, int(bond[1].item()) + offset]
                )

            offset += molecule_topology.n_atoms

    if len(all_bonds) > 0:
        all_bonds = np.unique(all_bonds, axis=0).tolist()

    # We have to check whether there are any existing bonds, because mdtraj will
    # sometimes automatically detect some based on residue names (e.g HOH), and
    # this behaviour cannot be disabled.
    existing_bonds = []

    for bond in trajectory.topology.bonds:
        existing_bonds.append(bond)

    for bond in all_bonds:

        atom_a = trajectory.topology.atom(bond[0])
        atom_b = trajectory.topology.atom(bond[1])

        bond_exists = False

        for existing_bond in existing_bonds:

            if (existing_bond.atom1 == atom_a and existing_bond.atom2 == atom_b) or (
                existing_bond.atom2 == atom_a and existing_bond.atom1 == atom_b
            ):

                bond_exists = True
                break

        if bond_exists:
            continue

        trajectory.topology.add_bond(atom_a, atom_b)

    return all_positions, trajectory.topology.to_openmm()


def _create_pdb_and_topology(molecule, file_path):
    """Creates a uniform PDB file and `mdtraj.Topology` from an
    openeye molecule.
    Parameters
    ----------
    molecule: openforcefield.topology.Molecule
        The component to create the PDB and topology for.
    file_path: str
        The path pointing to where the PDB file should be created.
    Returns
    -------
    mdtraj.Topology
        The topology of the created PDB file.
    """
    import mdtraj
    from mdtraj.core import residue_names
    from openforcefield.topology import Topology
    from simtk.openmm.app import PDBFile

    # Check whether the molecule has a configuration defined, and if not,
    # define one.
    if molecule.n_conformers <= 0:
        molecule.generate_conformers(n_conformers=1)

    # Create a temporary pdb file then reload it using mdtraj. This is a
    # necessary workaround as sometimes the PDB saved by the toolkit will
    # have a different atom ordering to the original molecule object.
    topology = Topology.from_molecules([molecule])

    with tempfile.NamedTemporaryFile(mode="r+", suffix=".pdb") as pdb_file:
        PDBFile.writeFile(topology.to_openmm(), molecule.conformers[0], pdb_file)
        pdb_file.flush()
        mdtraj_molecule = mdtraj.load_pdb(pdb_file.name)

    # Change the assigned residue name (sometimes molecules are assigned
    # an amino acid residue name even if that molecule is not an amino acid,
    # e.g. C(CO)N is not Gly) and save the altered object as a pdb.
    smiles = molecule.to_smiles()

    # Choose a random residue name.
    residue_map = {}

    for residue in mdtraj_molecule.topology.residues:

        residue_map[residue.name] = None

        if smiles == "[H]O[H]":

            residue_map[residue.name] = "HOH"

            # Re-assign the water atom names. These need to be set to get
            # correct CONECT statements.
            h_counter = 1

            for atom in residue.atoms:

                if atom.element.symbol == "O":
                    atom.name = "O1"
                else:
                    atom.name = f"H{h_counter}"
                    h_counter += 1

        elif smiles == "[Cl-]":

            residue_map[residue.name] = "Cl-"

            for atom in residue.atoms:
                atom.name = "Cl-"

        elif smiles == "[Na+]":

            residue_map[residue.name] = "Na+"

            for atom in residue.atoms:
                atom.name = "Na+"

    for original_residue_name in residue_map:

        if residue_map[original_residue_name] is not None:
            continue

        # Make sure the residue name is not already reserved as this can
        # occasionally result in bonds being automatically added in the wrong
        # places when loading the pdb file either through mdtraj or openmm

        # noinspection PyProtectedMember
        forbidden_residue_names = [
            *residue_names._AMINO_ACID_CODES,
            *residue_names._SOLVENT_TYPES,
            *residue_names._WATER_RESIDUES,
            "ADE",
            "CYT",
            "CYX",
            "DAD",
            "DGU",
            "FOR",
            "GUA",
            "HID",
            "HIE",
            "HIH",
            "HSD",
            "HSH",
            "HSP",
            "NMA",
            "THY",
            "URA",
        ]

        new_residue_name = "".join(
            [random.choice(string.ascii_uppercase) for _ in range(3)]
        )

        while new_residue_name in forbidden_residue_names:
            # Re-choose the residue name until we find a safe one.
            new_residue_name = "".join(
                [random.choice(string.ascii_uppercase) for _ in range(3)]
            )

        residue_map[original_residue_name] = new_residue_name

    for residue in mdtraj_molecule.topology.residues:
        residue.name = residue_map[residue.name]

    # Create the final pdb file.
    mdtraj_molecule.save_pdb(file_path)
    return mdtraj_molecule.topology
