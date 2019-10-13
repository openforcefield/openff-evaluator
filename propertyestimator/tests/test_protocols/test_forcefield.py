"""
Units tests for propertyestimator.protocols.forcefield
"""
import tempfile
from os import path
from tempfile import NamedTemporaryFile

import pytest
from openforcefield.topology import Molecule, Topology
from simtk.openmm.app import PDBFile

from propertyestimator.forcefield import TLeapForceFieldSource
from propertyestimator.protocols.coordinates import BuildCoordinatesPackmol
from propertyestimator.protocols.forcefield import BuildTLeapSystem, BuildSmirnoffSystem
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.utils import get_data_filename


def test_build_smirnoff_system():

    with tempfile.TemporaryDirectory() as directory:

        force_field_path = path.join(directory, 'ff.json')

        with open(force_field_path, 'w') as file:
            file.write(build_tip3p_smirnoff_force_field().json())

        substance = Substance.from_components('C', 'CO', 'C(=O)N')

        build_coordinates = BuildCoordinatesPackmol('build_coordinates')
        build_coordinates.max_molecules = 9
        build_coordinates.substance = substance
        build_coordinates.execute(directory, None)

        assign_parameters = BuildSmirnoffSystem(f'assign_parameters')
        assign_parameters.force_field_path = force_field_path
        assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
        assign_parameters.substance = substance
        result = assign_parameters.execute(directory, None)

        assert not isinstance(result, PropertyEstimatorException)
        assert path.isfile(assign_parameters.system_path)


def test_build_tleap_system():

    with tempfile.TemporaryDirectory() as directory:

        force_field_path = path.join(directory, 'ff.json')

        with open(force_field_path, 'w') as file:
            file.write(TLeapForceFieldSource().json())

        substance = Substance.from_components('C', 'CO', 'C(=O)N')

        build_coordinates = BuildCoordinatesPackmol('build_coordinates')
        build_coordinates.max_molecules = 9
        build_coordinates.substance = substance
        build_coordinates.execute(directory, None)

        assign_parameters = BuildTLeapSystem(f'assign_parameters')
        assign_parameters.force_field_path = force_field_path
        assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
        assign_parameters.substance = substance
        result = assign_parameters.execute(directory, None)

        assert not isinstance(result, PropertyEstimatorException)
        assert path.isfile(assign_parameters.system_path)


@pytest.mark.parametrize('charge_backend', [BuildTLeapSystem.ChargeBackend.OpenEye,
                                            BuildTLeapSystem.ChargeBackend.AmberTools])
def test_topology_mol_to_mol2(charge_backend):
    """Tests taking an openforcefield topology molecule, generating a conformer,
    calculating partial charges, and writing it to mol2."""

    # Testing to find the correct connectivity information should indicate
    # that the mol2 conversion was successful
    expected_contents = ['''
@<TRIPOS>BOND
     1    1    3 1
     2    2    3 1
     3    3    9 1
     4    3    8 1
     5    1    4 1
     6    2    5 1
     7    2    6 1
     8    2    7 1''', '''@<TRIPOS>BOND
     1    4    6 ar
     2    2    4 ar
     3    1    2 ar
     4    1    3 ar
     5    3    5 ar
     6    5    6 ar
     7    6    7 1
     8    4   11 1
     9    2    9 1
    10    1    8 1
    11    3   10 1
    12    5   12 1
    13    7   13 1
    14    7   14 1
    15    7   15 1''']

    # Constructing the molecule using this SMILES will ensure that the reference molecule
    # (generated here) and topology molecule (generated below from PDB) have a different atom order
    ethanol_smiles = 'C(O)C'
    toluene_smiles = 'c1ccccc1C'

    ethanol = Molecule.from_smiles(ethanol_smiles)
    toluene = Molecule.from_smiles(toluene_smiles)

    pdb_file = PDBFile(get_data_filename('test/molecules/ethanol_toluene.pdb'))
    topology = Topology.from_openmm(pdb_file.topology, unique_molecules=[ethanol, toluene])

    for topology_molecule_index, topology_molecule in enumerate(topology.topology_molecules):
        with NamedTemporaryFile(suffix='.mol2') as output_file:
            BuildTLeapSystem._topology_molecule_to_mol2(topology_molecule,
                                                        output_file.name,
                                                        charge_backend=charge_backend)

            mol2_contents = open(output_file.name).read()

            # Ensure we find the correct connectivity info
            assert expected_contents[topology_molecule_index] in mol2_contents

            # Ensure that the first atom has nonzero coords and charge
            first_atom_line = mol2_contents.split('\n')[7].split()
            assert float(first_atom_line[2]) != 0.
            assert float(first_atom_line[3]) != 0.
            assert float(first_atom_line[4]) != 0.
            assert float(first_atom_line[8]) != 0.
