"""
Units tests for propertyestimator.protocols.forcefield
"""
import re
import tempfile
from os import path
from tempfile import NamedTemporaryFile

import pytest
from openforcefield.topology import Molecule, Topology
from simtk.openmm.app import PDBFile

from propertyestimator.forcefield import (
    LigParGenForceFieldSource,
    TLeapForceFieldSource,
)
from propertyestimator.protocols.coordinates import BuildCoordinatesPackmol
from propertyestimator.protocols.forcefield import (
    BuildLigParGenSystem,
    BuildSmirnoffSystem,
    BuildTLeapSystem,
)
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.utils.utils import get_data_filename


def test_build_smirnoff_system():

    with tempfile.TemporaryDirectory() as directory:

        force_field_path = path.join(directory, "ff.json")

        with open(force_field_path, "w") as file:
            file.write(build_tip3p_smirnoff_force_field().json())

        substance = Substance.from_components("C", "CO", "C(=O)N")

        build_coordinates = BuildCoordinatesPackmol("build_coordinates")
        build_coordinates.max_molecules = 9
        build_coordinates.substance = substance
        build_coordinates.execute(directory, None)

        assign_parameters = BuildSmirnoffSystem(f"assign_parameters")
        assign_parameters.force_field_path = force_field_path
        assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
        assign_parameters.substance = substance
        assign_parameters.execute(directory, None)
        assert path.isfile(assign_parameters.system_path)


def test_build_tleap_system():

    with tempfile.TemporaryDirectory() as directory:

        force_field_path = path.join(directory, "ff.json")

        with open(force_field_path, "w") as file:
            file.write(TLeapForceFieldSource().json())

        substance = Substance.from_components("C", "O", "C(=O)N")

        build_coordinates = BuildCoordinatesPackmol("build_coordinates")
        build_coordinates.max_molecules = 9
        build_coordinates.substance = substance
        build_coordinates.execute(directory, None)

        assign_parameters = BuildTLeapSystem(f"assign_parameters")
        assign_parameters.force_field_path = force_field_path
        assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
        assign_parameters.substance = substance
        assign_parameters.execute(directory, None)
        assert path.isfile(assign_parameters.system_path)


def test_build_ligpargen_system(requests_mock):

    force_field_source = LigParGenForceFieldSource(
        request_url="http://testligpargen.com/request",
        download_url="http://testligpargen.com/download",
    )

    substance = Substance.from_components("C", "O")

    def request_callback(request, context):
        context.status_code = 200
        smiles = re.search(r'"smiData"\r\n\r\n(.*?)\r\n', request.text).group(1)

        assert smiles == "C"
        return 'value="/tmp/0000.xml"'

    def download_callback(_, context):
        context.status_code = 200
        return """
<ForceField>
<AtomTypes>
<Type name="opls_802" class="H802" element="H" mass="1.008000" />
<Type name="opls_804" class="H804" element="H" mass="1.008000" />
<Type name="opls_803" class="H803" element="H" mass="1.008000" />
<Type name="opls_800" class="C800" element="C" mass="12.011000" />
<Type name="opls_801" class="H801" element="H" mass="1.008000" />
</AtomTypes>
<Residues>
<Residue name="UNK">
<Atom name="C00" type="opls_800" />
<Atom name="H01" type="opls_801" />
<Atom name="H02" type="opls_802" />
<Atom name="H03" type="opls_803" />
<Atom name="H04" type="opls_804" />
<Bond from="0" to="1"/>
<Bond from="0" to="2"/>
<Bond from="0" to="3"/>
<Bond from="0" to="4"/>
</Residue>
</Residues>
<HarmonicBondForce>
<Bond class1="H801" class2="C800" length="0.109000" k="284512.000000"/>
<Bond class1="H802" class2="C800" length="0.109000" k="284512.000000"/>
<Bond class1="H803" class2="C800" length="0.109000" k="284512.000000"/>
<Bond class1="H804" class2="C800" length="0.109000" k="284512.000000"/>
</HarmonicBondForce>
<HarmonicAngleForce>
<Angle class1="H801" class2="C800" class3="H802" angle="1.881465" k="276.144000"/>
<Angle class1="H801" class2="C800" class3="H803" angle="1.881465" k="276.144000"/>
<Angle class1="H801" class2="C800" class3="H804" angle="1.881465" k="276.144000"/>
<Angle class1="H802" class2="C800" class3="H803" angle="1.881465" k="276.144000"/>
<Angle class1="H803" class2="C800" class3="H804" angle="1.881465" k="276.144000"/>
<Angle class1="H802" class2="C800" class3="H804" angle="1.881465" k="276.144000"/>
</HarmonicAngleForce>
<PeriodicTorsionForce>
<Improper class1="C800" class2="H801" class3="H802" class4="H803" k1="0.000000" k2="0.000000" k3="0.000000"
k4="0.000000" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00"
phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
<Improper class1="C800" class2="H801" class3="H802" class4="H804" k1="0.000000" k2="0.000000" k3="0.000000"
k4="0.000000" periodicity1="1" periodicity2="2" periodicity3="3" periodicity4="4" phase1="0.00"
phase2="3.141592653589793" phase3="0.00" phase4="3.141592653589793"/>
</PeriodicTorsionForce>
<NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
<Atom type="opls_803" charge="0.074800" sigma="0.250000" epsilon="0.125520" />
<Atom type="opls_802" charge="0.074800" sigma="0.250000" epsilon="0.125520" />
<Atom type="opls_800" charge="-0.299400" sigma="0.350000" epsilon="0.276144" />
<Atom type="opls_804" charge="0.074800" sigma="0.250000" epsilon="0.125520" />
<Atom type="opls_801" charge="0.074800" sigma="0.250000" epsilon="0.125520" />
</NonbondedForce>
</ForceField>
"""

    requests_mock.post(force_field_source.request_url, text=request_callback)
    requests_mock.post(force_field_source.download_url, text=download_callback)

    with tempfile.TemporaryDirectory() as directory:

        force_field_path = path.join(directory, "ff.json")

        with open(force_field_path, "w") as file:
            file.write(force_field_source.json())

        build_coordinates = BuildCoordinatesPackmol("build_coordinates")
        build_coordinates.max_molecules = 8
        build_coordinates.substance = substance
        build_coordinates.execute(directory, None)

        assign_parameters = BuildLigParGenSystem(f"assign_parameters")
        assign_parameters.force_field_path = force_field_path
        assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
        assign_parameters.substance = substance
        assign_parameters.execute(directory, None)
        assert path.isfile(assign_parameters.system_path)


@pytest.mark.parametrize(
    "charge_backend",
    [BuildTLeapSystem.ChargeBackend.OpenEye, BuildTLeapSystem.ChargeBackend.AmberTools],
)
def test_topology_mol_to_mol2(charge_backend):
    """Tests taking an openforcefield topology molecule, generating a conformer,
    calculating partial charges, and writing it to mol2."""

    # Testing to find the correct connectivity information should indicate
    # that the mol2 conversion was successful
    expected_contents = [
        """
@<TRIPOS>BOND
     1    1    3 1
     2    2    3 1
     3    3    9 1
     4    3    8 1
     5    1    4 1
     6    2    5 1
     7    2    6 1
     8    2    7 1""",
        """@<TRIPOS>BOND
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
    15    7   15 1""",
    ]

    # Constructing the molecule using this SMILES will ensure that the reference molecule
    # (generated here) and topology molecule (generated below from PDB) have a different atom order
    ethanol_smiles = "C(O)C"
    toluene_smiles = "c1ccccc1C"

    ethanol = Molecule.from_smiles(ethanol_smiles)
    toluene = Molecule.from_smiles(toluene_smiles)

    pdb_file = PDBFile(get_data_filename("test/molecules/ethanol_toluene.pdb"))
    topology = Topology.from_openmm(
        pdb_file.topology, unique_molecules=[ethanol, toluene]
    )

    for topology_molecule_index, topology_molecule in enumerate(
        topology.topology_molecules
    ):
        with NamedTemporaryFile(suffix=".mol2") as output_file:
            BuildTLeapSystem._topology_molecule_to_mol2(
                topology_molecule, output_file.name, charge_backend=charge_backend
            )

            mol2_contents = open(output_file.name).read()

            # Ensure we find the correct connectivity info
            assert expected_contents[topology_molecule_index] in mol2_contents

            # Ensure that the first atom has nonzero coords and charge
            first_atom_line = mol2_contents.split("\n")[7].split()
            assert float(first_atom_line[2]) != 0.0
            assert float(first_atom_line[3]) != 0.0
            assert float(first_atom_line[4]) != 0.0
            assert float(first_atom_line[8]) != 0.0
