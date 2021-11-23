"""
Units tests for openff.evaluator.protocols.forcefield
"""
import re
import tempfile
from os import path

import simtk.unit as simtk_unit
from cmiles.utils import load_molecule, mol_to_smiles
from simtk.openmm import XmlSerializer
from simtk.openmm.app import PDBFile

from openff.evaluator.datasets.taproom import TaproomDataSet
from openff.evaluator.forcefield import LigParGenForceFieldSource, TLeapForceFieldSource
from openff.evaluator.forcefield.forcefield import GAFFForceField
from openff.evaluator.protocols.coordinates import BuildCoordinatesPackmol
from openff.evaluator.protocols.forcefield import (
    BuildLigParGenSystem,
    BuildSmirnoffSystem,
    BuildTLeapSystem,
)
from openff.evaluator.protocols.paprika.coordinates import PreparePullCoordinates
from openff.evaluator.protocols.paprika.forcefield import PaprikaBuildTLeapSystem
from openff.evaluator.substances import Substance
from openff.evaluator.tests.utils import build_tip3p_smirnoff_force_field
from openff.evaluator.utils import is_file_and_not_empty


def test_build_smirnoff_system():
    with tempfile.TemporaryDirectory() as directory:
        force_field_path = path.join(directory, "ff.json")

        with open(force_field_path, "w") as file:
            file.write(build_tip3p_smirnoff_force_field().json())

        substance = Substance.from_components("C", "O", "CO", "C(=O)N")

        build_coordinates = BuildCoordinatesPackmol("build_coordinates")
        build_coordinates.max_molecules = 8
        build_coordinates.substance = substance
        build_coordinates.execute(directory)

        assign_parameters = BuildSmirnoffSystem("assign_parameters")
        assign_parameters.force_field_path = force_field_path
        assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
        assign_parameters.substance = substance
        assign_parameters.enable_hmr = True
        assign_parameters.execute(directory)
        assert path.isfile(assign_parameters.parameterized_system.system_path)

        pdbfile = PDBFile(assign_parameters.parameterized_system.topology_path)
        with open(assign_parameters.parameterized_system.system_path, "r") as f:
            system = XmlSerializer.deserialize(f.read())

        for atom in pdbfile.topology.atoms():
            if atom.element.name == "hydrogen" and atom.residue.name != "HOH":
                assert system.getParticleMass(atom.index) == 3.024 * simtk_unit.dalton


def test_build_tleap_system():
    with tempfile.TemporaryDirectory() as directory:
        force_field_path = path.join(directory, "ff.json")

        with open(force_field_path, "w") as file:
            file.write(TLeapForceFieldSource().json())

        substance = Substance.from_components("CCCCCCCC", "O", "C(=O)N")

        build_coordinates = BuildCoordinatesPackmol("build_coordinates")
        build_coordinates.max_molecules = 9
        build_coordinates.substance = substance
        build_coordinates.execute(directory)

        assign_parameters = BuildTLeapSystem("assign_parameters")
        assign_parameters.force_field_path = force_field_path
        assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
        assign_parameters.substance = substance
        assign_parameters.enable_hmr = True
        assign_parameters.execute(directory)
        assert path.isfile(assign_parameters.parameterized_system.system_path)

        pdbfile = PDBFile(assign_parameters.parameterized_system.topology_path)
        with open(assign_parameters.parameterized_system.system_path, "r") as f:
            system = XmlSerializer.deserialize(f.read())

        for atom in pdbfile.topology.atoms():
            if atom.element.name == "hydrogen" and atom.residue.name != "HOH":
                assert system.getParticleMass(atom.index) == 3.024 * simtk_unit.dalton


def test_paprika_build_tleap_system():
    with tempfile.TemporaryDirectory() as directory:
        # Get Taproom info
        data_set = TaproomDataSet(
            host_codes=["acd"], guest_codes=["bam"], in_vacuum=True
        )

        host_file_paths = data_set.properties[0].metadata["host_file_paths"]
        guest_file_paths = data_set.properties[0].metadata["guest_file_paths"]
        guest_orientation_mask = data_set.properties[0].metadata[
            "guest_orientation_mask"
        ]
        complex_file_path = data_set.properties[0].metadata["guest_orientations"][0][
            "coordinate_path"
        ]
        pull_distance = data_set.properties[0].metadata["pull_distance"]
        n_pull_windows = data_set.properties[0].metadata["n_pull_windows"]
        substance = data_set.properties[0].substance

        # Build host-guest complex coordinates
        build_coordinates = PreparePullCoordinates("build_coordinates")
        build_coordinates.substance = substance
        build_coordinates.complex_file_path = complex_file_path
        build_coordinates.guest_orientation_mask = guest_orientation_mask
        build_coordinates.pull_window_index = 0
        build_coordinates.pull_distance = pull_distance
        build_coordinates.n_pull_windows = n_pull_windows
        build_coordinates.remove_pbc_vectors = True
        build_coordinates.execute(directory)

        # Test TLeap FF
        force_field_path = path.join(directory, "ff.json")
        force_field = GAFFForceField(
            data_set, gaff_version="gaff2", igb=2, sa_model="ACE"
        )
        assert force_field.frcmod_parameters is not None

        with open(force_field_path, "w") as file:
            file.write(TLeapForceFieldSource.from_object(force_field).json())

        assert is_file_and_not_empty(force_field_path)

        assign_parameters = PaprikaBuildTLeapSystem("assign_tleap_parameters")
        assign_parameters.force_field_path = force_field_path
        assign_parameters.substance = substance
        assign_parameters.host_file_paths = host_file_paths
        assign_parameters.guest_file_paths = guest_file_paths
        assign_parameters.coordinate_file_path = (
            build_coordinates.output_coordinate_path
        )
        assign_parameters.enable_hmr = True
        assign_parameters.execute(directory)
        assert path.isfile(assign_parameters.parameterized_system.system_path)

        pdbfile = PDBFile(assign_parameters.parameterized_system.topology_path)
        with open(assign_parameters.parameterized_system.system_path, "r") as f:
            system = XmlSerializer.deserialize(f.read())

        for atom in pdbfile.topology.atoms():
            if atom.element.name == "hydrogen" and atom.residue.name != "HOH":
                assert system.getParticleMass(atom.index) == 3.024 * simtk_unit.dalton


def test_build_ligpargen_system(requests_mock):
    force_field_source = LigParGenForceFieldSource(
        request_url="http://testligpargen.com/request",
        download_url="http://testligpargen.com/download",
    )

    substance = Substance.from_components("C", "O")

    def request_callback(request, context):
        context.status_code = 200
        smiles = re.search(r'"smiData"\r\n\r\n(.*?)\r\n', request.text).group(1)

        cmiles_molecule = load_molecule(smiles, toolkit="rdkit")
        smiles = mol_to_smiles(
            cmiles_molecule, isomeric=False, explicit_hydrogen=False, mapped=False
        )

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
        build_coordinates.execute(directory)

        assign_parameters = BuildLigParGenSystem("assign_parameters")
        assign_parameters.force_field_path = force_field_path
        assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
        assign_parameters.substance = substance
        assign_parameters.enable_hmr = True
        assign_parameters.execute(directory)
        assert path.isfile(assign_parameters.parameterized_system.system_path)

        pdbfile = PDBFile(assign_parameters.parameterized_system.topology_path)
        with open(assign_parameters.parameterized_system.system_path, "r") as f:
            system = XmlSerializer.deserialize(f.read())

        for atom in pdbfile.topology.atoms():
            if atom.element.name == "hydrogen" and atom.residue.name != "HOH":
                assert system.getParticleMass(atom.index) == 3.024 * simtk_unit.dalton
