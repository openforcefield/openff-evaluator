"""
Units tests for openff.evaluator.protocols.forcefield
"""
import re
import tempfile
from os import path

from openff.toolkit.topology import Molecule
from openff.toolkit.utils.rdkit_wrapper import RDKitToolkitWrapper
from openff.units import unit
from openff.utilities import skip_if_missing

from openff.evaluator._tests.utils import build_tip3p_smirnoff_force_field
from openff.evaluator.backends import ComputeResources
from openff.evaluator.forcefield import LigParGenForceFieldSource, TLeapForceFieldSource
from openff.evaluator.forcefield.forcefield import FoyerForceFieldSource
from openff.evaluator.protocols.coordinates import BuildCoordinatesPackmol
from openff.evaluator.protocols.forcefield import (
    BuildFoyerSystem,
    BuildLigParGenSystem,
    BuildSmirnoffSystem,
    BuildTLeapSystem,
)
from openff.evaluator.protocols.openmm import OpenMMEnergyMinimisation
from openff.evaluator.substances import Substance


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
        assign_parameters.execute(directory)
        assert path.isfile(assign_parameters.parameterized_system.system_path)


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
        assign_parameters.execute(directory)
        assert path.isfile(assign_parameters.parameterized_system.system_path)


def test_build_ligpargen_system(requests_mock):
    force_field_source = LigParGenForceFieldSource(
        request_url="http://testligpargen.com/request",
        download_url="http://testligpargen.com/download",
    )

    substance = Substance.from_components("C", "O")

    def request_callback(request, context):
        context.status_code = 200
        smiles = re.search(r'"smiData"\r\n\r\n(.*?)\r\n', request.text).group(1)

        molecule = Molecule.from_smiles(smiles, toolkit_registry=RDKitToolkitWrapper())
        smiles = molecule.to_smiles(
            isomeric=False,
            explicit_hydrogens=False,
            mapped=False,
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
        assign_parameters.execute(directory)
        assert path.isfile(assign_parameters.parameterized_system.system_path)


@skip_if_missing("foyer")
def test_build_foyer_oplsaa_system():
    force_field_source = FoyerForceFieldSource("oplsaa")
    substance = Substance.from_components("C", "CC", "c1ccccc1", "CC(=O)O")

    with tempfile.TemporaryDirectory() as directory:
        force_field_path = path.join(directory, "ff.json")

        with open(force_field_path, "w") as file:
            file.write(force_field_source.json())

        build_coordinates = BuildCoordinatesPackmol("build_coordinates")
        build_coordinates.max_molecules = 8
        build_coordinates.substance = substance
        build_coordinates.mass_density = 0.005 * unit.gram / unit.milliliter
        build_coordinates.execute(directory)

        assign_parameters = BuildFoyerSystem("assign_parameters")
        assign_parameters.force_field_path = force_field_path
        assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
        assign_parameters.substance = substance
        assign_parameters.execute(directory)
        assert path.isfile(assign_parameters.parameterized_system.system_path)

        energy_minimisation = OpenMMEnergyMinimisation("energy_minimisation")
        energy_minimisation.input_coordinate_file = (
            build_coordinates.coordinate_file_path
        )
        energy_minimisation.parameterized_system = (
            assign_parameters.parameterized_system
        )
        energy_minimisation.execute(directory, ComputeResources())
        assert path.isfile(energy_minimisation.output_coordinate_file)


@skip_if_missing("foyer")
def test_build_foyer_xml_system():
    with tempfile.TemporaryDirectory() as directory:
        force_field_source_path = path.join(directory, "ff.json")
        force_field_xml_path = path.join(directory, "foyer_ff.xml")
        force_field_source = FoyerForceFieldSource(force_field_xml_path)
        substance = Substance.from_components("C")

        with open(force_field_source_path, "w") as file:
            file.write(force_field_source.json())

        with open(force_field_xml_path, "w") as file:
            file.write(
                """<ForceField name="methane_test" version="0.0.3" combining_rule="geometric">
 <AtomTypes>
  <Type name="H" class="H" element="H" mass="1.008" def="H"/>
  <Type name="C" class="C" element="C" mass="12.011" def="C"/>
 </AtomTypes>
 <NonbondedForce coulomb14scale="0.5" lj14scale="0.5">
  <Atom type="H" charge="0.1" sigma="2.0" epsilon="0.1"/>
  <Atom type="C" charge="-0.4" sigma="3.0" epsilon="0.5"/>
 </NonbondedForce>
 <HarmonicBondForce>
  <Bond class1="C" class2="H" length="1.0" k="1000."/>
 </HarmonicBondForce>
 <HarmonicAngleForce>
  <Angle class1="H" class2="C" class3="H" angle="1.88991" k="100."/>
 </HarmonicAngleForce>
</ForceField>"""
            )

        build_coordinates = BuildCoordinatesPackmol("build_coordinates")
        build_coordinates.max_molecules = 1
        build_coordinates.substance = substance
        build_coordinates.mass_density = 0.005 * unit.gram / unit.milliliter
        build_coordinates.execute(directory)

        assign_parameters = BuildFoyerSystem("assign_parameters")
        assign_parameters.force_field_path = force_field_source_path
        assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
        assign_parameters.substance = substance
        assign_parameters.execute(directory)
        assert path.isfile(assign_parameters.parameterized_system.system_path)

        energy_minimisation = OpenMMEnergyMinimisation("energy_minimisation")
        energy_minimisation.input_coordinate_file = (
            build_coordinates.coordinate_file_path
        )
        energy_minimisation.parameterized_system = (
            assign_parameters.parameterized_system
        )
        energy_minimisation.execute(directory, ComputeResources())
        assert path.isfile(energy_minimisation.output_coordinate_file)
