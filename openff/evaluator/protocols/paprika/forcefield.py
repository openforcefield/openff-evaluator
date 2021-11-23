import abc
import logging
import os
import subprocess
from typing import Optional, Union

import numpy as np
import parmed as pmd
from paprika.build.system import TLeap

try:
    import openmm
    import openmm.app as app
    import openmm.unit as openmm_unit
    from openmm.app import element as E
except ImportError:
    import simtk.openmm as openmm
    import simtk.openmm.app as app
    from simtk import unit as openmm_unit
    from simtk.openmm.app import element as E

from openff.units.openmm import to_openmm

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.forcefield import (
    ForceFieldSource,
    GAFFForceField,
    SmirnoffForceFieldSource,
    TLeapForceFieldSource,
)
from openff.evaluator.forcefield.system import ParameterizedSystem
from openff.evaluator.protocols.forcefield import BaseBuildSystem, BuildSmirnoffSystem
from openff.evaluator.substances import Substance
from openff.evaluator.utils import is_file_and_not_empty
from openff.evaluator.utils.utils import temporarily_change_directory
from openff.evaluator.workflow import Protocol, workflow_protocol
from openff.evaluator.workflow.attributes import InputAttribute, OutputAttribute

logger = logging.getLogger(__name__)


@workflow_protocol()
class PaprikaBuildSystem(Protocol, abc.ABC):
    force_field_path = InputAttribute(
        docstring="The file path to the force field parameters to assign to the system.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    coordinate_file_path = InputAttribute(
        docstring="The file path to the PDB coordinate file which defines the "
        "topology of the system to which the force field parameters "
        "will be assigned.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    host_file_paths = InputAttribute(
        docstring="The paths for host related files.",
        type_hint=dict,
        default_value=UNDEFINED,
    )
    guest_file_paths = InputAttribute(
        docstring="The paths for guest related files.",
        type_hint=Union[dict, None],
        default_value=None,
    )
    enable_hmr = InputAttribute(
        docstring="Whether to repartition the masses of hydrogen atoms.",
        type_hint=bool,
        default_value=False,
    )

    substance = InputAttribute(
        docstring="The composition of the system.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )

    parameterized_system = OutputAttribute(
        docstring="The parameterized system object.", type_hint=ParameterizedSystem
    )

    def _execute(self, directory, available_resources):
        force_field_source = ForceFieldSource.from_json(self.force_field_path)

        if isinstance(force_field_source, SmirnoffForceFieldSource):
            build_protocol = BuildSmirnoffSystem("")

        elif isinstance(force_field_source, TLeapForceFieldSource):
            build_protocol = PaprikaBuildTLeapSystem("")
            build_protocol.host_file_paths = self.host_file_paths
            build_protocol.guest_file_paths = self.guest_file_paths

        else:
            raise ValueError(
                "Only SMIRNOFF and GAFF force fields are supported by this protocol."
            )

        build_protocol.force_field_path = self.force_field_path
        build_protocol.substance = self.substance
        build_protocol.coordinate_file_path = self.coordinate_file_path
        build_protocol.enable_hmr = self.enable_hmr
        build_protocol.execute(directory, available_resources)

        self.parameterized_system = build_protocol.parameterized_system


@workflow_protocol()
class PaprikaBuildTLeapSystem(BaseBuildSystem):
    """Parametrise a host-guest system the AMBER-based force-field
    using the `tleap package <http://ambermd.org/AmberTools.php>`_.

    Notes
    -----
    * This protocol uses Taproom for the input files.
    """

    force_field_path = InputAttribute(
        docstring="The file path to the force field parameters to assign to the system.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    coordinate_file_path = InputAttribute(
        docstring="The file path to the PDB coordinate file which defines the "
        "topology of the system to which the force field parameters "
        "will be assigned.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    host_file_paths = InputAttribute(
        docstring="The paths for host related files.",
        type_hint=dict,
        default_value=UNDEFINED,
    )
    guest_file_paths = InputAttribute(
        docstring="The paths for guest related files.",
        type_hint=Union[dict, None],
        default_value=None,
    )
    enable_hmr = InputAttribute(
        docstring="Whether to repartition the masses of hydrogen atoms.",
        type_hint=bool,
        default_value=False,
    )

    @staticmethod
    def generate_gaff_atom_types(
        gaff_version: str,
        mol2_file: str,
        resname: Optional[str] = None,
        create_frcmod: Optional[bool] = True,
        working_directory: Optional[str] = "./",
    ):
        """
        Given a MOL2 file, generate another MOL2 file with GAFF atom type along
        with the *.frcmod file.

        Parameters
        ----------
        gaff_version
            The GAFF version to generate atom types and parameters.
        mol2_file
            The name of the MOL2 file.
        resname
            Residue name if different from the one in the MOL2 file.
        create_frcmod
            Option to generate frcmod file.
        working_directory
            Directory to store the files.

        Returns
        -------
        processed_mol2
            The name of the MOL2 file with GAFF atom types.
        frcmod_path
            The name of the *.frcmod file.
        residue_name
            The residue name from the MOL2 file.
        """

        with temporarily_change_directory(working_directory):
            processed_mol2 = mol2_file.split("/")[-1].replace(
                ".mol2", f".{gaff_version}.mol2"
            )
            antechamber_exec = [
                "antechamber",
                "-fi",
                "mol2",
                "-i",
                mol2_file,
                "-fo",
                "mol2",
                "-o",
                processed_mol2,
                "-at",
                gaff_version,
                "-pf",
                "y",
            ]
            if resname is not None:
                antechamber_exec.append("-rn")
                antechamber_exec.append(resname)

            antechamber_process = subprocess.Popen(
                antechamber_exec,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            antechamber_output, antechamber_error = antechamber_process.communicate()
            antechamber_exit_code = antechamber_process.returncode

            with open(processed_mol2.replace(".mol2", ".antechamber.log"), "w") as file:
                file.write(f"error code: {antechamber_exit_code}\nstdout:\n\n")
                file.write("stdout:\n\n")
                file.write(antechamber_output.decode())
                file.write("\nstderr:\n\n")
                file.write(antechamber_error.decode())

            if not os.path.isfile(processed_mol2):
                raise RuntimeError(
                    f"antechamber failed to assign atom types to the input mol2 file "
                    f"({mol2_file})"
                )

            structure = pmd.load_file(processed_mol2, structure=True)
            residue_name = np.unique(
                [atom.residue.name for atom in structure.topology.atoms()]
            )[0]

            if create_frcmod:
                frcmod_path = PaprikaBuildTLeapSystem.generate_frcmod(
                    gaff_version,
                    processed_mol2,
                )

                return processed_mol2, frcmod_path, residue_name

        return processed_mol2, residue_name

    @staticmethod
    def generate_frcmod(
        gaff_version: str,
        mol2_file: str,
        print_all_parm: Optional[bool] = False,
        working_directory: Optional[str] = "./",
    ):
        with temporarily_change_directory(working_directory):
            frcmod_path = mol2_file.replace(".mol2", ".frcmod")

            prmchk2_exec = [
                "parmchk2",
                "-i",
                mol2_file,
                "-f",
                "mol2",
                "-o",
                frcmod_path,
                "-s",
                gaff_version,
            ]
            if print_all_parm:
                prmchk2_exec += ["-a", "Y"]

            prmchk2_process = subprocess.Popen(
                prmchk2_exec,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            prmchk2_output, prmchk2_error = prmchk2_process.communicate()
            prmchk2_exit_code = prmchk2_process.returncode

            with open(mol2_file.replace(".mol2", ".parmchk2.log"), "w") as file:
                file.write(f"error code: {prmchk2_exit_code}\nstdout:\n\n")
                file.write(prmchk2_output.decode())
                file.write("\nstderr:\n\n")
                file.write(prmchk2_error.decode())

            if not os.path.isfile(frcmod_path):
                raise RuntimeError(
                    f"parmchk2 failed to assign missing {gaff_version} parameters "
                    f"to the antechamber created mol2 file ({mol2_file})",
                )

        return os.path.join(working_directory, frcmod_path)

    @staticmethod
    def generate_single_topology(
        gaff_version: str,
        mol2_file: str,
        frcmod_file: Optional[str] = None,
        resname: Optional[str] = "MOL",
        ignore_warnings: bool = True,
        working_directory: Optional[str] = "./",
    ):
        """
        Generate AMBER topology file for a single molecule with GAFF.

        Parameters
        ----------
        gaff_version
            The GAFF version to generate atom types and parameters.
        mol2_file
            The name of the MOL2 file.
        frcmod_file
            The name of the frcmod file.
        resname
            Residue name if different from the one in the MOL2 file.
        ignore_warnings
            Whether to ignore warnings from TLeap output.
        working_directory
            Directory to store the files.
        """

        system = TLeap()
        system.output_path = working_directory
        system.output_prefix = mol2_file.split(".")[0] + f".{gaff_version}"
        system.pbc_type = None
        system.neutralize = False
        system.template_lines = [
            f"source leaprc.{gaff_version}",
            f"{resname} = loadmol2 {mol2_file}",
            f"saveamberparm {resname} {system.output_prefix}.prmtop {system.output_prefix}.rst7",
            "quit",
        ]
        if frcmod_file is not None:
            system.template_lines.insert(1, f"loadamberparams {frcmod_file}")

        system.build(clean_files=False, ignore_warnings=ignore_warnings)

    def _execute(self, directory, available_resources):
        # from paprika.evaluator.amber import generate_gaff
        # Check GAFF version
        force_field_source = ForceFieldSource.from_json(self.force_field_path)
        if force_field_source.leap_source == "leaprc.gaff2":
            amber_type = "gaff2"
        elif force_field_source.leap_source == "leaprc.gaff":
            amber_type = "gaff"
        else:
            raise ValueError(
                f"The {force_field_source.leap_source} source is currently "
                f"unsupported. Only the 'leaprc.gaff2' and 'leaprc.gaff' "
                f" sources are supported."
            )

        # Generate GAFF Atom types
        host_mol2, host_frcmod, host_resname = self.generate_gaff_atom_types(
            gaff_version=amber_type,
            mol2_file=self.host_file_paths["host_mol2_path"],
            working_directory=directory,
        )
        if self.guest_file_paths:
            guest_mol2, guest_frcmod, guest_resname = self.generate_gaff_atom_types(
                gaff_version=amber_type,
                mol2_file=self.guest_file_paths["guest_mol2_path"],
                working_directory=directory,
            )

        coordinate_file_path = os.path.relpath(self.coordinate_file_path, directory)
        host_monomer = self.host_file_paths["host_monomer_path"].split("/")[-1]

        # Build Amber topology
        system = TLeap()
        system.output_path = directory
        system.output_prefix = "system"
        system.pbc_type = None
        system.neutralize = False

        # Set GAFF library
        system.template_lines = [f"source {force_field_source.leap_source}"]

        # Set GB Radii if running with implicit solvent
        if force_field_source.igb:
            gb_radii = {
                1: "mbondi",
                2: "mbondi2",
                5: "mbondi2",
            }
            system.template_lines += [
                f"set default PBRadii {gb_radii[force_field_source.igb]}"
            ]
        else:
            system.template_lines += ["source leaprc.water.tip3p"]

        # Load frcmod file(s)
        system.template_lines += [f"loadamberparams {host_frcmod}"]

        if self.guest_file_paths:
            system.template_lines += [f"loadamberparams {guest_frcmod}"]

        if force_field_source.custom_frcmod:
            gaff_force_field = GAFFForceField()
            gaff_force_field.frcmod_parameters = force_field_source.custom_frcmod
            gaff_force_field.to_file(os.path.join(directory, "custom.frcmod"))

            system.template_lines += ["loadamberparams custom.frcmod"]

        # Add MOL2 file(s)
        with open(self.host_file_paths["host_tleap_template"], "r") as f:
            for line in f.readlines():
                if line == "\n" or line.startswith("save"):
                    continue
                if host_monomer in line:
                    line = line.replace(
                        host_monomer, self.host_file_paths["host_monomer_path"]
                    )
                system.template_lines += [line]

        if self.guest_file_paths:
            system.template_lines += [
                f"{guest_resname} = loadmol2 {guest_mol2}",
            ]

        # Add coordinate file
        system.template_lines += [f"model = loadpdb {coordinate_file_path}"]

        if force_field_source.igb is None:
            pdbfile = app.PDBFile(self.coordinate_file_path)
            vector = pdbfile.topology.getPeriodicBoxVectors()
            vec_x = vector[0][0].value_in_unit(openmm_unit.angstrom)
            vec_y = vector[1][1].value_in_unit(openmm_unit.angstrom)
            vec_z = vector[2][2].value_in_unit(openmm_unit.angstrom)
            system.template_lines += [
                f"set model box {{ {vec_x:.3f} {vec_y:.3f} {vec_z:.3f} }}"
            ]

        # Build TLeap
        system.build(clean_files=False)

        if not is_file_and_not_empty(
            os.path.join(system.output_path, system.output_prefix + ".prmtop")
        ) and not is_file_and_not_empty(
            os.path.join(system.output_path, system.output_prefix + ".rst7")
        ):
            raise RuntimeError("tleap failed to execute.")

        # Create OpenMM XML file
        prmtop = app.AmberPrmtopFile(
            os.path.join(system.output_path, system.output_prefix + ".prmtop")
        )
        inpcrd = app.AmberInpcrdFile(
            os.path.join(system.output_path, system.output_prefix + ".rst7")
        )

        if force_field_source.igb:
            solvent_model = {
                1: app.HCT,
                2: app.OBC1,
                5: app.OBC2,
            }
            system = prmtop.createSystem(
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
                implicitSolvent=solvent_model[force_field_source.igb],
                gbsaModel=force_field_source.sa_model,
                hydrogenMass=3.024 * openmm_unit.dalton if self.enable_hmr else None,
            )

            # Change GB radii if specified in custom_frcmod file.
            if force_field_source.custom_frcmod:
                if len(force_field_source.custom_frcmod["GBSA"]) != 0:
                    # Get GB Force object from system
                    gbsa_force = None
                    for force in system.getForces():
                        if isinstance(force, openmm.CustomGBForce) or isinstance(
                            force, openmm.GBSAOBCForce
                        ):
                            gbsa_force = force

                    # Loop over custom GB Radii
                    offset_factor = 0.009  # nm
                    all_bonds = app.internal._get_bonded_atom_list(prmtop.topology)
                    for atom_mask in gaff_force_field.frcmod_parameters["GBSA"]:
                        GB_radii = gaff_force_field.frcmod_parameters["GBSA"][
                            atom_mask
                        ]["radius"]
                        GB_scale = gaff_force_field.frcmod_parameters["GBSA"][
                            atom_mask
                        ]["scale"]

                        # Get element of atom
                        mask_element = E.get_by_symbol(atom_mask[0])
                        connect_element = None
                        if "-" in atom_mask:
                            connect_element = E.get_by_symbol(atom_mask.split("-")[-1])

                        # Find atom in system
                        for atom in prmtop.topology.atoms():
                            current_atom = None
                            element = atom.element

                            if element is mask_element and connect_element is None:
                                current_atom = atom

                            elif element is mask_element and connect_element:
                                bondeds = all_bonds[atom]
                                if bondeds[0].element is connect_element:
                                    current_atom = atom

                            if current_atom:
                                current_param = gbsa_force.getParticleParameters(
                                    current_atom.index
                                )
                                charge = current_param[0]
                                offset_radii = GB_radii - offset_factor
                                scaled_radii = offset_radii * GB_scale
                                gbsa_force.setParticleParameters(
                                    current_atom.index,
                                    [charge, offset_radii, scaled_radii],
                                )

        else:
            system = prmtop.createSystem(
                nonbondedMethod=app.PME,
                nonbondedCutoff=to_openmm(force_field_source.cutoff),
                constraints=app.HBonds,
                rigidWater=True,
                removeCMMotion=False,
                hydrogenMass=3.024 * openmm_unit.dalton if self.enable_hmr else None,
            )
            system.setDefaultPeriodicBoxVectors(*inpcrd.getBoxVectors())

        # Serialize the system object.
        system_path = os.path.join(directory, "system.xml")

        with open(system_path, "w") as file:
            file.write(openmm.XmlSerializer.serialize(system))

        self.parameterized_system = ParameterizedSystem(
            substance=self.substance,
            force_field=force_field_source,
            topology_path=self.coordinate_file_path,
            system_path=system_path,
        )
