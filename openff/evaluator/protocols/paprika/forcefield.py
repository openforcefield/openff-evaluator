import abc
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Optional, Tuple, Union

import numpy as np
import parmed as pmd
import simtk.unit as simtk_unit
from openff.toolkit.topology import Molecule
from paprika.build.system import TLeap
from simtk.openmm.app import AmberPrmtopFile
from simtk.openmm.app import element as E

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.forcefield import (
    ForceFieldSource,
    SmirnoffForceFieldSource,
    TLeapForceFieldSource,
)
from openff.evaluator.forcefield.system import ParameterizedSystem
from openff.evaluator.protocols.forcefield import BaseBuildSystem, BuildSmirnoffSystem
from openff.evaluator.substances import Component, Substance
from openff.evaluator.utils.openmm import pint_quantity_to_openmm
from openff.evaluator.utils.utils import (
    is_file_and_not_empty,
    is_number,
    temporarily_change_directory,
)
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
            build_protocol.force_field_path = self.force_field_path
            build_protocol.substance = self.substance
            build_protocol.coordinate_file_path = self.coordinate_file_path
            build_protocol.enable_hmr = self.enable_hmr
            build_protocol.execute(directory, available_resources)
            self.parameterized_system = build_protocol.parameterized_system

        elif isinstance(force_field_source, TLeapForceFieldSource):
            build_protocol = PaprikaBuildTLeapSystem("")
            build_protocol.force_field_path = self.force_field_path
            build_protocol.substance = self.substance
            build_protocol.host_file_paths = self.host_file_paths
            build_protocol.guest_file_paths = self.guest_file_paths
            build_protocol.coordinate_file_path = self.coordinate_file_path
            build_protocol.enable_hmr = self.enable_hmr
            build_protocol.execute(directory, available_resources)
            self.parameterized_system = build_protocol.parameterized_system

        else:
            raise ValueError(
                "Only SMIRNOFF and GAFF force fields are supported by this protocol."
            )


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
    ) -> Tuple[str, str, str]:
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

        system.build(clean_files=False, ignore_warnings=True)

    def _execute(self, directory, available_resources):

        import simtk.openmm as openmm
        import simtk.openmm.app as app

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
            # with open(os.path.join(directory, "custom.frcmod"), "w") as f:
            #    f.writelines("Remark line goes here\n")

            #    for key in force_field_source.custom_frcmod.keys():
            #        if key == "GBSA":
            #            continue

            #        f.writelines(f"{key.upper()}\n")
            #        for line in force_field_source.custom_frcmod[key]:
            #            f.writelines(line)
            #        f.writelines("\n")

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
            vec_x = vector[0][0].value_in_unit(simtk_unit.angstrom)
            vec_y = vector[1][1].value_in_unit(simtk_unit.angstrom)
            vec_z = vector[2][2].value_in_unit(simtk_unit.angstrom)
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
                hydrogenMass=3.024 if self.enable_hmr else None,
            )

            # Change GB radii if specified in custom_frcmod file.
            if force_field_source.custom_frcmod:
                if len(force_field_source.custom_frcmod["GBSA"]) != 0:

                    from simtk.openmm import CustomGBForce, GBSAOBCForce
                    from simtk.openmm.app.internal.customgbforces import (
                        _get_bonded_atom_list,
                        _screen_parameter,
                    )

                    # Get GB Force object from system
                    gbsa_force = None
                    for force in system.getForces():
                        if isinstance(force, CustomGBForce) or isinstance(
                            force, GBSAOBCForce
                        ):
                            gbsa_force = force

                    # Loop over custom GB Radii
                    offset_factor = 0.009
                    all_bonds = _get_bonded_atom_list(prmtop.topology)
                    for atom_mask in gaff_force_field.frcmod_parameters["GBSA"]:
                        GB_radii = (
                            gaff_force_field.frcmod_parameters["GBSA"][atom_mask][
                                "radius"
                            ]
                            / 10
                        )

                        mask_element = E.get_by_symbol(atom_mask[0])
                        connect_element = None
                        if "-" in atom_mask:
                            connect_element = E.get_by_symbol(atom_mask.split("-")[-1])

                        # Loop over atoms in the system
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
                                scaled_radii = offset_radii * _screen_parameter(atom)[0]
                                gbsa_force.setParticleParameters(
                                    current_atom.index,
                                    [charge, offset_radii, scaled_radii],
                                )

                    # for atom_mask in force_field_source.custom_frcmod["GBSA"]:
                    #    atom_type = app.element.get_by_symbol(split_line[0])
                    #    GB_radii = float(split_line[1]) / 10  # convert Angstrom to nm

                    #    # Loop through each atom of the system
                    #    for atom in [atom for atom in prmtop_file.topology.atoms()]:
                    #        if atom.element == atom_type:
                    #            current_param = gb_force.getParticleParameters(
                    #                atom.index
                    #            )
                    #            charge = current_param[0]
                    #            offset_radii = GB_radii - offset_factor
                    #            scaled_radii = offset_radii * _screen_parameter(atom)[0]
                    #            gb_force.setParticleParameters(
                    #                atom.index, [charge, offset_radii, scaled_radii]
                    #            )

        else:
            system = prmtop.createSystem(
                nonbondedMethod=app.PME,
                nonbondedCutoff=pint_quantity_to_openmm(force_field_source.cutoff),
                constraints=app.HBonds,
                rigidWater=True,
                removeCMMotion=False,
                hydrogenMass=3.024 if self.enable_hmr else None,
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


class GAFFForceField:
    @property
    def gaff_version(self):
        return self._gaff_version

    @property
    def data_set(self):
        return self._data_set

    @property
    def igb(self):
        return self._igb

    @property
    def smiles_list(self):
        return self._smiles_list

    @property
    def topology(self):
        return self._topology

    @property
    def frcmod_parameters(self):
        return self._frcmod_parameters

    @frcmod_parameters.setter
    def frcmod_parameters(self, value):
        self._frcmod_parameters = value

    def __init__(self, data_set=None, gaff_version="gaff", igb=None):
        self._gaff_version = gaff_version
        self._data_set = data_set
        self._igb = igb
        self._smiles_list = []
        self._topology = None
        self._frcmod_parameters = {}

        if data_set is not None:
            self._initialize()

    def _initialize(self):
        from openff.evaluator.datasets.taproom import TaproomDataSet

        # Check GAFF version
        if self.gaff_version not in ["gaff", "gaff2"]:
            raise KeyError(
                f"Specified GAFF version {self.gaff_version} is not supported."
            )

        # Collect all smiles in data set
        self._smiles_list = []

        if isinstance(self._data_set, TaproomDataSet):
            for substance in self._data_set.substances:
                for component in substance.components:
                    if component.role == Component.Role.Solvent:
                        continue

                    if component.smiles not in self._smiles_list:
                        self._smiles_list.append(component.smiles)

        elif isinstance(self._data_set, Substance):
            for component in self._data_set.components:
                if component.role == Component.Role.Solvent:
                    continue

                if component.smiles not in self._smiles_list:
                    self._smiles_list.append(component.smiles)

        # Extract GAFF parameters
        working_directory = tempfile.mkdtemp()
        with temporarily_change_directory(working_directory):

            molecule_list = []

            for i, smiles in enumerate(self._smiles_list):
                mol = Molecule.from_smiles(smiles)
                mol.partial_charges = (
                    np.zeros(mol.n_atoms) * simtk_unit.elementary_charge
                )
                mol.to_file(f"MOL{i}.mol2", file_format="MOL2")

                (
                    mol2_file,
                    frcmod_file,
                    resname,
                ) = PaprikaBuildTLeapSystem.generate_gaff_atom_types(
                    gaff_version=self.gaff_version,
                    mol2_file=f"MOL{i}.mol2",
                    resname=f"MOL{i}",
                    create_frcmod=True,
                )
                PaprikaBuildTLeapSystem.generate_single_topology(
                    gaff_version=self.gaff_version,
                    mol2_file=mol2_file,
                    frcmod_file=frcmod_file,
                    resname=resname,
                )
                molecule_list.append(f"MOL{i}.{self.gaff_version}.prmtop")

            # Load Topologies in ParmEd
            topology = pmd.load_file(molecule_list[0], structure=True)
            for i, molecule in enumerate(molecule_list):
                if i == 0:
                    continue
                topology += pmd.load_file(molecule, structure=True)
            topology.save("full.prmtop")
            self._topology = AmberPrmtopFile(
                "full.prmtop"
            )  # Doing this because of a bug in ParmEd

            # Generate full frcmod file
            pmd.tools.writeFrcmod(topology, "complex.frcmod").execute()
            self._frcmod_parameters = GAFFForceField.parse_frcmod("complex.frcmod")

        shutil.rmtree(working_directory)

        if self.igb:
            from simtk.openmm.app.internal.customgbforces import _get_bonded_atom_list

            all_bonds = _get_bonded_atom_list(self._topology.topology)

            # Sets the mbondi radii
            if self.igb == 1:
                default_radius = 1.5
                element_to_const_radius = {
                    E.nitrogen: 1.55,
                    E.oxygen: 1.5,
                    E.fluorine: 1.5,
                    E.silicon: 2.1,
                    E.phosphorus: 1.85,
                    E.sulfur: 1.8,
                    E.chlorine: 1.7,
                }

                for atom in self._topology.topology.atoms():
                    element = atom.element

                    # Radius of H atom depends on element it is bonded to
                    if element in (E.hydrogen, E.deuterium):
                        bondeds = all_bonds[atom]
                        if bondeds[0].element in (E.carbon, E.nitrogen):
                            radii = 1.3
                            mask = "H-C" if bondeds[0].element is E.carbon else "H-N"
                        elif bondeds[0].element in (E.oxygen, E.sulfur):
                            radii = 0.8
                            mask = "H-O" if bondeds[0].element is E.oxygen else "H-S"
                        else:
                            radii = 1.2
                            mask = "H"

                    # Radius of C atom depeends on what type it is
                    elif element is E.carbon:
                        radii = 1.7
                        mask = "C"

                    # All other elements have fixed radii
                    else:
                        radii = element_to_const_radius.get(element, default_radius)
                        mask = element.symbol

                    # Store radii into dictionary
                    if mask not in self._frcmod_parameters["GBSA"]:
                        self._frcmod_parameters["GBSA"].update(
                            {
                                mask: {
                                    "radius": radii,
                                    "cosmetic": None,
                                }
                            }
                        )

            # Sets the mbondi2 radii
            elif self.igb in [2, 5]:
                default_radius = 1.5
                element_to_const_radius = {
                    E.nitrogen: 1.55,
                    E.oxygen: 1.5,
                    E.fluorine: 1.5,
                    E.silicon: 2.1,
                    E.phosphorus: 1.85,
                    E.sulfur: 1.8,
                    E.chlorine: 1.7,
                }

                for atom in self._topology.topology.atoms():
                    element = atom.element

                    # Radius of H atom depends on element it is bonded to
                    if element in (E.hydrogen, E.deuterium):
                        bondeds = all_bonds[atom]
                        if bondeds[0].element is E.nitrogen:
                            radii = 1.3
                            mask = "H-N"
                        else:
                            radii = 1.2
                            mask = "H"

                    # Radius of C atom depeends on what type it is
                    elif element is E.carbon:
                        radii = 1.7
                        mask = "C"

                    # All other elements have fixed radii
                    else:
                        radii = element_to_const_radius.get(element, default_radius)
                        mask = element.symbol

                    # Store radii into dictionary
                    if mask not in self._frcmod_parameters["GBSA"]:
                        self._frcmod_parameters["GBSA"].update(
                            {
                                mask: {
                                    "radius": radii,
                                    "cosmetic": None,
                                }
                            }
                        )

    def get_parameter_value(self, tagname, atom_mask, *parameter):
        if tagname not in self._frcmod_parameters.keys():
            raise KeyError(f"The tag `{tagname}` does not exist in the parameter list.")
        if atom_mask not in self._frcmod_parameters[tagname]:
            raise KeyError(f"The atom mask `{atom_mask}` is listed under `{tagname}`.")

        param_attribute = {}
        for param in parameter:
            if param in self._frcmod_parameters[tagname][atom_mask]:
                param_attribute[param] = self._frcmod_parameters[tagname][atom_mask][
                    param
                ]
            else:
                raise KeyError(
                    f"The attribute `{param}` is not an attribute of `{atom_mask}`."
                )

        return param_attribute

    def set_parameter_value(self, tagname, atom_mask, parameter, value):
        if tagname not in self._frcmod_parameters.keys():
            raise KeyError(f"The tag `{tagname}` does not exist in the parameter list.")
        if atom_mask not in self._frcmod_parameters[tagname]:
            raise KeyError(f"The atom mask `{atom_mask}` is listed under `{tagname}`.")
        if parameter not in self._frcmod_parameters[tagname][atom_mask]:
            raise KeyError(
                f"The attribute `{parameter}` is not an attribute of `{atom_mask}`."
            )

        self._frcmod_parameters[tagname][atom_mask][parameter] = value

    def tag_parameter_to_optimize(self, tagname, atom_mask, *parameter):
        if tagname not in self._frcmod_parameters.keys():
            raise KeyError(f"The tag `{tagname}` does not exist in the parameter list.")
        if atom_mask not in self._frcmod_parameters[tagname]:
            raise KeyError(f"The atom mask `{atom_mask}` is listed under `{tagname}`.")

        for param in parameter:
            if param in self._frcmod_parameters[tagname][atom_mask]:
                cosmetic = "# PRM"

                if tagname == "BOND":
                    if param == "k":
                        cosmetic += " 1"
                    elif param == "length":
                        cosmetic += " 2"

                elif tagname == "ANGLE":
                    if param == "k":
                        cosmetic += " 1"
                    elif param == "angle":
                        cosmetic += " 2"

                elif tagname == "DIHE":
                    if param == "scaling":
                        cosmetic += " 1"
                    elif param == "barrier":
                        cosmetic += " 2"
                    elif param == "phase":
                        cosmetic += " 3"
                    elif param == "periodicity":
                        cosmetic += " 4"

                elif tagname == "IMPROPER":
                    if param == "barrier":
                        cosmetic += " 1"
                    elif param == "phase":
                        cosmetic += " 2"
                    elif param == "periodicity":
                        cosmetic += " 3"

                elif tagname in "NONBON":
                    if param == "rmin_half":
                        cosmetic += " 1"
                    elif param == "epsilon":
                        cosmetic += " 2"

                elif tagname == "GBSA":
                    if param == "radius":
                        cosmetic += " 1"

                self._frcmod_parameters[tagname][atom_mask]["cosmetic"] = cosmetic

            else:
                raise KeyError(
                    f"The attribute `{param}` is not an attribute of `{atom_mask}`."
                )

    @staticmethod
    def parse_frcmod(file_path):
        """Reads in a frcmod file and store the information in a dictionary.

        .. note::
            Parameters with polarizabilities are not supported yet and will be ignored.

        Parameters
        ----------
        file: os.PathLike
            The file path to check.

        Returns
        -------
        frcmod_dict: dict
            A dictionary containing the custom parameters.
        """

        frcmod_dict = {
            "MASS": {},
            "BOND": {},
            "ANGLE": {},
            "DIHE": {},
            "IMPROPER": {},
            "NONBON": {},
            "GBSA": {},
        }

        with open(file_path, "r") as f:

            for i, line in enumerate(f.readlines()):

                if line.strip() == 0 or line.startswith("\n") or i == 0:
                    continue

                if re.match("MASS", line.strip().upper()):
                    keyword = "MASS"
                    continue
                elif re.match("BOND|BONDS", line.strip().upper()):
                    keyword = "BOND"
                    continue
                elif re.match("ANGLE", line.strip().upper()):
                    keyword = "ANGLE"
                    continue
                elif re.match("DIHE|DIHEDRAL", line.strip().upper()):
                    keyword = "DIHE"
                    continue
                elif re.match("IMPROPER", line.strip().upper()):
                    keyword = "IMPROPER"
                    continue
                elif re.match("NONBON|NONB|NONBONDED", line.strip().upper()):
                    keyword = "NONBON"
                    continue
                elif re.match("RADII|GBSA|GBRADII", line.strip().upper()):
                    keyword = "GBSA"
                    continue

                # Read parameter
                cosmetic = None
                parameter = line.split()
                if "#" in line:
                    parameter = line[: line.index("#")].split()
                    cosmetic = line[line.index("#") :]

                atom_columns = []
                for j in range(len(parameter)):
                    # Convert to float
                    if is_number(parameter[j]) and not parameter[j].isdigit():
                        parameter[j] = float(parameter[j])

                    # Convert to int
                    elif is_number(parameter[j]) and parameter[j].isdigit():
                        parameter[j] = int(parameter[j])

                    # Get list element that are strings
                    elif "SC" not in parameter[j]:
                        atom_columns.append(j)

                # Get proper formatting for atom masks
                mask = parameter[0]
                if len(atom_columns) > 1:
                    atom_mask = "".join(parameter[: len(atom_columns)])
                    for k, col in enumerate(atom_columns):
                        parameter.remove(parameter[col - k])
                    mask = "-".join(f"{atom:2s}" for atom in atom_mask.split("-"))
                else:
                    parameter.pop(0)

                # Build parameter dictionary
                if keyword == "MASS":
                    param_dict = {"mass": parameter[0], "cosmetic": cosmetic}
                elif keyword == "BOND":
                    param_dict = {
                        "k": parameter[0],
                        "length": parameter[1],
                        "cosmetic": cosmetic,
                    }
                elif keyword == "ANGLE":
                    param_dict = {
                        "k": parameter[0],
                        "angle": parameter[1],
                        "cosmetic": cosmetic,
                    }
                elif keyword == "DIHE":
                    param_dict = {
                        "scaling": parameter[0],
                        "barrier": parameter[1],
                        "phase": parameter[2],
                        "periodicity": parameter[3],
                        "SCEE": None,
                        "SCNB": None,
                        "cosmetic": cosmetic,
                    }
                    if len(parameter) > 4:
                        if "SCEE" in parameter[4]:
                            param_dict["SCEE"] = float(parameter[4].split("=")[1])
                        if "SCNB" in parameter[4]:
                            param_dict["SCNB"] = float(parameter[4].split("=")[1])
                    if len(parameter) > 5:
                        if "SCEE" in parameter[5]:
                            param_dict["SCEE"] = float(parameter[5].split("=")[1])
                        if "SCNB" in parameter[5]:
                            param_dict["SCNB"] = float(parameter[5].split("=")[1])
                elif keyword == "IMPROPER":
                    param_dict = {
                        "barrier": parameter[0],
                        "phase": parameter[1],
                        "periodicity": parameter[2],
                        "cosmetic": cosmetic,
                    }
                elif keyword == "NONBON":
                    param_dict = {
                        "rmin_half": parameter[0],
                        "epsilon": parameter[1],
                        "cosmetic": cosmetic,
                    }
                elif keyword == "GBSA":
                    param_dict = {"radius": parameter[0], "cosmetic": cosmetic}

                # Update dictionary
                frcmod_dict[keyword].update({mask: param_dict})

        return frcmod_dict

    @staticmethod
    def _parameter_to_string(tagname, atom_mask, parameters):
        if tagname == "MASS":
            parameter_line = f"{atom_mask:2s}"
            parameter_line += f"{parameters['mass']:10.3f}"
        if tagname == "BOND":
            parameter_line = f"{atom_mask:5s}"
            parameter_line += f"{parameters['k']:10.3f}"
            parameter_line += f"{parameters['length']:10.3f}"
        if tagname == "ANGLE":
            parameter_line = f"{atom_mask:8s}"
            parameter_line += f"{parameters['k']:10.3f}"
            parameter_line += f"{parameters['angle']:10.3f}"
        if tagname == "DIHE":
            parameter_line = f"{atom_mask:11s}"
            parameter_line += f"{parameters['scaling']:4d}"
            parameter_line += f"{parameters['barrier']:15.8f}"
            parameter_line += f"{parameters['phase']:10.3f}"
            parameter_line += f"{parameters['periodicity']:10.2f}"
            if parameters["SCEE"]:
                parameter_line += f"  SCEE={parameters['SCEE']:.1f}"
            if parameters["SCNB"]:
                parameter_line += f"  SCNB={parameters['SCNB']:.1f}"
        if tagname == "IMPROPER":
            parameter_line = f"{atom_mask:11s}"
            parameter_line += f"{parameters['barrier']:15.8f}"
            parameter_line += f"{parameters['phase']:10.3f}"
            parameter_line += f"{parameters['periodicity']:10.2f}"
        if tagname == "NONBON":
            parameter_line = f"{atom_mask:4s}"
            parameter_line += f"{parameters['rmin_half']:15.8f}"
            parameter_line += f"{parameters['epsilon']:15.8f}"
        if tagname == "GBSA":
            parameter_line = f"{atom_mask:4s}"
            parameter_line += f"{parameters['radius']:15.8f}"

        if parameters["cosmetic"]:
            parameter_line += f"   {parameters['cosmetic']}"

        parameter_line += "\n"

        return parameter_line

    def to_file(self, file_path, skip_gbsa=True):

        with open(file_path, "w") as f:
            f.writelines("Remark line goes here\n")

            for tagname in self._frcmod_parameters.keys():
                if skip_gbsa and tagname == "GBSA":
                    continue
                f.writelines(f"{tagname.upper()}\n")
                for atom_mask in self._frcmod_parameters[tagname]:
                    f.writelines(
                        self._parameter_to_string(
                            tagname,
                            atom_mask,
                            self._frcmod_parameters[tagname][atom_mask],
                        )
                    )
                f.writelines("\n")

    def json(self, file_path):
        with open(file_path, "w") as f:
            json.dump(self._frcmod_parameters, f)

    # @classmethod
    # def from_json(cls, file_path):
    #    obj = cls.__new__(cls)
    #    super(GAFFForceField, obj).__init__()
    #    with open(file_path, "r") as f:
    #        obj.frcmod_parameters = json.loads(f.read())
