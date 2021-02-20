import abc
import logging
import os
import subprocess

from paprika.build.system import TLeap

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.forcefield import (
    ForceFieldSource,
    SmirnoffForceFieldSource,
    TLeapForceFieldSource,
)
from openff.evaluator.forcefield.system import ParameterizedSystem
from openff.evaluator.protocols.forcefield import BaseBuildSystem, BuildSmirnoffSystem
from openff.evaluator.substances import Substance
from openff.evaluator.utils.openmm import pint_quantity_to_openmm
from openff.evaluator.utils.utils import (
    is_file_and_not_empty,
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
    host_mol2_path = InputAttribute(
        docstring="The file path to the MOL2 file for the host which defines the "
        "topology of the system and charges to which the force field "
        "parameters will be assigned.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    guest_mol2_path = InputAttribute(
        docstring="The file path to the MOL2 file for the guest which defines the "
        "topology of the system and charges to which the force field "
        "parameters will be assigned.",
        type_hint=str,
        default_value=UNDEFINED,
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
            build_protocol.host_mol2_path = self.host_mol2_path
            build_protocol.guest_mol2_path = self.guest_mol2_path
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
    """Parametrise a host-guest system the Amber based force field.
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
    host_mol2_path = InputAttribute(
        docstring="The file path to the MOL2 file for the host which defines the "
        "topology of the system and charges to which the force field "
        "parameters will be assigned.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    guest_mol2_path = InputAttribute(
        docstring="The file path to the MOL2 file for the guest which defines the "
        "topology of the system and charges to which the force field "
        "parameters will be assigned.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    enable_hmr = InputAttribute(
        docstring="Whether to repartition the masses of hydrogen atoms.",
        type_hint=bool,
        default_value=False,
    )

    @staticmethod
    def _generate_gaff_atom_types(mol2_file, gaff_version, directory):
        import numpy as np
        import parmed as pmd

        with temporarily_change_directory(directory):

            processed_mol2_path = mol2_file.replace(".mol2", f".{gaff_version}.mol2")
            antechamber_process = subprocess.Popen(
                [
                    "antechamber",
                    "-fi",
                    "mol2",
                    "-i",
                    mol2_file,
                    "-fo",
                    "mol2",
                    "-o",
                    processed_mol2_path,
                    "-at",
                    gaff_version,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            antechamber_output, antechamber_error = antechamber_process.communicate()
            antechamber_exit_code = antechamber_process.returncode

            with open(mol2_file.replace(".mol2", ".antechamber.log"), "w") as file:
                file.write(f"error code: {antechamber_exit_code}\nstdout:\n\n")
                file.write("stdout:\n\n")
                file.write(antechamber_output.decode())
                file.write("\nstderr:\n\n")
                file.write(antechamber_error.decode())

            if not os.path.isfile(processed_mol2_path):

                raise RuntimeError(
                    f"antechamber failed to assign atom types to the input mol2 file "
                    f"({mol2_file})"
                )

            frcmod_path = mol2_file.replace(".mol2", ".frcmod")
            prmchk2_process = subprocess.Popen(
                [
                    "parmchk2",
                    "-i",
                    processed_mol2_path,
                    "-f",
                    "mol2",
                    "-o",
                    frcmod_path,
                    "-s",
                    gaff_version,
                ],
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
                    f"to the antechamber created mol2 file ({processed_mol2_path})",
                )

        structure = pmd.load_file(processed_mol2_path, structure=True)
        residue_name = np.unique(
            [atom.residue.name for atom in structure.topology.atoms()]
        )[0]

        return processed_mol2_path, frcmod_path, residue_name

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
        host_mol2, host_frcmod, host_resname = self._generate_gaff_atom_types(
            self.host_mol2_path, amber_type, directory
        )
        guest_mol2, guest_frcmod, guest_resname = self._generate_gaff_atom_types(
            self.host_mol2_path, amber_type, directory
        )

        # Build Amber model
        system = TLeap()
        system.output_path = directory
        system.output_prefix = "system"
        system.pbc_type = None
        system.neutralize = False
        system.template_lines = [
            f"source {force_field_source.leap_source}",
            f"loadamberparams {host_frcmod}",
            f"loadamberparams {guest_frcmod}",
            f"{host_resname} = loadmol2 {host_mol2}",
            f"{guest_resname} = loadmol2 {guest_mol2}",
            f"model = loadpdb {self.coordinate_file_path}",
        ]

        # Set GB Radii if running with implicit solvent
        if force_field_source.igb:
            gb_radii = {
                "1": "mbondi",
                "2": "mbondi2",
                "5": "mbondi2",
                "7": "bondi",
                "8": "mbondi3",
            }
            system.template_lines.index(
                1,
                f"set default PBRadii {gb_radii[force_field_source.igb]}",
            )

        system.build(clean_files=False)
        if not is_file_and_not_empty(
            os.path.join(system.output_path, system.output_prefix + ".prmtop")
        ) and not is_file_and_not_empty(
            os.path.join(system.output_path, system.output_prefix + ".rst7")
        ):
            raise RuntimeError("tleap failed to execute.")

        # Create OpenMM XML file
        prmtop_file = app.AmberPrmtopFile(
            os.path.join(system.output_path, system.output_prefix + ".prmtop")
        )
        if force_field_source.igb:
            solvent_model = {
                "1": app.HCT,
                "2": app.OBC1,
                "5": app.OBC2,
                "7": app.GBn,
                "8": app.GBn2,
            }
            system = prmtop_file.createSystem(
                nonbondedMethod=app.NoCutoff,
                constraints=app.HBonds,
                implicitSolvent=solvent_model[force_field_source.igb],
                hydrogenMass=3.024 if self.enable_hmr else None,
            )
        else:
            system = prmtop_file.createSystem(
                nonbondedMethod=app.PME,
                nonbondedCutoff=pint_quantity_to_openmm(force_field_source.cutoff),
                constraints=app.HBonds,
                rigidWater=True,
                removeCMMotion=False,
                hydrogenMass=3.024 if self.enable_hmr else None,
            )

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
