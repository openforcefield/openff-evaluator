"""
A collection of protocols for performing free energy calculations using
the pAPRika software package.
"""
import logging
import os
import os.path
import shutil
import traceback
import typing
from queue import Queue
from subprocess import Popen
from threading import Thread

import numpy as np
import pint
from simtk.openmm import XmlSerializer
from simtk.openmm.app import PME, AmberPrmtopFile, HBonds, PDBFile

from evaluator import unit
from evaluator.attributes import UNDEFINED
from evaluator.backends import ComputeResources
from evaluator.forcefield import (
    ForceFieldSource,
    SmirnoffForceFieldSource,
    TLeapForceFieldSource,
)
from evaluator.protocols import coordinates, forcefield, groups, miscellaneous, openmm
from evaluator.substances import Component, Substance
from evaluator.thermodynamics import Ensemble, ThermodynamicState
from evaluator.utils.exceptions import EvaluatorException
from evaluator.utils.utils import temporarily_change_directory
from evaluator.workflow import Protocol, workflow_protocol
from evaluator.workflow.attributes import (
    InequalityMergeBehaviour,
    InputAttribute,
    OutputAttribute,
)
from evaluator.workflow.utils import ProtocolPath

logger = logging.getLogger(__name__)


@workflow_protocol()
class BasePaprikaProtocol(Protocol):
    """A protocol which will setup and run a pAPRika host-guest
    binding affinity calculation, starting from a host and guest
    `taproom` style .yaml definition file.
    """

    substance = InputAttribute(
        docstring="The substance which defines the host, guest and solvent.",
        type_hint=Substance,
        default_value=UNDEFINED,
    )
    thermodynamic_state = InputAttribute(
        docstring="The thermodynamic conditions to simulate under",
        type_hint=ThermodynamicState,
        default_value=UNDEFINED,
    )

    force_field_path = InputAttribute(
        docstring="A path to the force field to use in the calculation.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    water_model = InputAttribute(
        docstring="The water model to use for the calculation. This is "
                  "temporarily treated as separate from the force field "
                  "until the two are better integrated.",
        type_hint=forcefield.BaseBuildSystem.WaterModel,
        default_value=forcefield.BaseBuildSystem.WaterModel.TIP3P,
    )

    taproom_host_name = InputAttribute(
        docstring="The taproom three letter identifier of the host. This "
                  "is temporary until this protocol is decoupled from taproom.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    taproom_guest_name = InputAttribute(
        docstring="The taproom three letter identifier of the guest. This "
                  "is temporary until this protocol is decoupled from taproom.",
        type_hint=typing.Union[str, None],
        default_value=None,
    )
    taproom_guest_orientation = InputAttribute(
        docstring="The taproom one letter identifier of the orientation of "
                  "the guest. This is temporary until this protocol is decoupled "
                  "from taproom.",
        type_hint=typing.Union[str, None],
        default_value=None,
    )

    timestep = InputAttribute(
        docstring="The timestep to evolve the system by at each step.",
        type_hint=unit.Quantity,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=2.0 * unit.femtosecond,
    )
    thermalisation_timestep = InputAttribute(
        docstring="The timestep to evolve the system during thermalisation "
                  "by at each step.",
        type_hint=unit.Quantity,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=1.0 * unit.femtosecond,
    )

    number_of_thermalisation_steps = InputAttribute(
        docstring="The number of NPT thermalisation steps to take. Data from "
                  "the equilibration simulations will be discarded.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=50000,
    )
    thermalisation_output_frequency = InputAttribute(
        docstring="The frequency with which to write statistics during "
                  "thermalisation. Data from the thermalisation simulations "
                  "will be discarded.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=5000,
    )

    number_of_equilibration_steps = InputAttribute(
        docstring="The number of NPT equilibration steps to take. Data from "
                  "the equilibration simulations will be discarded.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=200000,
    )
    equilibration_output_frequency = InputAttribute(
        docstring="The frequency with which to write statistics during "
                  "equilibration. Data from the equilibration simulations "
                  "will be discarded.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=5000,
    )

    number_of_production_steps = InputAttribute(
        docstring="The number of NPT production steps to take.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=1000000,
    )
    production_output_frequency = InputAttribute(
        docstring="The frequency with which to write statistics during production.",
        type_hint=int,
        merge_behavior=InequalityMergeBehaviour.LargestValue,
        default_value=5000,
    )

    number_of_solvent_molecules = InputAttribute(
        docstring="The number of solvent molecules to solvate the host and guest with.",
        type_hint=int,
        default_value=3000,
    )
    packmol_tolerance = InputAttribute(
        docstring="The distance tolerance for packing molecules in packmol.",
        type_hint=unit.Quantity,
        merge_behavior=InequalityMergeBehaviour.SmallestValue,
        default_value=2.4 * unit.angstrom,
    )
    simulation_box_aspect_ratio = InputAttribute(
        docstring="The aspect ratio of the box. This should be a list of three "
                  "floats, corresponding to the relative length of each side of "
                  "the box.",
        type_hint=list,
        default_value=[1.0, 1.0, 2.0],
    )

    attach_free_energy = OutputAttribute(
        docstring="The free energy of...", type_hint=pint.Measurement
    )
    pull_free_energy = OutputAttribute(
        docstring="The free energy of...", type_hint=pint.Measurement
    )

    release_free_energy = OutputAttribute(
        docstring="The free energy of...", type_hint=pint.Measurement
    )
    symmetry_correction = OutputAttribute(
        docstring="The free energy of...", type_hint=pint.Measurement
    )
    reference_free_energy = OutputAttribute(
        docstring="The free energy of...", type_hint=pint.Measurement
    )

    def __init__(self, protocol_id):
        """Initializes a new BasePaprikaProtocol object.
        """

        super().__init__(protocol_id)

        self._force_field_source = None
        self._paprika_setup = None

        self._solvated_coordinate_paths = {}

        self._results_dictionary = None

        # Useful debug variables for either enabling or disabling parts of
        # this protocol.
        self.setup = True
        self.simulate = True
        self.analyze = True

    def _setup_paprika(self, directory):

        import paprika

        generate_gaff_files = isinstance(
            self._force_field_source, TLeapForceFieldSource
        )

        gaff_version = "gaff2"

        if generate_gaff_files:
            gaff_version = self._force_field_source.leap_source.replace("leaprc.", "")

        self._paprika_setup = paprika.Setup(
            host=self.taproom_host_name,
            guest=self.taproom_guest_name,
            guest_orientation=self.taproom_guest_orientation,
            directory_path=directory,
            generate_gaff_files=generate_gaff_files,
            gaff_version=gaff_version,
        )

    def _solvate_windows(self, directory, available_resources):

        # Extract out only the solvent components of the substance (e.g H2O,
        # Na+, Cl-...)
        filter_solvent = miscellaneous.FilterSubstanceByRole("filter_solvent")
        filter_solvent.input_substance = self.substance
        filter_solvent.component_role = Component.Role.Solvent

        filter_solvent.execute(directory, available_resources)

        reference_structure_path = None

        for index, window_file_path in enumerate(
                self._paprika_setup.desolvated_window_paths
        ):

            window_directory = os.path.dirname(window_file_path)
            os.makedirs(window_directory, exist_ok=True)

            self._solvated_coordinate_paths[index] = os.path.join(
                window_directory, "restrained.pdb"
            )

            if os.path.isfile(self._solvated_coordinate_paths[index]):
                logger.info(
                    f"Skipping the setup of window {index + 1} as "
                    f"{self._solvated_coordinate_paths[index]} already "
                    f"exists."
                )
                continue

            # Solvate the window.
            solvate_complex = coordinates.SolvateExistingStructure("solvate_window")
            solvate_complex.max_molecules = self.number_of_solvent_molecules
            solvate_complex.box_aspect_ratio = self.simulation_box_aspect_ratio
            solvate_complex.tolerance = self.packmol_tolerance
            solvate_complex.center_solute_in_box = False
            if self.number_of_solvent_molecules < 20:
                solvate_complex.mass_density = 0.005 * unit.grams / unit.milliliters
            solvate_complex.substance = filter_solvent.filtered_substance
            solvate_complex.solute_coordinate_file = window_file_path
            solvate_complex.execute(window_directory, available_resources)

            # Store the path to the structure of the first window, which will
            # serve as a reference point when adding the dummy atoms.
            if index == 0:
                reference_structure_path = solvate_complex.coordinate_file_path

            # Add the aligning dummy atoms to the solvated pdb files.
            self._add_dummy_atoms(
                index, solvate_complex.coordinate_file_path, reference_structure_path
            )

            # Extra step to create GAFF1/2 structures properly
            if isinstance(self._force_field_source, TLeapForceFieldSource):

                # Fix atom names of guest molecule for Tleap processing
                if self._paprika_setup.guest is self.taproom_guest_name:
                    import parmed as pmd

                    structure_mol = pmd.load_file(
                        os.path.join(
                            window_directory[:len(window_directory) - 13],
                            f"{self._paprika_setup.guest}.gaff.mol2"
                        )
                    )
                    structure_pdb = pmd.load_file(
                        self._solvated_coordinate_paths[index]
                    )

                    # Get atom names of guest molecule from restrained.pdb and *.gaff.mol2
                    mol_name = []
                    pdb_name = []
                    for original, guest in zip(
                            structure_mol,
                            structure_pdb[f":{self._paprika_setup.guest.upper()}"]
                    ):
                        mol_name.append(original.name)
                        pdb_name.append(guest.name)

                    # Change guest atom names of restrained.pdb to that of *.gaff.mol2
                    fin = open(self._solvated_coordinate_paths[index], "r")
                    pdb_lines = fin.readlines()
                    i_atom = 0
                    i_file = 0
                    for line in pdb_lines:
                        if line.startswith("HETATM"):
                            if line.split()[3].upper() == f"{self._paprika_setup.guest.upper()}":
                                if len(pdb_name[i_atom]) - len(mol_name[i_atom]) == 2:
                                    pdb_lines[i_file] = line.replace(
                                        pdb_name[i_atom],
                                        ' ' + mol_name[i_atom] + ' '
                                    )
                                elif len(pdb_name[i_atom]) - len(mol_name[i_atom]) == 1:
                                    pdb_lines[i_file] = line.replace(
                                        pdb_name[i_atom],
                                        ' ' + mol_name[i_atom]
                                    )
                                elif len(pdb_name[i_atom]) < len(mol_name[i_atom]):
                                    pdb_lines[i_file] = line.replace(
                                        pdb_name[i_atom] + ' ',
                                        mol_name[i_atom]
                                    )
                                else:
                                    pdb_lines[i_file] = line.replace(
                                        pdb_name[i_atom],
                                        mol_name[i_atom]
                                    )
                                i_atom += 1
                        i_file += 1
                    fin.close()

                    # Overwrite restrained.pdb with the correct guest atom names
                    fout = open(self._solvated_coordinate_paths[index], "w")
                    for line in pdb_lines:
                        fout.writelines(line)
                    fout.close()

                # Extract water and ions from restrained.pdb
                import pytraj as pt
                structure = pt.iterload(self._solvated_coordinate_paths[index])
                water_ions_sel = f"!@DUM&!:MGO&!:{self._paprika_setup.guest.upper()}"
                structure[water_ions_sel].save(
                    os.path.join(window_directory, "water_ions.pdb")
                )

                # Create *.mol2 file for water and ions
                from paprika.tleap import System
                system = System()
                system.output_path = window_directory
                system.pbc_type = None
                system.neutralize = False
                system.template_lines = [
                    f"source leaprc.water.tip3p",
                    f"HOH = loadpdb water_ions.pdb",
                    f"savemol2 HOH water_ions.mol2 1",
                    f"quit",
                ]
                system.build()

                # Delete water_ions.pdb
                os.remove(os.path.join(window_directory, "water_ions.pdb"))

            logger.info(
                f"Set up window {index + 1} of "
                f"{len(self._paprika_setup.desolvated_window_paths)}"
            )

    def _add_dummy_atoms(
            self, index, solvated_structure_path, reference_structure_path
    ):

        self._paprika_setup.add_dummy_atoms(
            reference_structure_path,
            solvated_structure_path,
            None,
            self._solvated_coordinate_paths[index],
            None,
        )

    def _apply_restraint_masks(self, use_amber_indices):

        import parmed as pmd
        from paprika.utils import index_from_mask

        for index, window in enumerate(self._paprika_setup.window_list):

            window_directory = os.path.join(
                self._paprika_setup.directory, "windows", window
            )

            build_pdb_file = pmd.load_file(
                f"{window_directory}/build.pdb", structure=True
            )

            for restraint in (
                    self._paprika_setup.static_restraints
                    + self._paprika_setup.conformational_restraints
                    + self._paprika_setup.symmetry_restraints
                    + self._paprika_setup.wall_restraints
                    + self._paprika_setup.guest_restraints
            ):

                restraint.index1 = index_from_mask(
                    build_pdb_file, restraint.mask1, use_amber_indices
                )
                restraint.index2 = index_from_mask(
                    build_pdb_file, restraint.mask2, use_amber_indices
                )
                if restraint.mask3:
                    restraint.index3 = index_from_mask(
                        build_pdb_file, restraint.mask3, use_amber_indices
                    )
                if restraint.mask4:
                    restraint.index4 = index_from_mask(
                        build_pdb_file, restraint.mask4, use_amber_indices
                    )

    def _setup_restraints(self):

        (
            self._paprika_setup.static_restraints,
            self._paprika_setup.conformational_restraints,
            self._paprika_setup.symmetry_restraints,
            self._paprika_setup.wall_restraints,
            self._paprika_setup.guest_restraints,
        ) = self._paprika_setup.initialize_restraints(
            self._solvated_coordinate_paths[0]
        )

    def _apply_parameters(self):

        if not isinstance(self._force_field_source, TLeapForceFieldSource):
            # Due to the OpenFF toolkit's lack of support for dummy particles we
            # assign the SMIRNOFF parameters while adding the dummy particules,
            # so we can skip this step.
            return

        for index, window_file_path in enumerate(
                self._paprika_setup.desolvated_window_paths
        ):
            window_directory = os.path.dirname(window_file_path)
            self._build_amber_parameters(index, window_directory)

    @staticmethod
    def _create_dummy_files(directory):

        dummy_frcmod_lines = [
            "Parameters for dummy atom with type Du\n",
            "MASS\n",
            "Du     208.00\n",
            "\n",
            "BOND\n",
            "\n",
            "ANGLE\n",
            "\n",
            "DIHE\n",
            "\n",
            "IMPROPER\n",
            "\n",
            "NONBON\n",
            "  Du       0.000     0.0000000\n",
        ]

        with open(os.path.join(directory, "dummy.frcmod"), "w") as file:
            file.writelines(dummy_frcmod_lines)

        dummy_mol2_template = (
            "@<TRIPOS>MOLECULE\n"
            "{0:s}\n"
            "    1     0     1     0     1\n"
            "SMALL\n"
            "USER_CHARGES\n"
            "\n"
            "@<TRIPOS>ATOM\n"
            "  1 DUM     0.000000    0.000000    0.000000 Du    1 {0:s}     0.0000 ****\n"
            "@<TRIPOS>BOND\n"
            "@<TRIPOS>SUBSTRUCTURE\n"
            "      1  {0:s}              1 ****               0 ****  ****    0 ROOT\n"
        )

        for dummy_name in ["DM1", "DM2", "DM3"]:
            with open(
                    os.path.join(directory, f"{dummy_name.lower()}.mol2"), "w"
            ) as file:
                file.write(dummy_mol2_template.format(dummy_name))

    def _build_amber_parameters(self, index, window_directory):

        from paprika.tleap import System

        window_directory_to_base = os.path.relpath(
            os.path.abspath(self._paprika_setup.directory), window_directory
        )

        window_coordinates = os.path.relpath(
            self._solvated_coordinate_paths[index], window_directory
        )

        self._create_dummy_files(self._paprika_setup.directory)

        os.makedirs(window_directory, exist_ok=True)

        system = System()
        system.output_path = window_directory
        system.pbc_type = None
        system.neutralize = False

        gaff_version = self._force_field_source.leap_source.replace("leaprc.", "")

        # Host definition
        host_frcmod = os.path.join(
            window_directory_to_base,
            f"{self._paprika_setup.host}.{gaff_version}.frcmod",
        )
        host_mol2 = os.path.join(
            window_directory_to_base,
            f"{self._paprika_setup.host}.{gaff_version}.mol2"
        )

        load_host_frcmod = f"loadamberparams {host_frcmod}"
        load_host_mol2 = (
            f'{self._paprika_setup.host_yaml["resname"].upper()} = loadmol2 {host_mol2}'
        )
        load_host_def = [
            load_host_mol2,
            "set MGO name \"MGO\"",
            "set MGO head MGO.1.C4",
            "set MGO tail MGO.1.O1",
            "set MGO.1 connect0 MGO.1.C4",
            "set MGO.1 connect1 MGO.1.O1",
            "set MGO.1 restype saccharide",
            "set MGO.1 name \"MGO\"",
        ]
        load_host_chain = ""
        model_bond = ""
        if self._paprika_setup.host.lower() == "acd":
            load_host_chain = [
                "ACDOH = sequence {MGO MGO MGO MGO MGO MGO MGO}",
                "set ACDOH head ACDOH.1.C4",
                "set ACDOH tail ACDOH.6.O1",
                "impose ACDOH {1 2 3 4 5 6} {{O5 C1 O1 C4 90.0} {C1 O1 C4 C5 -95.0}}",
                "bond ACDOH.1.C4 ACDOH.6.O1",
            ]
            model_bond = "bond model.1.C4 model.6.O1"

        elif self._paprika_setup.host.lower() == "bcd":
            load_host_chain = [
                "BCDOH = sequence {MGO MGO MGO MGO MGO MGO MGO MGO}",
                "set BCDOH head BCDOH.1.C4",
                "set BCDOH tail BCDOH.7.O1",
                "impose BCDOH {1 2 3 4 5 6 7} {{O5 C1 O1 C4 98.0} {C1 O1 C4 C5 -103.0}}",
                "bond BCDOH.1.C4 BCDOH.7.O1",
            ]
            model_bond = "bond model.1.C4 model.7.O1"

        # Solvent definition
        load_solvent_mol2 = f"SOL = loadmol2 water_ions.mol2"

        # Guest definition
        load_guest_frcmod = ""
        load_guest_mol2 = ""

        if self.taproom_guest_name is not None:
            guest_frcmod = os.path.join(
                window_directory_to_base,
                f"{self._paprika_setup.guest}.{gaff_version}.frcmod",
            )
            guest_mol2 = os.path.join(
                window_directory_to_base,
                f"{self._paprika_setup.guest}.{gaff_version}.mol2",
            )

            load_guest_frcmod = f"loadamberparams {guest_frcmod}"
            load_guest_mol2 = f'{self._paprika_setup.guest_yaml["name"].upper()} = loadmol2 {guest_mol2}'

        # window_pdb_file = PDBFile(self._solvated_coordinate_paths[index])
        # cell_vectors = window_pdb_file.topology.getPeriodicBoxVectors()

        force_field_lines = [
            f"source leaprc.{gaff_version}",
            f"source leaprc.water.tip3p",
            f"source leaprc.protein.ff14SB",
            load_host_frcmod,
            load_guest_frcmod,
            f"loadamberparams {os.path.join(window_directory_to_base, 'dummy.frcmod')}",
        ]

        host_def_lines = load_host_def + load_host_chain

        hetatom_lines = [
            load_guest_mol2,
            load_solvent_mol2,
            f"DM1 = loadmol2 {os.path.join(window_directory_to_base, 'dm1.mol2')}",
            f"DM2 = loadmol2 {os.path.join(window_directory_to_base, 'dm2.mol2')}",
            f"DM3 = loadmol2 {os.path.join(window_directory_to_base, 'dm3.mol2')}",
        ]

        model_lines = [
            f"model = loadpdb {window_coordinates}",
            model_bond,
            f"setBox model \"centers\"",
            "check model",
        ]

        save_structure_lines = [
            "saveamberparm model structure.prmtop structure.rst7",
        ]

        system.template_lines = force_field_lines + \
                                host_def_lines + \
                                hetatom_lines + \
                                model_lines + \
                                save_structure_lines
        system.build()

        # Delete water_ions.mol2
        os.remove(os.path.join(window_directory, "water_ions.mol2"))

    def _run_windows(self, available_resources):

        # Create the queue which will pass the run arguments to the created
        # threads.
        queue = Queue(maxsize=0)
        chunk_size = max(1, available_resources.number_of_gpus)

        # Start the threads.
        for _ in range(chunk_size):
            worker = Thread(target=self._run_window, args=(queue,))
            worker.setDaemon(True)
            worker.start()

        exceptions = []

        window_indices = [
            index for index in range(len(self._paprika_setup.window_list))
        ]

        # Determine how many 'chunks' to break the full window list into depending
        # on the available compute resources.
        full_multiples = int(np.floor(len(window_indices) / chunk_size))
        chunks = [
                     [i * chunk_size, (i + 1) * chunk_size] for i in range(full_multiples)
                 ] + [[full_multiples * chunk_size, len(window_indices)]]

        counter = 0

        for chunk in chunks:

            for window_index in sorted(window_indices)[chunk[0]: chunk[1]]:

                logger.info(
                    f"Running window {window_index + 1} out of {len(self._paprika_setup.window_list)}"
                )
                resources = ComputeResources(number_of_threads=1)

                if available_resources.number_of_gpus > 0:
                    resources = ComputeResources(
                        number_of_threads=1,
                        number_of_gpus=1,
                        preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA,
                    )
                    resources._gpu_device_indices = f"{counter}"

                self._enqueue_window(queue, window_index, resources, exceptions)

                counter += 1

                if counter == chunk_size:

                    queue.join()
                    counter = 0

                    if len(exceptions) > 0:
                        message = ", ".join(
                            [f"{exception.message}" for exception in exceptions]
                        )
                        raise RuntimeError(message)

        if not queue.empty():
            queue.join()

        if len(exceptions) > 0:
            message = ", ".join([f"{exception.message}" for exception in exceptions])
            raise RuntimeError(message)

        return None

    def _enqueue_window(self, queue, index, available_resources, exceptions):
        raise NotImplementedError()

    @staticmethod
    def _run_window(queue):
        raise NotImplementedError()

    def _perform_analysis(self, directory):

        if self._results_dictionary is None:
            raise ValueError("The results dictionary is empty.")

        if "attach" in self._results_dictionary:
            self.attach_free_energy = unit.Measurement(
                -self._results_dictionary["attach"]["ti-block"]["fe"]
                * unit.kilocalorie
                / unit.mole,
                self._results_dictionary["attach"]["ti-block"]["sem"]
                * unit.kilocalorie
                / unit.mole,
            )

        if "pull" in self._results_dictionary:
            self.pull_free_energy = unit.Measurement(
                -self._results_dictionary["pull"]["ti-block"]["fe"]
                * unit.kilocalorie
                / unit.mole,
                self._results_dictionary["pull"]["ti-block"]["sem"]
                * unit.kilocalorie
                / unit.mole,
            )

        if "release" in self._results_dictionary:
            self.release_free_energy = unit.Measurement(
                self._results_dictionary["release"]["ti-block"]["fe"]
                * unit.kilocalorie
                / unit.mole,
                self._results_dictionary["release"]["ti-block"]["sem"]
                * unit.kilocalorie
                / unit.mole,
            )

        if "ref_state_work" in self._results_dictionary:
            self.reference_free_energy = unit.Measurement(
                -self._results_dictionary["ref_state_work"]
                * unit.kilocalorie
                / unit.mole,
                0 * unit.kilocalorie / unit.mole,
            )

        if "symmetry_correction" in self._results_dictionary:
            self.symmetry_correction = unit.Measurement(
                self._results_dictionary["symmetry_correction"]
                * unit.kilocalorie
                / unit.mole,
                0 * unit.kilocalorie / unit.mole,
            )

        return None

    def _setup(self, directory, available_resources):

        from paprika.io import save_restraints

        # Create a new setup object which will load in a pAPRika host
        # and guest yaml file, setup a directory structure for the
        # paprika calculations, and create a set of coordinates for
        # each of the windows along the pathway (without any solvent).
        self._setup_paprika(directory)

        # Define where the final restraints definition file should be written
        restraints_path = os.path.join(self._paprika_setup.directory, "restraints.json")

        if os.path.isfile(restraints_path):
            # We can skip setup if the restraints file already exists as this is the
            # last step of setup.
            return

        # Solvate each of the structures along the calculation path.
        self._solvate_windows(directory, available_resources)

        if len(self._solvated_coordinate_paths) == 0:
            raise RuntimeError(
                "There were no defined windows to a/p/r the guest along.",
            )

        # Apply parameters to each of the windows.
        self._apply_parameters()

        # Setup the actual restraints.
        self._setup_restraints()

        # Save the restraints to a file, ready for analysis.
        save_restraints(
            restraint_list=self._paprika_setup.static_restraints
                           + self._paprika_setup.conformational_restraints
                           + self._paprika_setup.symmetry_restraints
                           + self._paprika_setup.wall_restraints
                           + self._paprika_setup.guest_restraints,
            filepath=restraints_path,
        )

    def _simulate(self, directory, available_resources):

        import paprika

        if not self._paprika_setup:
            self._paprika_setup = paprika.setup(
                host=self.taproom_host_name,
                guest=self.taproom_guest_name,
                guest_orientation=self.taproom_guest_orientation,
                build=False,
                directory_path=directory,
            )

            base_path = os.path.join(
                directory,
                self._paprika_setup.host,
                f"{self._paprika_setup.guest}-{self.taproom_guest_orientation}"
                if self._paprika_setup.guest
                else "",
                "windows",
            )

            window_directories = [
                os.path.join(base_path, window)
                for window in self._paprika_setup.window_list
            ]

        else:
            window_directories = [
                os.path.dirname(window_path)
                for window_path in self._paprika_setup.desolvated_window_paths
            ]

        for index, window_directory in enumerate(window_directories):

            self._solvated_coordinate_paths[index] = os.path.join(
                window_directory, "restrained.pdb"
            )
            self._solvated_system_xml_paths[index] = os.path.join(
                window_directory, "restrained.xml"
            )

            if not os.path.isfile(self._solvated_coordinate_paths[index]):
                raise RuntimeError(
                    f"The {self._solvated_coordinate_paths[index]} file "
                    f"does not exist. Make sure setup ran successfully.",
                )

            if not os.path.isfile(self._solvated_system_xml_paths[index]):
                raise RuntimeError(
                    f"The {self._solvated_system_xml_paths[index]} file "
                    f"does not exist. Make sure setup ran successfully.",
                )

        # Run the simulations
        self._run_windows(available_resources)

    def _analyse(self, directory):

        import paprika

        if not self._paprika_setup:
            self._paprika_setup = paprika.setup(
                host=self.taproom_host_name,
                guest=self.taproom_guest_name,
                guest_orientation=self.taproom_guest_orientation,
                build=False,
                directory_path=directory,
            )

        # Finally, do the analysis to extract the free energy of binding.
        self._perform_analysis(directory)

    def _execute(self, directory, available_resources):

        # Make sure the available resources are commensurate with the
        # implemented parallelisation scheme.
        if (
                available_resources.number_of_gpus > 0
                and available_resources.number_of_gpus
                != available_resources.number_of_threads
        ):
            raise RuntimeError(
                "The number of available CPUs must match the number"
                "of available GPUs for this parallelisation scheme.",
            )

        # Load in the force field to use.
        with open(self.force_field_path) as file:
            self._force_field_source = ForceFieldSource.parse_json(file.read())

        if not isinstance(
                self._force_field_source, SmirnoffForceFieldSource
        ) and not isinstance(self._force_field_source, TLeapForceFieldSource):
            raise RuntimeError(
                "Only SMIRNOFF and TLeap based force fields may "
                "be used with this protocol.",
            )

        with temporarily_change_directory(directory):

            original_force_field_path = self.force_field_path
            self.force_field_path = os.path.relpath(
                original_force_field_path, directory
            )

            if self.setup:
                self._setup("", available_resources)

            if self.simulate:
                self._simulate("", available_resources)

            if self.analyze:
                self._analyse("")

            self.force_field_path = original_force_field_path


@workflow_protocol()
class OpenMMPaprikaProtocol(BasePaprikaProtocol):
    """A protocol which will setup and run a pAPRika host-guest
    binding affinity calculation using OpenMM, starting from a
    host and guest `taproom` style .yaml definition file.
    """

    def __init__(self, protocol_id):
        super().__init__(protocol_id)
        self._solvated_system_xml_paths = {}

    def _add_dummy_atoms(
            self, index, solvated_structure_path, reference_structure_path
    ):

        # We pull the host charges from the specified mol2 file.
        host_mol2_path = str(
            self._paprika_setup.benchmark_path.joinpath(
                self._paprika_setup.host_yaml["structure"]
            )
        )

        window_directory = os.path.dirname(solvated_structure_path)

        unrestrained_xml_path = None
        self._solvated_system_xml_paths[index] = os.path.join(
            window_directory, "restrained.xml"
        )

        if isinstance(self._force_field_source, SmirnoffForceFieldSource):
            # Assign force field parameters to the solvated complex system.
            # Because the openforcefield toolkit does not yet support dummy atoms,
            # we have to assign the smirnoff parameters before adding the dummy atoms.
            # Hence this specialised method.
            build_solvated_complex_system = forcefield.BuildSmirnoffSystem(
                "build_solvated_window_system"
            )
            build_solvated_complex_system.force_field_path = self.force_field_path
            build_solvated_complex_system.coordinate_file_path = solvated_structure_path
            build_solvated_complex_system.substance = self.substance
            build_solvated_complex_system.charged_molecule_paths = [host_mol2_path]
            build_solvated_complex_system.execute(window_directory)

            unrestrained_xml_path = build_solvated_complex_system.system_path

        self._paprika_setup.add_dummy_atoms(
            reference_structure_path,
            solvated_structure_path,
            unrestrained_xml_path,
            self._solvated_coordinate_paths[index],
            self._solvated_system_xml_paths[index],
        )

    def _apply_parameters(self):

        from simtk import unit as simtk_unit

        super(OpenMMPaprikaProtocol, self)._apply_parameters()

        if not isinstance(self._force_field_source, TLeapForceFieldSource):
            return

        # Convert the amber files to OMM system objects.
        for index in range(len(self._paprika_setup.window_list)):

            window_directory = os.path.dirname(self._solvated_system_xml_paths[index])

            # Make sure to use the reordered pdb file as the new input
            shutil.copyfile(
                os.path.join(window_directory, "build.pdb"),
                self._solvated_coordinate_paths[index],
            )

            pdb_file = PDBFile(self._solvated_coordinate_paths[index])
            cell_vectors = pdb_file.topology.getPeriodicBoxVectors()

            new_positions = []

            for position in pdb_file.positions:
                position += cell_vectors[0] / 2.0
                position += cell_vectors[1] / 2.0
                position += cell_vectors[2] / 2.0

                new_positions.append(position.value_in_unit(simtk_unit.angstrom))

            with open(self._solvated_coordinate_paths[index], "w+") as file:

                PDBFile.writeFile(
                    pdb_file.topology,
                    new_positions * simtk_unit.angstrom,
                    file,
                    keepIds=True,
                )

            prmtop = AmberPrmtopFile(os.path.join(window_directory, "structure.prmtop"))

            cutoff = (
                    self._force_field_source.cutoff.to(unit.angstrom).magnitude
                    * simtk_unit.angstrom
            )

            system = prmtop.createSystem(
                nonbondedMethod=PME,
                nonbondedCutoff=cutoff,
                constraints=HBonds,
                removeCMMotion=False,
            )

            system_xml = XmlSerializer.serialize(system)

            with open(self._solvated_system_xml_paths[index], "wb") as file:
                file.write(system_xml.encode("utf-8"))

    def _setup_restraints(self):

        super(OpenMMPaprikaProtocol, self)._setup_restraints()

        if isinstance(self._force_field_source, TLeapForceFieldSource):
            self._apply_restraint_masks(use_amber_indices=False)

        # Apply the restraint forces to the solvated system xml files.
        for index, window in enumerate(self._paprika_setup.window_list):
            self._paprika_setup.initialize_calculation(
                window,
                self._solvated_coordinate_paths[index],
                self._solvated_system_xml_paths[index],
                self._solvated_system_xml_paths[index],
            )

    def _enqueue_window(self, queue, index, available_resources, exceptions):

        queue.put(
            (
                index,
                self._solvated_coordinate_paths[index],
                self._solvated_system_xml_paths[index],
                self.thermodynamic_state,
                self.timestep,
                self.thermalisation_timestep,
                self.number_of_thermalisation_steps,
                self.thermalisation_output_frequency,
                self.number_of_equilibration_steps,
                self.equilibration_output_frequency,
                self.number_of_production_steps,
                self.production_output_frequency,
                available_resources,
                exceptions,
            )
        )

    @staticmethod
    def _run_window(queue):

        while True:

            (
                index,
                window_coordinate_path,
                window_system_path,
                thermodynamic_state,
                timestep,
                thermalisation_timestep,
                number_of_thermalisation_steps,
                thermalisation_output_frequency,
                number_of_equilibration_steps,
                equilibration_output_frequency,
                number_of_production_steps,
                production_output_frequency,
                available_resources,
                exceptions,
            ) = queue.get()

            try:

                window_directory = os.path.dirname(window_system_path)

                final_trajectory_path = os.path.join(window_directory, "trajectory.dcd")
                final_topology_path = os.path.join(window_directory, "input.pdb")

                if os.path.isfile(final_trajectory_path) and os.path.isfile(
                        final_topology_path
                ):
                    queue.task_done()
                    continue

                if (
                        os.path.isfile(final_trajectory_path)
                        and not os.path.isfile(final_topology_path)
                ) or (
                        not os.path.isfile(final_trajectory_path)
                        and os.path.isfile(final_topology_path)
                ):
                    exceptions.append(
                        EvaluatorException(
                            f"This window either has a trajectory but "
                            f"not topology pdb file, or does not have a "
                            f"trajectory but does have a topology pdb "
                            f"file. This should not happen",
                        )
                    )

                    queue.task_done()
                    continue

                simulation_directory = os.path.join(window_directory, "simulations")
                os.makedirs(simulation_directory, exist_ok=True)

                # Minimisation
                energy_minimisation = openmm.OpenMMEnergyMinimisation(
                    "energy_minimisation"
                )
                energy_minimisation.input_coordinate_file = window_coordinate_path
                energy_minimisation.system_path = window_system_path

                # Thermalisation
                npt_thermalisation = openmm.OpenMMSimulation("npt_thermalisation")
                npt_thermalisation.steps_per_iteration = number_of_thermalisation_steps
                npt_thermalisation.output_frequency = thermalisation_output_frequency
                npt_thermalisation.timestep = thermalisation_timestep
                npt_thermalisation.ensemble = Ensemble.NPT
                npt_thermalisation.thermodynamic_state = thermodynamic_state
                npt_thermalisation.system_path = window_system_path
                npt_thermalisation.input_coordinate_file = ProtocolPath(
                    "output_coordinate_file", energy_minimisation.id
                )

                # Equilibration
                npt_equilibration = openmm.OpenMMSimulation("npt_equilibration")
                npt_equilibration.steps_per_iteration = number_of_equilibration_steps
                npt_equilibration.output_frequency = equilibration_output_frequency
                npt_equilibration.timestep = timestep
                npt_equilibration.ensemble = Ensemble.NPT
                npt_equilibration.thermodynamic_state = thermodynamic_state
                npt_equilibration.system_path = window_system_path
                npt_equilibration.input_coordinate_file = ProtocolPath(
                    "output_coordinate_file", npt_thermalisation.id
                )

                # Production
                npt_production = openmm.OpenMMSimulation("npt_production")
                npt_production.steps_per_iteration = number_of_production_steps
                npt_production.output_frequency = production_output_frequency
                npt_production.timestep = timestep
                npt_production.ensemble = Ensemble.NPT
                npt_production.thermodynamic_state = thermodynamic_state
                npt_production.system_path = window_system_path
                npt_production.input_coordinate_file = ProtocolPath(
                    "output_coordinate_file", npt_equilibration.id
                )

                simulation_protocol = groups.ProtocolGroup(f"simulation_{index}")
                simulation_protocol.add_protocols(
                    energy_minimisation, npt_thermalisation, npt_equilibration, npt_production
                )

                simulation_protocol.execute(simulation_directory, available_resources)

                trajectory_path = simulation_protocol.get_value(
                    ProtocolPath("trajectory_file_path", "npt_production")
                )
                coordinate_path = simulation_protocol.get_value(
                    ProtocolPath("output_coordinate_file", "npt_equilibration")
                )

                shutil.move(trajectory_path, final_trajectory_path)
                shutil.move(coordinate_path, final_topology_path)

                # shutil.rmtree(simulation_directory)

            except Exception as e:

                formatted_exception = traceback.format_exception(
                    None, e, e.__traceback__
                )

                exceptions.append(
                    EvaluatorException(
                        message=f"An uncaught exception was raised: "
                                f"{formatted_exception}",
                    )
                )

            queue.task_done()

    def _perform_analysis(self, directory):

        import paprika

        self._results_dictionary = paprika.analyze(
            host=self._paprika_setup.host,
            guest=self._paprika_setup.guest,
            guest_orientation=self.taproom_guest_orientation,
            topology_file="restrained.pdb",
            trajectory_mask="trajectory.dcd",
            directory_path=directory,
            guest_residue_name=self._paprika_setup.guest_yaml["name"]
            if self._paprika_setup.guest != "release"
            else None,
        ).results

        super(OpenMMPaprikaProtocol, self)._perform_analysis(directory)


@workflow_protocol()
class AmberPaprikaProtocol(BasePaprikaProtocol):
    """A protocol which will setup and run a pAPRika host-guest
    binding affinity calculation using Amber, starting from a
    host and guest `taproom` style .yaml definition file.
    """

    def __init__(self, protocol_id):
        super().__init__(protocol_id)
        self._solvated_system_top_paths = {}
        self._force_field_source = None
        self.gaff_cutoff = None

    def _setup_restraints(self):

        from paprika.restraints import amber

        super(AmberPaprikaProtocol, self)._setup_restraints()

        if isinstance(self._force_field_source, TLeapForceFieldSource):
            # Apply the restraint masks which will re-map the restraint indices
            # to the correct atoms (tleap re-orders the packmol file). All indices
            # Get set to index+1 here ready for creating the disang files.
            self._apply_restraint_masks(use_amber_indices=True)

        for index, window in enumerate(self._paprika_setup.window_list):

            window_directory = os.path.join(
                self._paprika_setup.directory, "windows", window
            )

            with open(f"{window_directory}/disang.rest", "w") as file:

                value = ""

                for restraint in (
                        self._paprika_setup.static_restraints
                        + self._paprika_setup.conformational_restraints
                        + self._paprika_setup.wall_restraints
                        + self._paprika_setup.guest_restraints
                ):
                    value += amber.amber_restraint_line(restraint, window)

                file.write(value)

        if isinstance(self._force_field_source, TLeapForceFieldSource):
            # Undo the amber index ready for saving the restraints
            # JSON file.
            self._apply_restraint_masks(use_amber_indices=False)

    def _enqueue_window(self, queue, index, available_resources, exceptions):

        queue.put(
            (
                index,
                self._solvated_coordinate_paths[index],
                None,
                self.thermodynamic_state,
                self.gaff_cutoff,
                self.timestep,
                self.thermalisation_timestep,
                self.number_of_thermalisation_steps,
                self.thermalisation_output_frequency,
                self.number_of_equilibration_steps,
                self.equilibration_output_frequency,
                self.number_of_production_steps,
                self.production_output_frequency,
                available_resources,
                exceptions,
            )
        )

    @staticmethod
    def _run_window(queue):

        from paprika.amber import Simulation

        while True:

            (
                index,
                window_coordinate_path,
                window_system_path,
                thermodynamic_state,
                gaff_cutoff,
                timestep,
                thermalisation_timestep,
                number_of_thermalisation_steps,
                thermalisation_output_frequency,
                number_of_equilibration_steps,
                equilibration_output_frequency,
                number_of_production_steps,
                production_output_frequency,
                available_resources,
                exceptions,
            ) = queue.get()

            window_directory = os.path.dirname(window_coordinate_path)

            environment = os.environ.copy()

            if available_resources.number_of_gpus < 1:
                exceptions.append(
                    EvaluatorException(
                        message="Currently Amber may only be run" "on GPUs",
                    )
                )

                queue.task_done()
                continue

            if (
                    available_resources.preferred_gpu_toolkit
                    != ComputeResources.GPUToolkit.CUDA
            ):
                raise ValueError("Paprika can only be ran either on CPUs or CUDA GPUs.")

            devices_split = [
                int(index.strip())
                for index in os.environ["CUDA_VISIBLE_DEVICES"].split(",")
            ]

            if len(available_resources.gpu_device_indices) > len(devices_split):
                raise ValueError(
                    f"The number of requested GPUs "
                    f"({len(available_resources.gpu_device_indices)}) "
                    f"is greater than the number available "
                    f"({len(devices_split)})"
                )

            requested_split = [
                int(index.strip())
                for index in available_resources.gpu_device_indices.split(",")
            ]
            visible_devices = [str(devices_split[index]) for index in requested_split]

            devices_string = ",".join(visible_devices)
            environment["CUDA_VISIBLE_DEVICES"] = f"{devices_string}"

            logger.info(f"Starting a set of Amber simulations on GPUs {devices_string}")

            simulation_directory = os.path.join(window_directory, "simulations")
            os.makedirs(simulation_directory, exist_ok=True)

            # Minimisation
            minimise_directory = os.path.join(simulation_directory, "energy_minimisation")
            topology_directory = os.path.relpath(window_directory, minimise_directory)
            os.makedirs(minimise_directory, exist_ok=True)

            amber_simulation = Simulation()
            amber_simulation.executable = "pmemd"

            amber_simulation.path = f"{minimise_directory}/"
            amber_simulation.prefix = "minimise"

            amber_simulation.inpcrd = f"{topology_directory}/structure.rst7"
            amber_simulation.ref = f"{topology_directory}/structure.rst7"
            amber_simulation.topology = f"{topology_directory}/structure.prmtop"
            amber_simulation.restraint_file = f"{topology_directory}/disang.rest"

            amber_simulation.config_pbc_min()

            amber_simulation.cntrl["ntf"] = 2
            amber_simulation.cntrl["ntc"] = 2
            amber_simulation.cntrl["maxcyc"] = 5000
            amber_simulation.cntrl["ncyc"] = 1000
            amber_simulation.cntrl["ntr"] = 1
            amber_simulation.cntrl["restraint_wt"] = 50.0
            amber_simulation.cntrl["restraintmask"] = "'@DUM'"

            logging.info(f'Running minimisation with {amber_simulation.executable}')
            amber_simulation.run()

            # Thermalisation
            thermalisation_directory = os.path.join(simulation_directory, "npt_thermalisation")
            input_directory = os.path.relpath(minimise_directory, thermalisation_directory)
            os.makedirs(thermalisation_directory, exist_ok=True)

            amber_simulation = Simulation()
            amber_simulation.executable = "pmemd.cuda -AllowSmallBox"

            amber_simulation.path = f"{thermalisation_directory}/"
            amber_simulation.prefix = "thermalisation"

            amber_simulation.inpcrd = f"{input_directory}/minimise.rst7"
            amber_simulation.ref = f"{topology_directory}/structure.rst7"
            amber_simulation.topology = f"{topology_directory}/structure.prmtop"
            amber_simulation.restraint_file = f"{topology_directory}/disang.rest"

            amber_simulation.config_pbc_md()
            amber_simulation.cntrl["ntr"] = 1
            amber_simulation.cntrl["restraint_wt"] = 50.0
            amber_simulation.cntrl["restraintmask"] = "'@DUM'"
            amber_simulation.cntrl["dt"] = thermalisation_timestep.to(
                unit.picoseconds
            ).magnitude
            amber_simulation.cntrl["cut"] = gaff_cutoff.to(unit.angstrom).magnitude + 1.0  # necessary to relax box
            amber_simulation.cntrl["nstlim"] = number_of_thermalisation_steps
            amber_simulation.cntrl["ntpr"] = thermalisation_output_frequency
            amber_simulation.cntrl["ntwx"] = thermalisation_output_frequency
            amber_simulation.cntrl["ntwe"] = thermalisation_output_frequency
            amber_simulation.cntrl["ntwr"] = thermalisation_output_frequency
            amber_simulation.cntrl["barostat"] = 2

            logging.info(f'Running thermalisation with {amber_simulation.executable}')
            amber_simulation.run()

            # Equilibration
            equilibration_directory = os.path.join(simulation_directory, "npt_equilibration")
            input_directory = os.path.relpath(thermalisation_directory, equilibration_directory)
            os.makedirs(equilibration_directory, exist_ok=True)

            amber_simulation = Simulation()
            amber_simulation.executable = "pmemd.cuda"

            amber_simulation.path = f"{equilibration_directory}/"
            amber_simulation.prefix = "equilibration"

            amber_simulation.inpcrd = f"{input_directory}/thermalisation.rst7"
            amber_simulation.ref = f"{topology_directory}/structure.rst7"
            amber_simulation.topology = f"{topology_directory}/structure.prmtop"
            amber_simulation.restraint_file = f"{topology_directory}/disang.rest"

            amber_simulation.config_pbc_md()
            amber_simulation.cntrl["ntr"] = 1
            amber_simulation.cntrl["restraint_wt"] = 50.0
            amber_simulation.cntrl["restraintmask"] = "'@DUM'"
            amber_simulation.cntrl["dt"] = timestep.to(unit.picoseconds).magnitude
            amber_simulation.cntrl["cut"] = gaff_cutoff.to(unit.angstrom).magnitude
            amber_simulation.cntrl["nstlim"] = number_of_equilibration_steps
            amber_simulation.cntrl["ntpr"] = equilibration_output_frequency
            amber_simulation.cntrl["ntwx"] = equilibration_output_frequency
            amber_simulation.cntrl["ntwe"] = equilibration_output_frequency
            amber_simulation.cntrl["ntwr"] = equilibration_output_frequency
            amber_simulation.cntrl["barostat"] = 2

            logging.info(f'Running equilibration with {amber_simulation.executable}')
            amber_simulation.run()

            # Production
            production_directory = os.path.join(simulation_directory, "npt_production")
            input_directory = os.path.relpath(equilibration_directory, production_directory)
            os.makedirs(production_directory, exist_ok=True)

            amber_simulation = Simulation()
            amber_simulation.executable = "pmemd.cuda"

            amber_simulation.path = f"{production_directory}/"
            amber_simulation.prefix = "production"

            amber_simulation.inpcrd = f"{input_directory}/equilibration.rst7"
            amber_simulation.ref = f"{topology_directory}/structure.rst7"
            amber_simulation.topology = f"{topology_directory}/structure.prmtop"
            amber_simulation.restraint_file = f"{topology_directory}/disang.rest"

            amber_simulation.config_pbc_md()
            amber_simulation.cntrl["ntr"] = 1
            amber_simulation.cntrl["restraint_wt"] = 50.0
            amber_simulation.cntrl["restraintmask"] = "'@DUM'"
            amber_simulation.cntrl["dt"] = timestep.to(unit.picoseconds).magnitude
            amber_simulation.cntrl["cut"] = gaff_cutoff.to(unit.angstrom).magnitude
            amber_simulation.cntrl["nstlim"] = number_of_production_steps
            amber_simulation.cntrl["ntpr"] = production_output_frequency
            amber_simulation.cntrl["ntwx"] = production_output_frequency
            amber_simulation.cntrl["ntwe"] = production_output_frequency
            amber_simulation.cntrl["ntwr"] = production_output_frequency
            amber_simulation.cntrl["barostat"] = 2

            logging.info(f'Running production with {amber_simulation.executable}')
            amber_simulation.run()

            # Clean up files
            trajectory_path = os.path.join(production_directory, "production.nc")
            final_trajectory_path = os.path.join(window_directory, "production.nc")

            shutil.move(trajectory_path, final_trajectory_path)
            # shutil.rmtree(simulation_directory)

            queue.task_done()

    def _simulate(self, directory, available_resources):
        import paprika

        if not self._paprika_setup:
            self._paprika_setup = paprika.setup(
                host=self.taproom_host_name,
                guest=self.taproom_guest_name,
                guest_orientation=self.taproom_guest_orientation,
                build=False,
                directory_path=directory
            )

            base_path = os.path.join(
                directory,
                self._paprika_setup.host,
                f"{self._paprika_setup.guest}-{self.taproom_guest_orientation}"
                if self._paprika_setup.guest else "",
                'windows'
            )

            window_directories = [
                os.path.join(base_path, window) for window in self._paprika_setup.window_list
            ]

        else:
            window_directories = [
                os.path.dirname(window) for window in self._paprika_setup.desolvated_window_paths
            ]

        for index, window_directory in enumerate(window_directories):
            self._solvated_coordinate_paths[index] = os.path.join(window_directory, "structure.rst7")
            self._solvated_system_top_paths[index] = os.path.join(window_directory, "structure.prmtop")

            if not os.path.isfile(self._solvated_system_top_paths[index]):
                return EvaluatorException(
                    directory,
                    f"The {self._solvated_coordinate_paths[index]} file"
                    f"does not exist. Make sure setup ran succesfully"
                )

            if not os.path.isfile(self._solvated_system_top_paths[index]):
                return EvaluatorException(
                    directory,
                    f"The {self._solvated_system_top_paths[index]} file"
                    f"does not exist. Make sure setup ran successfully"
                )

        self._run_windows(available_resources)

    def _perform_analysis(self, directory):

        import paprika

        self._results_dictionary = paprika.analyze(
            host=self._paprika_setup.host,
            guest=self._paprika_setup.guest,
            guest_orientation=self.taproom_guest_orientation,
            topology_file="structure.prmtop",
            trajectory_mask="production.nc",
            directory_path=directory,
            guest_residue_name=self._paprika_setup.guest_yaml["name"]
            if self._paprika_setup.guest != "release"
            else None,
        ).results

        super(AmberPaprikaProtocol, self)._perform_analysis(directory)

    def _execute(self, directory, available_resources):

        with open(self.force_field_path) as file:
            self._force_field_source = ForceFieldSource.parse_json(file.read())

        if not isinstance(
                self._force_field_source, TLeapForceFieldSource
        ) or self._force_field_source.leap_source not in [
            "leaprc.gaff",
            "leaprc.gaff2",
        ]:
            raise RuntimeError(
                message="Currently GAFF(1/2) are the only force fields "
                        "supported with the AmberPaprikaProtocol.",
            )

        self.gaff_cutoff = self._force_field_source.cutoff

        super(AmberPaprikaProtocol, self)._execute(directory, available_resources)
