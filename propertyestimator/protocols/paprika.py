"""
A collection of protocols for performing free energy calculations using
the pAPRika software package.

TODO: Add checkpointing.
"""
import logging
import os
import shutil
import traceback
from enum import Enum
from queue import Queue
from subprocess import Popen
from threading import Thread

import numpy as np
import parmed as pmd
import paprika
from paprika.amber import Simulation
from paprika.io import save_restraints
from paprika.restraints import amber
from paprika.tleap import System
from paprika.utils import index_from_mask
from simtk.openmm import XmlSerializer
from simtk.openmm.app import AmberPrmtopFile, HBonds, PME, PDBFile
from simtk.unit import angstrom

from propertyestimator import unit
from propertyestimator.backends import ComputeResources
from propertyestimator.protocols import miscellaneous, coordinates, forcefield, simulation, groups
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState, Ensemble
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow.decorators import protocol_input, MergeBehaviour, protocol_output
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol
from propertyestimator.workflow.utils import ProtocolPath


@register_calculation_protocol()
class BasePaprikaProtocol(BaseProtocol):
    """A protocol which will setup and run a pAPRika host-guest
    binding affinity calculation, starting from a host and guest
    `taproom` style .yaml definition file.
    """

    class WaterModel(Enum):
        TIP3P = 'TIP3P'

    class ForceField(Enum):
        SMIRNOFF = 'SMIRNOFF'
        GAFF = 'GAFF'
        GAFF2 = 'GAFF2'

    @protocol_input(Substance)
    def substance(self):
        """The substance which defines the host, guest and solvent."""
        pass

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self):
        """The state at which to run the calculations."""
        pass

    @protocol_input(ForceField)
    def force_field(self):
        pass

    @protocol_input(str)
    def force_field_path(self):
        pass

    @protocol_input(WaterModel)
    def water_model(self):
        pass

    @protocol_input(str)
    def taproom_host_name(self):
        pass

    @protocol_input(str)
    def taproom_guest_name(self):
        pass

    @protocol_input(str)
    def taproom_guest_orientation(self):
        pass

    @protocol_input(unit.Quantity)
    def gaff_cutoff(self):
        pass

    @protocol_input(unit.Quantity)
    def timestep(self):
        pass

    @protocol_input(int, merge_behavior=MergeBehaviour.GreatestValue)
    def number_of_equilibration_steps(self):
        pass

    @protocol_input(int, merge_behavior=MergeBehaviour.GreatestValue)
    def equilibration_output_frequency(self):
        pass

    @protocol_input(int, merge_behavior=MergeBehaviour.GreatestValue)
    def number_of_production_steps(self):
        pass

    @protocol_input(int, merge_behavior=MergeBehaviour.GreatestValue)
    def production_output_frequency(self):
        pass

    @protocol_input(int)
    def number_of_solvent_molecules(self):
        pass

    @protocol_input(list)
    def simulation_box_aspect_ratio(self):
        pass

    @protocol_input(bool)
    def setup(self):
        """Whether to run prepare the structures, apply the restraints, and solvate the windows."""
        pass

    @protocol_input(bool)
    def simulate(self):
        """Whether to simulate the windows."""
        pass

    @protocol_input(bool)
    def analyze(self):
        """Whether to analyze the results of the simulation."""
        pass

    @protocol_input(bool)
    def debug(self):
        """ Whether to run in debug mode."""
        pass

    @protocol_output(EstimatedQuantity)
    def attach_free_energy(self):
        pass

    @protocol_output(EstimatedQuantity)
    def pull_free_energy(self):
        pass

    @protocol_output(EstimatedQuantity)
    def release_free_energy(self):
        pass

    @protocol_output(EstimatedQuantity)
    def symmetry_correction(self):
        pass

    @protocol_output(EstimatedQuantity)
    def reference_free_energy(self):
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        self._debug = False

        # Protocol inputs / outputs
        self._substance = None
        self._thermodynamic_state = None

        self._force_field = self.ForceField.SMIRNOFF
        self._force_field_path = ''
        self._water_model = self.WaterModel.TIP3P

        self._taproom_host_name = None
        self._taproom_guest_name = None
        self._taproom_guest_orientation = None

        self._gaff_cutoff = 0.8 * unit.nanometer
        self._timestep = 1.0 * unit.femtosecond

        self._number_of_equilibration_steps = 200000
        self._equilibration_output_frequency = 5000

        self._number_of_production_steps = 1000000
        self._production_output_frequency = 5000

        self._number_of_solvent_molecules = 2000

        self._simulation_box_aspect_ratio = [1.0, 1.0, 2.0]

        self._attach_free_energy = None
        self._pull_free_energy = None
        self._release_free_energy = None
        self._reference_free_energy = None
        self._symmetry_correction = None

        self._paprika_setup = None

        self._solvated_coordinate_paths = {}
        self._setup = True
        self._simulate = True
        self._analyze = True
        self._results_dictionary = None

    def _setup_paprika(self, directory):

        generate_gaff_files = self._force_field == self.ForceField.GAFF or self._force_field == self.ForceField.GAFF2
        gaff_version = 'gaff2'

        if self._force_field == self.ForceField.GAFF:
            gaff_version = 'gaff'

        self._paprika_setup = paprika.Setup(host=self._taproom_host_name,
                                            guest=self._taproom_guest_name,
                                            guest_orientation=self._taproom_guest_orientation,
                                            directory_path=directory,
                                            generate_gaff_files=generate_gaff_files,
                                            gaff_version=gaff_version)

    def _solvate_windows(self, directory, available_resources):

        # Extract out only the solvent components of the substance (e.g H2O,
        # Na+, Cl-...)
        filter_solvent = miscellaneous.FilterSubstanceByRole('filter_solvent')
        filter_solvent.input_substance = self._substance
        filter_solvent.component_role = Substance.ComponentRole.Solvent

        protocol_result = filter_solvent.execute(directory, available_resources)

        if isinstance(protocol_result, PropertyEstimatorException):
            return protocol_result

        reference_structure_path = None

        # TODO: Parallelize using thread pools.
        for index, window_file_path in enumerate(self._paprika_setup.desolvated_window_paths):

            window_directory = os.path.dirname(window_file_path)
            os.makedirs(window_directory, exist_ok=True)

            # Solvate the window.
            solvate_complex = coordinates.SolvateExistingStructure('solvate_window')
            solvate_complex.max_molecules = self._number_of_solvent_molecules
            solvate_complex.box_aspect_ratio = self._simulation_box_aspect_ratio
            solvate_complex.center_solute_in_box = False

            if self._number_of_solvent_molecules < 10:
                solvate_complex.mass_density = 0.005 * unit.grams / unit.milliliters

            solvate_complex.substance = filter_solvent.filtered_substance
            solvate_complex.solute_coordinate_file = window_file_path

            protocol_result = solvate_complex.execute(window_directory, None)

            # Make sure the solvation was successful.
            if isinstance(protocol_result, PropertyEstimatorException):
                return protocol_result

            self._solvated_coordinate_paths[index] = os.path.join(window_directory, 'restrained.pdb')

            # Store the path to the structure of the first window, which will
            # serve as a reference point when adding the dummy atoms.
            if index == 0:
                reference_structure_path = solvate_complex.coordinate_file_path

            # Add the aligning dummy atoms to the solvated pdb files.
            protocol_result = self._add_dummy_atoms(index, solvate_complex.coordinate_file_path,
                                                    reference_structure_path)

            # Make sure the dummy atoms were added successfully.
            if isinstance(protocol_result, PropertyEstimatorException):
                return protocol_result

            logging.info(f'Set up window {index + 1} of '
                         f'{len(self._paprika_setup.desolvated_window_paths)}')

        return None

    def _add_dummy_atoms(self, index, solvated_structure_path, reference_structure_path):

        self._paprika_setup.add_dummy_atoms(reference_structure_path,
                                            solvated_structure_path,
                                            None,
                                            self._solvated_coordinate_paths[index],
                                            None)

    def _apply_restraint_masks(self, use_amber_indices):

        import parmed as pmd

        for index, window in enumerate(self._paprika_setup.window_list):

            window_directory = os.path.join(self._paprika_setup.directory,
                                            'windows', window)

            build_pdb_file = pmd.load_file(f'{window_directory}/build.pdb', structure=True)

            for restraint in self._paprika_setup.static_restraints + \
                             self._paprika_setup.conformational_restraints + \
                             self._paprika_setup.symmetry_restraints + \
                             self._paprika_setup.wall_restraints + \
                             self._paprika_setup.guest_restraints:

                restraint.index1 = index_from_mask(build_pdb_file, restraint.mask1, use_amber_indices)
                restraint.index2 = index_from_mask(build_pdb_file, restraint.mask2, use_amber_indices)
                if restraint.mask3:
                    restraint.index3 = index_from_mask(build_pdb_file, restraint.mask3, use_amber_indices)
                if restraint.mask4:
                    restraint.index4 = index_from_mask(build_pdb_file, restraint.mask4, use_amber_indices)

    def _setup_restraints(self):

        self._paprika_setup.static_restraints, self._paprika_setup.conformational_restraints, \
        self._paprika_setup.symmetry_restraints, self._paprika_setup.wall_restraints, \
        self._paprika_setup.guest_restraints = \
            self._paprika_setup.initialize_restraints(self._solvated_coordinate_paths[0])

    def _apply_parameters(self):

        if (self._force_field == BasePaprikaProtocol.ForceField.GAFF or
            self._force_field == BasePaprikaProtocol.ForceField.GAFF2):

            for index, window_file_path in enumerate(self._paprika_setup.desolvated_window_paths):

                window_directory = os.path.dirname(window_file_path)
                self._build_amber_parameters(index, window_directory)

    @staticmethod
    def _create_dummy_files(directory):

        dummy_frcmod_lines = [
            'Parameters for dummy atom with type Du\n',
            'MASS\n',
            'Du     208.00\n',
            '\n',
            'BOND\n',
            '\n',
            'ANGLE\n',
            '\n',
            'DIHE\n',
            '\n',
            'IMPROPER\n',
            '\n',
            'NONBON\n',
            '  Du       0.000     0.0000000\n'
        ]

        with open(os.path.join(directory, 'dummy.frcmod'), 'w') as file:
            file.writelines(dummy_frcmod_lines)

        dummy_mol2_template = '@<TRIPOS>MOLECULE\n' \
                              '{0:s}\n' \
                              '    1     0     1     0     1\n' \
                              'SMALL\n' \
                              'USER_CHARGES\n' \
                              '\n' \
                              '@<TRIPOS>ATOM\n' \
                              '  1 DUM     0.000000    0.000000    0.000000 Du    1 {0:s}     0.0000 ****\n' \
                              '@<TRIPOS>BOND\n' \
                              '@<TRIPOS>SUBSTRUCTURE\n' \
                              '      1  {0:s}              1 ****               0 ****  ****    0 ROOT\n'

        for dummy_name in ['DM1', 'DM2', 'DM3']:

            with open(os.path.join(directory, f'{dummy_name.lower()}.mol2'), 'w') as file:
                file.write(dummy_mol2_template.format(dummy_name))

    def _build_amber_parameters(self, index, window_directory):

        window_directory_to_base = os.path.relpath(
            os.path.abspath(self._paprika_setup.directory), window_directory)

        window_coordinates = os.path.relpath(self._solvated_coordinate_paths[index], window_directory)

        self._create_dummy_files(self._paprika_setup.directory)

        os.makedirs(window_directory, exist_ok=True)

        system = System()
        system.output_path = window_directory
        system.pbc_type = None
        system.neutralize = False

        gaff_version = 'gaff2'

        if self._force_field == self.ForceField.GAFF:
            gaff_version = 'gaff'

        host_frcmod = os.path.join(window_directory_to_base, f'{self._paprika_setup.host}.{gaff_version}.frcmod')
        host_mol2 = os.path.join(window_directory_to_base, f'{self._paprika_setup.host}.{gaff_version}.mol2')

        load_host_frcmod = f'loadamberparams {host_frcmod}'
        load_host_mol2 = f'{self._paprika_setup.host_yaml["resname"].upper()} = loadmol2 {host_mol2}'

        load_guest_frcmod = ''
        load_guest_mol2 = ''

        if self._taproom_guest_name is not None:

            guest_frcmod = os.path.join(window_directory_to_base, f'{self._paprika_setup.guest}.{gaff_version}.frcmod')
            guest_mol2 = os.path.join(window_directory_to_base, f'{self._paprika_setup.guest}.{gaff_version}.mol2')

            load_guest_frcmod = f'loadamberparams {guest_frcmod}'
            load_guest_mol2 = f'{self._paprika_setup.guest_yaml["name"].upper()} = loadmol2 {guest_mol2}'

        # window_pdb_file = PDBFile(self._solvated_coordinate_paths[index])
        # cell_vectors = window_pdb_file.topology.getPeriodicBoxVectors()

        system.template_lines = [
            f"source leaprc.{gaff_version}",
            f"source leaprc.water.tip3p",
            f"source leaprc.protein.ff14SB",
            load_host_frcmod,
            load_guest_frcmod,
            f"loadamberparams {os.path.join(window_directory_to_base, 'dummy.frcmod')}",
            load_host_mol2,
            load_guest_mol2,
            f"DM1 = loadmol2 {os.path.join(window_directory_to_base, 'dm1.mol2')}",
            f"DM2 = loadmol2 {os.path.join(window_directory_to_base, 'dm2.mol2')}",
            f"DM3 = loadmol2 {os.path.join(window_directory_to_base, 'dm3.mol2')}",
            f"model = loadpdb {window_coordinates}",
            # f"set model box {{{cell_vectors[0][0].value_in_unit(unit.angstrom)} "
            #                 f"{cell_vectors[1][1].value_in_unit(unit.angstrom)} "
            #                 f"{cell_vectors[2][2].value_in_unit(unit.angstrom)}}}",
            f"setBox model \"centers\"",
            "check model",
            "saveamberparm model structure.prmtop structure.rst7"
        ]

        system.build()

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

        window_indices = [index for index in range(len(self._paprika_setup.window_list))]

        full_multiples = int(np.floor(len(window_indices) / chunk_size))
        chunks = [[i * chunk_size, (i + 1) * chunk_size] for i in range(full_multiples)] + [
            [full_multiples * chunk_size, len(window_indices)]
        ]

        counter = 0

        for chunk in chunks:

            for window_index in sorted(window_indices)[chunk[0]: chunk[1]]:

                logging.info(f'Running window {window_index + 1} out of {len(self._paprika_setup.window_list)}')

                if available_resources.number_of_gpus > 0:

                    resources = ComputeResources(number_of_threads=1, number_of_gpus=1,
                                                 preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA)

                    resources._gpu_device_indices = f'{counter}'

                else:
                    resources = ComputeResources(number_of_threads=1)

                self._enqueue_window(queue, window_index, resources, exceptions)

                counter += 1

                if counter == chunk_size:

                    queue.join()
                    counter = 0

                    if len(exceptions) > 0:

                        message = ', '.join([f'{exception.directory}: {exception.message}' for exception in exceptions])
                        return PropertyEstimatorException(directory='', message=message)

        if not queue.empty():
            queue.join()

        if len(exceptions) > 0:

            message = ', '.join([f'{exception.directory}: {exception.message}' for exception in exceptions])
            return PropertyEstimatorException(directory='', message=message)

        return None

    def _enqueue_window(self, queue, index, available_resources, exceptions):
        raise NotImplementedError()

    @staticmethod
    def _run_window(queue):
        raise NotImplementedError()

    def _perform_analysis(self, directory):

        if self._results_dictionary is None:

            return PropertyEstimatorException(directory=directory,
                                              message='The results dictionary is empty.')

        if 'attach' in self._results_dictionary:

            self._attach_free_energy = EstimatedQuantity(
                -self._results_dictionary['attach']['ti-block']['fe'] * unit.kilocalorie / unit.mole,
                self._results_dictionary['attach']['ti-block']['sem'] * unit.kilocalorie / unit.mole, self._id + "_attach")

        if 'pull' in self._results_dictionary:

            self._pull_free_energy = EstimatedQuantity(
                -self._results_dictionary['pull']['ti-block']['fe'] * unit.kilocalorie / unit.mole,
                self._results_dictionary['pull']['ti-block']['sem'] * unit.kilocalorie / unit.mole, self._id + "_pull")

        if 'release' in self._results_dictionary:

            self._release_free_energy = EstimatedQuantity(
                self._results_dictionary['release']['ti-block']['fe'] * unit.kilocalorie / unit.mole,
                self._results_dictionary['release']['ti-block']['sem'] * unit.kilocalorie / unit.mole, self._id + "_release")

        if 'ref_state_work' in self._results_dictionary:

            self._reference_free_energy = EstimatedQuantity(
                -self._results_dictionary['ref_state_work'] * unit.kilocalorie / unit.mole,
                0 * unit.kilocalorie / unit.mole, self._id)

        if 'symmetry_correction' in self._results_dictionary:
            self._symmetry_correction = EstimatedQuantity(
                self._results_dictionary['symmetry_correction'] * unit.kilocalorie / unit.mole,
                0 * unit.kilocalorie / unit.mole, self._id)


        return None

    def execute(self, directory, available_resources):

        if self.setup:

            # Make sure the force field path to smirnoff has been set. When
            # optional and mutual exclusive / dependant inputs are implemented
            # this will not be needed.
            if self._force_field == self.ForceField.SMIRNOFF and (self._force_field_path is None or
                                                                  len(self._force_field_path) == 0):

                return PropertyEstimatorException(directory=directory,
                                                  message='The path to a .offxml force field file must be specified '
                                                          'when running with a SMIRNOFF force field.')

            if (available_resources.number_of_gpus > 0 and
                available_resources.number_of_gpus != available_resources.number_of_threads):

                return PropertyEstimatorException(directory=directory,
                                                  message='The number of available CPUs must match the number'
                                                          'of available GPUs for this parallelisation scheme.')

            # Create a new setup object which will load in a pAPRika host
            # and guest yaml file, setup a directory structure for the
            # paprika calculations, and create a set of coordinates for
            # each of the windows along the pathway (without any solvent).
            self._setup_paprika(directory)

            # Solvate each of the structures along the calculation path.
            result = self._solvate_windows(directory, available_resources)

            if isinstance(result, PropertyEstimatorException):
                # Make sure the solvation was successful.
                return result

            if len(self._solvated_coordinate_paths) == 0:

                return PropertyEstimatorException(directory=directory,
                                                  message='There were no defined windows to a/p/r the guest along.')

            # Apply parameters to each of the windows.
            result = self._apply_parameters()

            # Setup the actual restraints.
            self._setup_restraints()

            # Save the restraints to a file, ready for analysis.
            save_restraints(restraint_list=self._paprika_setup.static_restraints +
                                           self._paprika_setup.conformational_restraints +
                                           self._paprika_setup.symmetry_restraints +
                                           self._paprika_setup.wall_restraints +
                                           self._paprika_setup.guest_restraints,
                            filepath=os.path.join(self._paprika_setup.directory, "restraints.json"))

            if isinstance(result, PropertyEstimatorException):
                # Make sure the parameter application was successful.
                return result

        if self.simulate:

            if not self._paprika_setup:
                self._paprika_setup = paprika.setup(host=self._taproom_host_name,
                                                    guest=self._taproom_guest_name,
                                                    guest_orientation=self._taproom_guest_orientation,
                                                    build=False,
                                                    directory_path=directory)

                self._paprika_setup.desolvated_window_paths = [os.path.join(directory, self._paprika_setup.host,
                                                                            f"{self._paprika_setup.guest}-{self._taproom_guest_orientation}" if self._paprika_setup.guest else "",
                                                                            'windows', window,
                                                                            f"{self._paprika_setup.host}-{self._paprika_setup.guest}.pdb"
                                                                            if self._paprika_setup.guest else f"{self._paprika_setup.host}.pdb")
                                                               for window in self._paprika_setup.window_list]
                for index, window_file_path in enumerate(self._paprika_setup.desolvated_window_paths):
                    window_directory = os.path.dirname(window_file_path)
                    self._solvated_coordinate_paths[index] = os.path.join(window_directory, 'restrained.pdb')
                    self._solvated_system_xml_paths[index] = os.path.join(window_directory, 'restrained.xml')

            # Run the simulations
            result = self._run_windows(available_resources)

            if isinstance(result, PropertyEstimatorException):
                # Make sure the simulations were successful.
                return result

        if self.analyze:

            if not self._paprika_setup:
                self._paprika_setup = paprika.setup(host=self._taproom_host_name,
                                                    guest=self._taproom_guest_name,
                                                    guest_orientation=self._taproom_guest_orientation,
                                                    build=False,
                                                    directory_path=directory)

            # Finally, do the analysis to extract the free energy of binding.
            result = self._perform_analysis(directory)

            if isinstance(result, PropertyEstimatorException):
                # Make sure the analysis was successful.
                return result

            return self._get_output_dictionary()


@register_calculation_protocol()
class OpenMMPaprikaProtocol(BasePaprikaProtocol):
    """A protocol which will setup and run a pAPRika host-guest
    binding affinity calculation using OpenMM, starting from a
    host and guest `taproom` style .yaml definition file.
    """

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        # Protocol inputs / outputs
        self._force_field_path = ''

        self._solvated_system_xml_paths = {}

    def _add_dummy_atoms(self, index, solvated_structure_path, reference_structure_path):

        # We pull the host charges from the specified mol2 file.
        host_mol2_path = str(self._paprika_setup.benchmark_path.joinpath(
                             self._paprika_setup.host_yaml['structure']))

        window_directory = os.path.dirname(solvated_structure_path)

        unrestrained_xml_path = None
        self._solvated_system_xml_paths[index] = os.path.join(window_directory, 'restrained.xml')

        if self._force_field == BasePaprikaProtocol.ForceField.SMIRNOFF:

            # Assign force field parameters to the solvated complex system.
            # Because the openforcefield toolkit does not yet support dummy atoms,
            # we have to assign the smirnoff parameters before adding the dummy atoms.
            # Hence this specialised method.
            build_solvated_complex_system = forcefield.BuildSmirnoffSystem('build_solvated_window_system')

            build_solvated_complex_system.force_field_path = self._force_field_path

            build_solvated_complex_system.coordinate_file_path = solvated_structure_path
            build_solvated_complex_system.substance = self._substance

            build_solvated_complex_system.charged_molecule_paths = [host_mol2_path]

            result = build_solvated_complex_system.execute(window_directory, None)

            if isinstance(result, PropertyEstimatorException):
                raise ValueError(f'{result.directory}: {result.message}')

            unrestrained_xml_path = build_solvated_complex_system.system_path

        self._paprika_setup.add_dummy_atoms(reference_structure_path,
                                            solvated_structure_path,
                                            unrestrained_xml_path,
                                            self._solvated_coordinate_paths[index],
                                            self._solvated_system_xml_paths[index])

    def _apply_parameters(self):

        from simtk import unit as simtk_unit
        super(OpenMMPaprikaProtocol, self)._apply_parameters()

        if (self._force_field == BasePaprikaProtocol.ForceField.GAFF or
            self._force_field == BasePaprikaProtocol.ForceField.GAFF2):

            # Convert the amber files to OMM system objects.
            for index in range(len(self._paprika_setup.window_list)):

                window_directory = os.path.dirname(self._solvated_system_xml_paths[index])

                # Make sure to use the reorded pdb file as the new input
                shutil.copyfile(os.path.join(window_directory, 'build.pdb'),
                                self._solvated_coordinate_paths[index])

                pdb_file = PDBFile(self._solvated_coordinate_paths[index])
                cell_vectors = pdb_file.topology.getPeriodicBoxVectors()

                new_positions = []

                for position in pdb_file.positions:

                    position += cell_vectors[0] / 2.0
                    position += cell_vectors[1] / 2.0
                    position += cell_vectors[2] / 2.0

                    new_positions.append(position.value_in_unit(simtk_unit.angstrom))

                with open(self._solvated_coordinate_paths[index], 'w+') as file:
                    PDBFile.writeFile(pdb_file.topology, new_positions * simtk_unit.angstrom,
                                      file, keepIds=True)

                prmtop = AmberPrmtopFile(os.path.join(window_directory, 'structure.prmtop'))

                cutoff = self._gaff_cutoff.to(unit.angstrom).magnitude * simtk_unit.angstrom

                system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=cutoff,
                                             constraints=HBonds, removeCMMotion=False)

                system_xml = XmlSerializer.serialize(system)

                with open(self._solvated_system_xml_paths[index], 'wb') as file:
                    file.write(system_xml.encode('utf-8'))

    def _enqueue_window(self, queue, index, available_resources, exceptions):

        queue.put((index,
                   self._solvated_coordinate_paths[index],
                   self._solvated_system_xml_paths[index],
                   self._thermodynamic_state,
                   self._gaff_cutoff,
                   self._timestep,
                   self._number_of_equilibration_steps,
                   self._equilibration_output_frequency,
                   self._number_of_production_steps,
                   self._production_output_frequency,
                   available_resources,
                   exceptions))

    def _setup_restraints(self):

        super(OpenMMPaprikaProtocol, self)._setup_restraints()

        if (self._force_field == self.ForceField.GAFF or
            self._force_field == self.ForceField.GAFF2):

            self._apply_restraint_masks(use_amber_indices=False)

        # Apply the restraint forces to the solvated system xml files.
        for index, window in enumerate(self._paprika_setup.window_list):

            self._paprika_setup.initialize_calculation(window, self._solvated_coordinate_paths[index],
                                                               self._solvated_system_xml_paths[index],
                                                               self._solvated_system_xml_paths[index])

    @staticmethod
    def _run_window(queue):

        while True:

            index, window_coordinate_path, window_system_path, thermodynamic_state, gaff_cutoff, timestep, \
                number_of_equilibration_steps, equilibration_output_frequency, number_of_production_steps, \
                production_output_frequency, available_resources, exceptions = queue.get()

            try:

                window_directory = os.path.dirname(window_system_path)
                simulation_directory = os.path.join(window_directory, 'simulations')

                os.makedirs(simulation_directory, exist_ok=True)

                # Equilibration

                energy_minimisation = simulation.RunEnergyMinimisation('energy_minimisation')

                energy_minimisation.input_coordinate_file = window_coordinate_path
                energy_minimisation.system_path = window_system_path

                npt_equilibration = simulation.RunOpenMMSimulation('npt_equilibration')

                npt_equilibration.steps = number_of_equilibration_steps
                npt_equilibration.output_frequency = equilibration_output_frequency
                npt_equilibration.timestep = timestep

                npt_equilibration.ensemble = Ensemble.NPT

                npt_equilibration.thermodynamic_state = thermodynamic_state

                npt_equilibration.input_coordinate_file = ProtocolPath('output_coordinate_file',
                                                                       energy_minimisation.id)
                npt_equilibration.system_path = window_system_path

                # Production
                npt_production = simulation.RunOpenMMSimulation('npt_production')

                npt_production.steps = number_of_production_steps
                npt_production.output_frequency = production_output_frequency
                npt_production.timestep = timestep

                npt_production.ensemble = Ensemble.NPT

                npt_production.thermodynamic_state = thermodynamic_state

                npt_production.input_coordinate_file = ProtocolPath('output_coordinate_file',
                                                                    npt_equilibration.id)
                npt_production.system_path = window_system_path

                simulation_protocol = groups.ProtocolGroup(f'simulation_{index}')
                simulation_protocol.add_protocols(energy_minimisation, npt_equilibration, npt_production)

                result = simulation_protocol.execute(simulation_directory, available_resources)

                if isinstance(result, PropertyEstimatorException):
                    # Make sure the simulations were successful.
                    exceptions.append(result)
                    queue.task_done()

                    continue

                trajectory_path = simulation_protocol.get_value(ProtocolPath('trajectory_file_path',
                                                                             'npt_production'))
                coordinate_path = simulation_protocol.get_value(ProtocolPath('output_coordinate_file',
                                                                             'npt_equilibration'))

                shutil.move(trajectory_path, os.path.join(window_directory, 'trajectory.dcd'))
                shutil.move(coordinate_path, os.path.join(window_directory, 'input.pdb'))

                # shutil.rmtree(simulation_directory)

            except Exception as e:

                formatted_exception = traceback.format_exception(None, e, e.__traceback__)

                exceptions.append(PropertyEstimatorException(directory=os.path.dirname(window_coordinate_path),
                                                             message=f'An uncaught exception was raised: '
                                                                     f'{formatted_exception}'))

            queue.task_done()

    def _perform_analysis(self, directory):

        self._results_dictionary = paprika.analyze(host=self._paprika_setup.host,
                                                   guest=self._paprika_setup.guest,
                                                   guest_orientation=self.taproom_guest_orientation,
                                                   topology_file='restrained.pdb',
                                                   trajectory_mask='trajectory.dcd',
                                                   directory_path=directory,
                                                   guest_residue_name=self._paprika_setup.guest_yaml["name"] if
                                                   self._paprika_setup.guest != "release" else None).results

        super(OpenMMPaprikaProtocol, self)._perform_analysis(directory)


@register_calculation_protocol()
class AmberPaprikaProtocol(BasePaprikaProtocol):
    """A protocol which will setup and run a pAPRika host-guest
    binding affinity calculation using Amber, starting from a
    host and guest `taproom` style .yaml definition file.
    """

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        # Protocol inputs / outputs
        self._force_field = self.ForceField.GAFF2

    def _enqueue_window(self, queue, index, available_resources, exceptions):

        queue.put((index,
                   self._solvated_coordinate_paths[index],
                   None,
                   self._thermodynamic_state,
                   self._gaff_cutoff,
                   self._timestep,
                   self._number_of_equilibration_steps,
                   self._equilibration_output_frequency,
                   self._number_of_production_steps,
                   self._production_output_frequency,
                   available_resources,
                   exceptions))

    def _setup_restraints(self):

        super(AmberPaprikaProtocol, self)._setup_restraints()

        if (self._force_field == self.ForceField.GAFF or
            self._force_field == self.ForceField.GAFF2):
            # Apply the restraint masks which will re-map the restraint indices
            # to the correct atoms (tleap re-orders the packmol file). All indices
            # Get set to index+1 here ready for creating the disang files.
            self._apply_restraint_masks(use_amber_indices=True)

        for index, window in enumerate(self._paprika_setup.window_list):

            window_directory = os.path.join(self._paprika_setup.directory,
                                            'windows', window)

            with open(f'{window_directory}/disang.rest', 'w') as file:

                value = ''

                for restraint in self._paprika_setup.static_restraints + \
                                 self._paprika_setup.conformational_restraints + \
                                 self._paprika_setup.wall_restraints + \
                                 self._paprika_setup.guest_restraints:

                    value += amber.amber_restraint_line(restraint, window)

                file.write(value)

        if (self._force_field == self.ForceField.GAFF or
            self._force_field == self.ForceField.GAFF2):
            # Undo the amber index ready for saving the restraints
            # JSON file.
            self._apply_restraint_masks(use_amber_indices=False)

    @staticmethod
    def _run_window(queue):

        while True:

            index, window_coordinate_path, window_system_path, thermodynamic_state, gaff_cutoff, timestep, \
                number_of_equilibration_steps, equilibration_output_frequency, number_of_production_steps, \
                production_output_frequency, available_resources, exceptions = queue.get()

            window_directory = os.path.dirname(window_coordinate_path)

            environment = os.environ.copy()

            if available_resources.number_of_gpus < 1:

                exceptions.append(PropertyEstimatorException(directory=window_directory,
                                                             message='Currently Amber may only be run'
                                                                     'on GPUs'))

                queue.task_done()
                continue

            if available_resources.preferred_gpu_toolkit != ComputeResources.GPUToolkit.CUDA:
                raise ValueError('Paprika can only be ran either on CPUs or CUDA GPUs.')

            devices_split = [int(index.strip()) for index in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

            if len(available_resources.gpu_device_indices) > len(devices_split):

                raise ValueError(f'The number of requested GPUs '
                                 f'({len(available_resources.gpu_device_indices)}) '
                                 f'is greater than the number available '
                                 f'({len(devices_split)})')

            requested_split = [int(index.strip()) for index in available_resources.gpu_device_indices.split(',')]
            visible_devices = [str(devices_split[index]) for index in requested_split]

            devices_string = ','.join(visible_devices)
            environment['CUDA_VISIBLE_DEVICES'] = f'{devices_string}'

            logging.info(f'Starting a set of Amber simulations on GPUs {devices_string}')

            amber_simulation = Simulation()

            amber_simulation.path = f"{window_directory}/"
            amber_simulation.prefix = "minimize"

            amber_simulation.inpcrd = "structure.rst7"
            amber_simulation.ref = "structure.rst7"
            amber_simulation.topology = "structure.prmtop"
            amber_simulation.restraint_file = "disang.rest"

            amber_simulation.config_pbc_min()

            amber_simulation.cntrl["ntf"] = 2
            amber_simulation.cntrl["ntc"] = 2

            amber_simulation.cntrl["maxcyc"] = 5000
            amber_simulation.cntrl["ncyc"] = 1000

            amber_simulation.cntrl["ntr"] = 1
            amber_simulation.cntrl["restraint_wt"] = 50.0
            amber_simulation.cntrl["restraintmask"] = "'@DUM'"

            amber_simulation._amber_write_input_file()

            Popen([
                'pmemd',
                '-O',
                '-p',
                'structure.prmtop',
                '-ref',
                'structure.rst7',
                '-c',
                'structure.rst7',
                '-i',
                'minimize.in',
                '-r',
                'minimize.rst7',
                '-o',
                'minimize.out',
                '-inf',
                'minimize.info',
            ], cwd=window_directory, env=environment).wait()

            # Equilibration
            amber_simulation = Simulation()
            amber_simulation.executable = "pmemd.cuda"

            amber_simulation.path = f"{window_directory}/"
            amber_simulation.prefix = "equilibration"

            amber_simulation.inpcrd = "minimize.rst7"
            amber_simulation.ref = "minimize.rst7"
            amber_simulation.topology = "structure.prmtop"
            amber_simulation.restraint_file = "disang.rest"

            amber_simulation.config_pbc_md()
            amber_simulation.cntrl["ntr"] = 1
            amber_simulation.cntrl["restraint_wt"] = 50.0
            amber_simulation.cntrl["restraintmask"] = "'@DUM'"
            amber_simulation.cntrl["dt"] = timestep.to(unit.picoseconds).magnitude
            amber_simulation.cntrl["nstlim"] = number_of_equilibration_steps
            amber_simulation.cntrl["ntwx"] = equilibration_output_frequency
            amber_simulation.cntrl["barostat"] = 2

            amber_simulation._amber_write_input_file()

            Popen([
                'pmemd.cuda',
                '-O',
                '-p',
                'structure.prmtop',
                '-ref',
                'minimize.rst7',
                '-c',
                'minimize.rst7',
                '-i',
                'equilibration.in',
                '-r',
                'equilibration.rst7',
                '-inf',
                'equilibration.info',
                '-o',
                'equilibration.out',
                '-x',
                'equilibration.nc'
            ], cwd=window_directory, env=environment).wait()

            # Production
            amber_simulation = Simulation()
            amber_simulation.executable = "pmemd.cuda"

            amber_simulation.path = f"{window_directory}/"
            amber_simulation.prefix = "production"

            amber_simulation.inpcrd = "equilibration.rst7"
            amber_simulation.ref = "equilibration.rst7"
            amber_simulation.topology = "structure.prmtop"
            amber_simulation.restraint_file = "disang.rest"

            amber_simulation.config_pbc_md()
            amber_simulation.cntrl["ntr"] = 1
            amber_simulation.cntrl["restraint_wt"] = 50.0
            amber_simulation.cntrl["restraintmask"] = "'@DUM'"
            amber_simulation.cntrl["dt"] = timestep.to(unit.picoseconds).magnitude
            amber_simulation.cntrl["nstlim"] = number_of_production_steps
            amber_simulation.cntrl["ntwx"] = production_output_frequency
            amber_simulation.cntrl["barostat"] = 2

            amber_simulation._amber_write_input_file()

            Popen([
                'pmemd.cuda',
                '-O',
                '-p',
                'structure.prmtop',
                '-ref',
                'equilibration.rst7',
                '-c',
                'equilibration.rst7',
                '-i',
                'production.in',
                '-r',
                'production.rst7',
                '-inf',
                'production.info',
                '-o',
                'production.out',
                '-x',
                'production.nc'
            ], cwd=window_directory, env=environment).wait()

            queue.task_done()

    def _perform_analysis(self, directory):

        self._results_dictionary = paprika.analyze(host=self._paprika_setup.host,
                                                   guest=self._paprika_setup.guest,
                                                   guest_orientation=self.taproom_guest_orientation,
                                                   topology_file='restrained.pdb',
                                                   trajectory_mask='production.nc',
                                                   directory_path=directory,
                                                   guest_residue_name=self._paprika_setup.guest_yaml["name"] if
                                                   self._paprika_setup.guest != "release" else None).results

        super(AmberPaprikaProtocol, self)._perform_analysis(directory)

    def execute(self, directory, available_resources):

        if (self._force_field != self.ForceField.GAFF and
            self._force_field != self.ForceField.GAFF2):

            return PropertyEstimatorException(directory=directory,
                                              message='Currently GAFF(2) are the only force fields '
                                                      'supported with the AmberPaprikaProtocol.')

        super(AmberPaprikaProtocol, self).execute(directory, available_resources)
