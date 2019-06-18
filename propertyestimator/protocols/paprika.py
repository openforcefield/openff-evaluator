"""
A collection of protocols for performing free energy calculations using
the pAPRika software package.
"""
import logging
import os
import shutil
import traceback
from enum import Enum

import paprika
from paprika.amber import Simulation
from paprika.io import save_restraints
from paprika.restraints import amber_restraints
from paprika.tleap import System
from paprika.utils import index_from_mask
from simtk import unit
from simtk.openmm import XmlSerializer
from simtk.openmm.app import AmberPrmtopFile, HBonds, PME

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

    @protocol_input(unit.Quantity)
    def gaff_cutoff(self):
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
    def reference_free_energy(self):
        pass

    def __init__(self, protocol_id):

        super().__init__(protocol_id)

        # Protocol inputs / outputs
        self._substance = None
        self._thermodynamic_state = None

        self._force_field = self.ForceField.SMIRNOFF
        self._force_field_path = ''
        self._water_model = self.WaterModel.TIP3P

        self._taproom_host_name = None
        self._taproom_guest_name = None

        self._gaff_cutoff = 0.8 * unit.nanometer

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

        self._paprika_setup = None

        self._solvated_coordinate_paths = {}
        self._results_dictionary = None

    def _setup_paprika(self, directory):

        self._paprika_setup = paprika.Setup(host=self._taproom_host_name,
                                            guest=self._taproom_guest_name,
                                            directory_path=directory)

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

    def _setup_restraints(self):

        self._paprika_setup.static_restraints, self._paprika_setup.conformational_restraints, \
            self._paprika_setup.wall_restraints, self._paprika_setup.guest_restraints = \
            self._paprika_setup.initialize_restraints(self._solvated_coordinate_paths[0])

        # Save the restraints to a file, ready for analysis.
        save_restraints(restraint_list=self._paprika_setup.static_restraints +
                                       self._paprika_setup.conformational_restraints +
                                       self._paprika_setup.wall_restraints +
                                       self._paprika_setup.guest_restraints,
                        filepath=os.path.join(self._paprika_setup.directory, "restraints.json"))

    def _apply_parameters(self):
        raise NotImplementedError

    def _build_amber_parameters(self, index, window_directory):

        window_directory_to_base = os.path.relpath(
            os.path.abspath(self._paprika_setup.directory), window_directory)

        window_coordinates = os.path.relpath(self._solvated_coordinate_paths[index], window_directory)

        os.makedirs(window_directory, exist_ok=True)

        system = System()
        system.output_path = window_directory
        system.pbc_type = None
        system.neutralize = False

        system.template_lines = [
            f"source leaprc.gaff",
            f"source leaprc.water.tip3p",
            f"source leaprc.protein.ff14SB",
            f"loadamberparams {os.path.join(window_directory_to_base, f'{self._paprika_setup.host}.gaff2.frcmod')}",
            f"loadamberparams {os.path.join(window_directory_to_base, f'{self._paprika_setup.guest}.gaff2.frcmod')}",
            f"loadamberparams {os.path.join(window_directory_to_base, 'dummy.frcmod')}",
            f"CB6 = loadmol2 {os.path.join(window_directory_to_base, f'{self._paprika_setup.host}.gaff2.mol2')}",
            f"BUT = loadmol2 {os.path.join(window_directory_to_base, f'{self._paprika_setup.guest}.gaff2.mol2')}",
            f"DM1 = loadmol2 {os.path.join(window_directory_to_base, 'dm1.mol2')}",
            f"DM2 = loadmol2 {os.path.join(window_directory_to_base, 'dm2.mol2')}",
            f"DM3 = loadmol2 {os.path.join(window_directory_to_base, 'dm3.mol2')}",
            f"model = loadpdb {window_coordinates}",
            f"setBox model \"centers\"",
            "check model",
            "saveamberparm model structure.prmtop structure.rst7"
        ]

        system.build()

    def _run_windows(self, available_resources):

        for index, window in enumerate(self._paprika_setup.window_list):

            logging.info(f'Running window {index + 1} out of {len(self._paprika_setup.window_list)}')
            self._run_window(index, available_resources)

    def _run_window(self, index, available_resources):
        raise NotImplementedError()

    def _perform_analysis(self, directory):

        if self._results_dictionary is None:

            return PropertyEstimatorException(directory=directory,
                                              message='The results dictionary is empty.')

        if 'attach' in self._results_dictionary:

            self._attach_free_energy = EstimatedQuantity(
                -self._results_dictionary['attach']['ti-block']['fe'] * unit.kilocalorie_per_mole,
                self._results_dictionary['attach']['ti-block']['sem'] * unit.kilocalorie_per_mole, self._id)

        if 'pull' in self._results_dictionary:

            self._pull_free_energy = EstimatedQuantity(
                -self._results_dictionary['pull']['ti-block']['fe'] * unit.kilocalorie_per_mole,
                self._results_dictionary['pull']['ti-block']['sem'] * unit.kilocalorie_per_mole, self._id)

        if 'release' in self._results_dictionary:

            self._release_free_energy = EstimatedQuantity(
                self._results_dictionary['release']['ti-block']['fe'] * unit.kilocalorie_per_mole,
                self._results_dictionary['release']['ti-block']['sem'] * unit.kilocalorie_per_mole, self._id)

        if 'ref_state_work' in self._results_dictionary:

            self._reference_free_energy = EstimatedQuantity(
                -self._results_dictionary['ref_state_work'] * unit.kilocalorie_per_mole,
                0 * unit.kilocalorie_per_mole, self._id)

        return None

    def execute(self, directory, available_resources):

        # Make sure the force field path to smirnoff has been set. When
        # optional and mutual exclusive / dependant inputs are implemented
        # this will not be needed.
        if self._force_field == self.ForceField.SMIRNOFF and (self._force_field_path is None or
                                                              len(self._force_field_path) == 0):

            return PropertyEstimatorException(directory=directory,
                                              message='The path to a .offxml force field file must be specified '
                                                      'when running with a SMIRNOFF force field.')

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

        # Setup the actual restraints.
        self._setup_restraints()

        # Apply parameters to each of the windows.
        result = self._apply_parameters()

        if isinstance(result, PropertyEstimatorException):
            # Make sure the parameter application was successful.
            return result

        # Run the simulations
        self._run_windows(available_resources)

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

            build_solvated_complex_system.execute(window_directory, None)

            unrestrained_xml_path = build_solvated_complex_system.system_path

        self.paprika_setup.add_dummy_atoms(reference_structure_path,
                                           solvated_structure_path,
                                           unrestrained_xml_path,
                                           solvated_structure_path,
                                           self._solvated_system_xml_paths[index])

        if self._force_field == BasePaprikaProtocol.ForceField.GAFF2:

            self._build_amber_parameters(index, window_directory)

            prmtop = AmberPrmtopFile('structure.prmtop')

            system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=self._gaff_cutoff,
                                         constraints=HBonds)

            system_xml = XmlSerializer.serialize(system)

            with open(self._solvated_system_xml_paths[index], 'wb') as file:
                file.write(system_xml.encode('utf-8'))

    def _apply_parameters(self):

        # Apply the restraint forces to the solvated system xml files.
        for index, window in enumerate(self._paprika_setup.window_list):

            self._paprika_setup.initialize_calculation(window, self._solvated_system_xml_paths[index],
                                                               self._solvated_system_xml_paths[index])

    def _build_simulation_protocol(self, group_id, coordinate_file, system_file):

        # Equilibration
        energy_minimisation = simulation.RunEnergyMinimisation('energy_minimisation')

        energy_minimisation.input_coordinate_file = coordinate_file
        energy_minimisation.system_path = system_file

        npt_equilibration = simulation.RunOpenMMSimulation('npt_equilibration')

        npt_equilibration.steps = self._number_of_equilibration_steps
        npt_equilibration.output_frequency = self._equilibration_output_frequency

        npt_equilibration.ensemble = Ensemble.NPT

        npt_equilibration.thermodynamic_state = self._thermodynamic_state

        npt_equilibration.input_coordinate_file = ProtocolPath('output_coordinate_file', energy_minimisation.id)
        npt_equilibration.system_path = system_file

        # Production
        npt_production = simulation.RunOpenMMSimulation('npt_production')

        npt_production.steps = self._number_of_production_steps
        npt_production.output_frequency = self._production_output_frequency

        npt_production.ensemble = Ensemble.NPT

        npt_production.thermodynamic_state = self._thermodynamic_state

        npt_production.input_coordinate_file = ProtocolPath('output_coordinate_file', npt_equilibration.id)
        npt_production.system_path = system_file

        grouped_protocols = groups.ProtocolGroup(group_id)
        grouped_protocols.add_protocols(energy_minimisation, npt_equilibration, npt_production)

        return grouped_protocols

    def _run_window(self, index, available_resources):

        window_coordinate_path = self._solvated_coordinate_paths[index]
        window_system_xml_path = self._solvated_system_xml_paths[index]

        try:

            window_directory = os.path.dirname(window_system_xml_path)
            simulation_directory = os.path.join(window_directory, 'simulations')

            os.makedirs(simulation_directory, exist_ok=True)

            simulation_protocol = self._build_simulation_protocol_group(f'simulation_{index}',
                                                                        window_coordinate_path,
                                                                        window_system_xml_path)

            result = simulation_protocol.execute(simulation_directory, available_resources)

            if isinstance(result, PropertyEstimatorException):
                # Make sure the simulations were successful.
                return result

            trajectory_path = simulation_protocol.get_value(ProtocolPath('trajectory_file_path', 'npt_production'))
            coordinate_path = simulation_protocol.get_value(ProtocolPath('output_coordinate_file', 'npt_equilibration'))

            shutil.move(trajectory_path, os.path.join(window_directory, 'trajectory.dcd'))
            shutil.move(coordinate_path, os.path.join(window_directory, 'input.pdb'))

            shutil.rmtree(simulation_directory)

        except Exception as e:

            formatted_exception = traceback.format_exception(None, e, e.__traceback__)

            return PropertyEstimatorException(directory=os.path.dirname(window_coordinate_path),
                                              message=f'An uncaught exception was raised: {formatted_exception}')

        return result

    def _perform_analysis(self, directory):

        self._results_dictionary = paprika.analyze(host=self._paprika_setup.host,
                                                   guest=self._paprika_setup.guest,
                                                   topology_file='restrained.pdb',
                                                   trajectory_mask='*.dcd',
                                                   directory_path=directory).results

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

    def _setup_paprika(self, directory):

        self._paprika_setup = paprika.Setup(host=self._taproom_host_name,
                                            guest=self._taproom_guest_name,
                                            directory_path=directory,
                                            generate_gaff_files=True)

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

    def _apply_parameters(self):

        import parmed as pmd

        self._create_dummy_files(self._paprika_setup.directory)

        for index, window in enumerate(self._paprika_setup.window_list):

            window_directory = os.path.join(self._paprika_setup.directory,
                                            'windows', window)

            self._build_amber_parameters(index, window_directory)

            build_pdb_file = pmd.load_file(f'{window_directory}/build.pdb', structure=True)

            with open(f'{window_directory}/disang.rest', 'w') as file:

                value = ''

                for restraint in self._paprika_setup.static_restraints + \
                                 self._paprika_setup.conformational_restraints + \
                                 self._paprika_setup.wall_restraints + \
                                 self._paprika_setup.guest_restraints:

                    try:
                        restraint.index1 = index_from_mask(build_pdb_file, restraint.mask1, True)
                    except:
                        pass
                    try:
                        restraint.index2 = index_from_mask(build_pdb_file, restraint.mask2, True)
                    except:
                        pass
                    try:
                        restraint.index3 = index_from_mask(build_pdb_file, restraint.mask3, True)
                    except:
                        pass
                    try:
                        restraint.index4 = index_from_mask(build_pdb_file, restraint.mask4, True)
                    except:
                        pass

                    value += amber_restraints.amber_restraint_line(restraint, window)

                file.write(value)

    def _run_window(self, index, available_resources):

        window_coordinate_path = self._solvated_coordinate_paths[index]
        window_directory = os.path.dirname(window_coordinate_path)

        environment = os.environ

        if available_resources.number_of_gpus >= 1:

            if available_resources.preferred_gpu_toolkit != ComputeResources.GPUToolkit.CUDA:
                raise ValueError('Paprika can only be ran either on CPUs or CUDA GPUs.')

            devices_split = [int(index.trim()) for index in os.environ['CUDA_VISIBLE_DEVICES'].split(',')]

            if len(available_resources.gpu_device_indices) > len(devices_split):

                raise ValueError(f'The number of requested GPUs '
                                 f'({len(available_resources.gpu_device_indices)}) '
                                 f'is greater than the number available '
                                 f'({len(devices_split)})')

            requested_split = [int(index.trim()) for index in available_resources.gpu_device_indices.split(',')]
            visible_devices = [str(devices_split[index]) for index in requested_split]

            devices_string = {','.join(visible_devices)}
            environment['CUDA_VISIBLE_DEVICES'] = f'{devices_string}'

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

        amber_simulation.cntrl["ntr"] = 1
        amber_simulation.cntrl["restraint_wt"] = 50.0
        amber_simulation.cntrl["restraintmask"] = "'@DUM'"

        amber_simulation._amber_write_input_file()

        os.subprocess.Popen([
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
            '-inf',
            'minimize.info',
        ], cwd=window_directory, env=environment)

        # Equilibration
        amber_simulation = Simulation()
        amber_simulation.executable = "pmemd.cuda"

        amber_simulation.path = f"{window_directory}/"
        amber_simulation.prefix = "equilibration"

        amber_simulation.inpcrd = "minimize.rst7"
        amber_simulation.ref = "structure.rst7"
        amber_simulation.topology = "structure.prmtop"
        amber_simulation.restraint_file = "disang.rest"

        amber_simulation.config_pbc_md()
        amber_simulation.cntrl["ntr"] = 1
        amber_simulation.cntrl["restraint_wt"] = 50.0
        amber_simulation.cntrl["restraintmask"] = "'@DUM'"
        amber_simulation.cntrl["dt"] = 0.001
        amber_simulation.cntrl["nstlim"] = self._number_of_equilibration_steps
        amber_simulation.cntrl["ntwx"] = self._equilibration_output_frequency
        amber_simulation.cntrl["barostat"] = 2

        amber_simulation._amber_write_input_file()

        os.subprocess.Popen([
            'pmemd',
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
            '-x',
            'equilibration.nc'
        ], cwd=window_directory, env=environment)

        # Production
        amber_simulation = Simulation()
        amber_simulation.executable = "pmemd.cuda"

        amber_simulation.path = f"{window_directory}/"
        amber_simulation.prefix = "production"

        amber_simulation.inpcrd = "equilibration.rst7"
        amber_simulation.ref = "structure.rst7"
        amber_simulation.topology = "structure.prmtop"
        amber_simulation.restraint_file = "disang.rest"

        amber_simulation.config_pbc_md()
        amber_simulation.cntrl["ntr"] = 1
        amber_simulation.cntrl["restraint_wt"] = 50.0
        amber_simulation.cntrl["restraintmask"] = "'@DUM'"
        amber_simulation.cntrl["dt"] = 0.001
        amber_simulation.cntrl["nstlim"] = self._number_of_production_steps
        amber_simulation.cntrl["ntwx"] = self._production_output_frequency
        amber_simulation.cntrl["barostat"] = 2

        amber_simulation._amber_write_input_file()

        os.subprocess.Popen([
            'pmemd',
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
            '-x',
            'production.nc'
        ], cwd=window_directory, env=environment)

    def _perform_analysis(self, directory):

        self._results_dictionary = paprika.analyze(host=self._paprika_setup.host,
                                                   guest=self._paprika_setup.guest,
                                                   topology_file='structure.prmtop',
                                                   trajectory_mask='production.nc',
                                                   directory_path=directory).results

        super(AmberPaprikaProtocol, self)._perform_analysis(directory)

    def execute(self, directory, available_resources):

        if self._force_field != self.ForceField.TIP3PGAFF:

            return PropertyEstimatorException(directory=directory,
                                              message='Currently GAFF2 is the only force field '
                                                      'supported with the AmberPaprikaProtocol.')

        super(AmberPaprikaProtocol, self).execute(directory, available_resources)
