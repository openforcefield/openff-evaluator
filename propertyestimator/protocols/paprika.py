"""
A collection of protocols for performing free energy calculations using
the pAPRika software package.
"""
import logging
import os
import shutil
import traceback

import paprika
from simtk import unit

from propertyestimator.protocols import miscellaneous, coordinates, forcefield
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState, Ensemble
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.workflow.decorators import protocol_input, MergeBehaviour
from propertyestimator.workflow.plugins import register_calculation_protocol
from propertyestimator.workflow.protocols import BaseProtocol


@register_calculation_protocol()
class BasePaprikaProtocol(BaseProtocol):
    """A protocol which will setup and run a pAPRika host-guest
    binding affinity calculation, starting from a host and guest
    `taproom` style .yaml definition file.
    """

    @protocol_input(Substance)
    def substance(self):
        """The substance which defines the host, guest and solvent."""
        pass

    @protocol_input(ThermodynamicState)
    def thermodynamic_state(self):
        """The state at which to run the calculations."""
        pass

    @protocol_input(str)
    def taproom_host_name(self):
        pass

    @protocol_input(str)
    def taproom_guest_name(self):
        pass

    @protocol_input(int, merge_behavior=MergeBehaviour.GreatestValue)
    def number_of_equilibration_steps(self):
        pass

    @protocol_input(int, merge_behavior=MergeBehaviour.GreatestValue)
    def number_of_production_steps(self):
        pass

    @protocol_input(int)
    def number_of_solvent_molecules(self):
        pass

    @protocol_input(list)
    def simulation_box_aspect_ratio(self):
        pass

    def __init__(self, protocol_id):
        super().__init__(protocol_id)

        self._substance = None
        self._thermodynamic_state = None

        self._taproom_host_name = None
        self._taproom_guest_name = None

        self._number_of_equilibration_steps = 200000
        self._number_of_production_steps = 1000000

        self._number_of_solvent_molecules = 2000

        self._simulation_box_aspect_ratio = [1.0, 1.0, 2.0]

        self._paprika_setup = None

    def build_simulation_protocol_group(self, group_id, coordinate_file, system_file, ensemble=Ensemble.NPT):

        thermodynamic_state = ThermodynamicState(298 * unit.kelvin, 1.0 * unit.atmosphere)

        # Equilibration
        energy_minimisation = simulation.RunEnergyMinimisation('energy_minimisation')

        energy_minimisation.input_coordinate_file = coordinate_file
        energy_minimisation.system_path = system_file

        npt_equilibration = simulation.RunOpenMMSimulation('npt_equilibration')

        npt_equilibration.steps = 200000
        npt_equilibration.output_frequency = 5000

        npt_equilibration.ensemble = ensemble

        npt_equilibration.thermodynamic_state = thermodynamic_state

        npt_equilibration.input_coordinate_file = ProtocolPath('output_coordinate_file', energy_minimisation.id)
        npt_equilibration.system_path = system_file

        # Production
        npt_production = simulation.RunOpenMMSimulation('npt_production')

        npt_production.steps = 1000000
        npt_production.output_frequency = 5000

        npt_production.ensemble = ensemble

        npt_production.thermodynamic_state = thermodynamic_state

        npt_production.input_coordinate_file = ProtocolPath('output_coordinate_file', npt_equilibration.id)
        npt_production.system_path = system_file

        grouped_protocols = groups.ProtocolGroup(group_id)
        grouped_protocols.add_protocols(energy_minimisation, npt_equilibration, npt_production)

        return grouped_protocols

    def analyse_run(*args, host_name, guest_name, setup_directory, available_resources, **kwargs):

        exception_raised = False

        for index, arg in enumerate(args):
            logging.info(f'Simulation {index} finished with result: {json.dumps(arg, cls=TypedJSONEncoder)}')

            if isinstance(arg, PropertyEstimatorException):
                exception_raised = True

        if exception_raised:
            return

        attach_free_energy = EstimatedQuantity(0 * unit.kilocalorie_per_mole,
                                               0 * unit.kilocalorie_per_mole,
                                               'paprika')

        pull_free_energy = EstimatedQuantity(0 * unit.kilocalorie_per_mole,
                                             0 * unit.kilocalorie_per_mole,
                                             'paprika')

        release_free_energy = EstimatedQuantity(0 * unit.kilocalorie_per_mole,
                                                0 * unit.kilocalorie_per_mole,
                                                'paprika')

        reference_free_energy = EstimatedQuantity(0 * unit.kilocalorie_per_mole,
                                                  0 * unit.kilocalorie_per_mole,
                                                  'paprika')

        try:
            results = paprika.analyze(host=host_name, guest=guest_name, topology_file='restrained.pdb',
                                      trajectory_mask='*.dcd', directory_path=setup_directory).results

            if 'attach' in results:
                attach_free_energy = EstimatedQuantity(-results['attach']['ti-block']['fe'] * unit.kilocalorie_per_mole,
                                                       results['attach']['ti-block']['sem'] * unit.kilocalorie_per_mole,
                                                       'paprika')

            if 'pull' in results:
                pull_free_energy = EstimatedQuantity(-results['pull']['ti-block']['fe'] * unit.kilocalorie_per_mole,
                                                     results['pull']['ti-block']['sem'] * unit.kilocalorie_per_mole,
                                                     'paprika')

            if 'release' in results:
                release_free_energy = EstimatedQuantity(
                    results['release']['ti-block']['fe'] * unit.kilocalorie_per_mole,
                    results['release']['ti-block']['sem'] * unit.kilocalorie_per_mole,
                    'paprika')

            if 'ref_state_work' in results:
                reference_free_energy = EstimatedQuantity(-results['ref_state_work'] * unit.kilocalorie_per_mole,
                                                          0 * unit.kilocalorie_per_mole,
                                                          'paprika')

        except Exception as e:

            formatted_exception = traceback.format_exception(None, e, e.__traceback__)
            logging.info(f'Failed to analyse Host {host_name} Guest {guest_name}: {formatted_exception}')

            return None, None, None, None, PropertyEstimatorException(directory='', message=f'Failed to analyse '
            f'Host {host_name} '
            f'Guest {guest_name}: '
            f'{formatted_exception}')

        return attach_free_energy, pull_free_energy, release_free_energy, reference_free_energy, None

    def run_window_simulation(self, index, window_coordinate_path, window_system_xml_path, available_resources, **_):

        try:
            window_directory = os.path.dirname(window_system_xml_path)

            if not os.path.isdir(window_directory):
                os.mkdir(window_directory)

            simulation_directory = os.path.join(window_directory, 'simulations')

            if not os.path.isdir(simulation_directory):
                os.mkdir(simulation_directory)

            simulation_protocol = build_simulation_protocol_group(f'simulation_{index}',
                                                                  window_coordinate_path,
                                                                  window_system_xml_path)

            result = simulation_protocol.execute(simulation_directory, available_resources)

            trajectory_path = simulation_protocol.get_value(ProtocolPath('trajectory_file_path', 'npt_production'))
            coordinate_path = simulation_protocol.get_value(ProtocolPath('output_coordinate_file', 'npt_equilibration'))

            shutil.copyfile(trajectory_path, os.path.join(window_directory, 'trajectory.dcd'))
            shutil.copyfile(coordinate_path, os.path.join(window_directory, 'input.pdb'))

            shutil.rmtree(simulation_directory)
        except Exception as e:

            formatted_exception = traceback.format_exception(None, e, e.__traceback__)

            return PropertyEstimatorException(directory=os.path.dirname(window_coordinate_path),
                                              message=f'An uncaught exception was raised: {formatted_exception}')

        return result

    def _solvate_windows(self, directory, available_resources):

        # Extract out only the solvent components of the substance (e.g H2O,
        # Na+, Cl-...)
        filter_solvent = miscellaneous.FilterSubstanceByRole('filter_solvent')
        filter_solvent.input_substance = self._substance
        filter_solvent.component_role = Substance.ComponentRole.Solvent

        protocol_output = filter_solvent.execute(directory, available_resources)

        if isinstance(protocol_output, PropertyEstimatorException):
            return protocol_output

        window_system_xml_paths = {}
        window_coordinate_paths = {}

        reference_structure_path = None

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

            solvate_complex.execute(window_directory, available_resources)

            # Assign force field parameters to the solvated complex system.
            build_solvated_complex_system = forcefield.BuildSmirnoffSystem('build_solvated_window_system')

            build_solvated_complex_system.force_field_path = force_field_path  # ProtocolPath('force_field_path', 'global')

            build_solvated_complex_system.coordinate_file_path = solvate_complex.coordinate_file_path  # ProtocolPath('coordinate_file_path', solvate_complex.id)
            build_solvated_complex_system.substance = self._substance

            build_solvated_complex_system.charged_molecule_paths = [host_mol2_path]

            build_solvated_complex_system.execute(window_directory, available_resources)

            # Add the aligning dummy atoms to the solvated pdb files.
            window_system_xml_paths[index] = os.path.join(window_directory, 'restrained.xml')
            window_coordinate_paths[index] = os.path.join(window_directory, 'restrained.pdb')

            if index == 0:
                reference_structure_path = solvate_complex.coordinate_file_path

            self.paprika_setup.add_dummy_atoms(reference_structure_path,
                                               solvate_complex.coordinate_file_path,
                                               build_solvated_complex_system.system_path,
                                               window_coordinate_paths[index],
                                               window_system_xml_paths[index])

            logging.info(f'Set up window {index}')

    def execute(self, directory, available_resources):

        # Create a new setup object which will load in a pAPRika host
        # and guest yaml file, setup a directory structure for the
        # paprika calculations, and create a set of coordinates for
        # each of the windows along the pathway (without any solvent).
        self._paprika_setup = paprika.Setup(host=self._taproom_host_name,
                                            guest=self.__taproom_guest_name,
                                            directory_path=directory)

        self._solvate_windows()

        if len(window_coordinate_paths) == 0:
            raise ValueError('There were no defined windows to pull the guest along.')

        # Setup the actual restraints.
        paprika_setup.static_restraints, \
        paprika_setup.conformational_restraints, \
        paprika_setup.wall_restraints, \
        paprika_setup.guest_restraints = paprika_setup.initialize_restraints(window_coordinate_paths[0])

        save_restraints(restraint_list=paprika_setup.static_restraints +
                                       paprika_setup.conformational_restraints +
                                       paprika_setup.wall_restraints +
                                       paprika_setup.guest_restraints,
                        filepath=os.path.join(paprika_setup.directory, "restraints.json"))

        simulation_futures = []

        for index, window in enumerate(paprika_setup.window_list):

            paprika_setup.initialize_calculation(window, window_system_xml_paths[index],
                                                 window_system_xml_paths[index])

            if calculation_backend is None:

                run_window_simulation(index, window_coordinate_paths[index],
                                      window_system_xml_paths[index],
                                      ComputeResources())

            else:

                logging.info(
                    f'Submitting window {index} {window_coordinate_paths[index]} {window_system_xml_paths[index]}')

                simulation_futures.append(calculation_backend.submit_task(run_window_simulation,
                                                                          index,
                                                                          window_coordinate_paths[index],
                                                                          window_system_xml_paths[index]))

        analysis_future = None

        if calculation_backend is None:

            analyse_run(host_name=host,
                        guest_name=guest,
                        *simulation_futures,
                        setup_directory=paprika_setup.directory,
                        available_resources=ComputeResources())

        else:

            analysis_future = calculation_backend.submit_task(analyse_run,
                                                              *simulation_futures,
                                                              host_name=host,
                                                              guest_name=guest,
                                                              setup_directory=base_directory)

        return analysis_future