#!/usr/bin/env python
import json
import logging
import os
import shutil
import traceback

import paprika
from paprika.io import save_restraints
from simtk import unit

from propertyestimator.backends import ComputeResources, DaskLocalClusterBackend, QueueWorkerResources, DaskLSFBackend
from propertyestimator.protocols import miscellaneous, coordinates, forcefield, simulation, groups
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.thermodynamics import Ensemble, ThermodynamicState
from propertyestimator.utils import setup_timestamp_logging
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.serialization import TypedJSONEncoder
from propertyestimator.workflow.utils import ProtocolPath


def mol2_to_smiles(file_path):
    """Loads a receptor from a mol2 file.

    Parameters
    ----------
    file_path: str
        The file path to the mol2 file.

    Returns
    -------
    str
        The smiles descriptor of the loaded receptor molecule
    """
    from openforcefield.topology import Molecule

    receptor_molecule = Molecule.from_file(file_path, 'MOL2')
    return receptor_molecule.to_smiles()


def build_substance(ligand_smiles, receptor_smiles, ionic_strength=None):
    """Builds a substance containing a ligand and receptor solvated
    in an aqueous solution with a given ionic strength

    Parameters
    ----------
    ligand_smiles: str, optional
        The smiles descriptor of the ligand.
    receptor_smiles: str
        The smiles descriptor of the host.
    ionic_strength: simtk.unit.Quantity, optional
        The ionic strength of the aqueous solvent.

    Returns
    -------
    Substance
        The built substance.
    """

    substance = Substance()

    if ligand_smiles is not None:

        ligand = Substance.Component(smiles=ligand_smiles,
                                     role=Substance.ComponentRole.Ligand)

        substance.add_component(component=ligand, amount=Substance.ExactAmount(1))

    receptor = Substance.Component(smiles=receptor_smiles,
                                   role=Substance.ComponentRole.Receptor)

    substance.add_component(component=receptor, amount=Substance.ExactAmount(1))

    water = Substance.Component(smiles='O', role=Substance.ComponentRole.Solvent)
    sodium = Substance.Component(smiles='[Na+]', role=Substance.ComponentRole.Solvent)
    chlorine = Substance.Component(smiles='[Cl-]', role=Substance.ComponentRole.Solvent)

    water_mole_fraction = 1.0

    if ionic_strength is not None:

        salt_mole_fraction = Substance.calculate_aqueous_ionic_mole_fraction(ionic_strength) / 2.0
        water_mole_fraction = 1.0 - salt_mole_fraction

        substance.add_component(component=sodium, amount=Substance.MoleFraction(salt_mole_fraction))
        substance.add_component(component=chlorine, amount=Substance.MoleFraction(salt_mole_fraction))

    substance.add_component(component=water, amount=Substance.MoleFraction(water_mole_fraction))

    return substance


def build_simulation_protocol_group(group_id, coordinate_file, system_file, ensemble=Ensemble.NPT):

    thermodynamic_state = ThermodynamicState(298*unit.kelvin, 1.0*unit.atmosphere)

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

            release_free_energy = EstimatedQuantity(results['release']['ti-block']['fe'] * unit.kilocalorie_per_mole,
                                                    results['release']['ti-block']['sem'] * unit.kilocalorie_per_mole,
                                                    'paprika')

        if 'ref_state_work' in results:

            reference_state_work = -results['ref_state_work'] * unit.kilocalorie_per_mole

            microstate_correction = (0.0 if 'symmetry_correction' not in results
                                     else results['symmetry_correction']) * unit.kilocalorie_per_mole

            reference_free_energy = EstimatedQuantity(reference_state_work + microstate_correction,
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


def run_window_simulation(index, window_coordinate_path, window_system_xml_path, available_resources, **_):

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


def run_paprika(host, guest, base_directory, calculation_backend=None):
    """Runs paprika for a given host and guest.

    Parameters
    ----------
    host : str
    guest : str, optional
    base_directory : str
    calculation_backend : PropertyEstimatorBackend

    Returns
    -------

    """
    setup_timestamp_logging()

    force_field_path = build_tip3p_smirnoff_force_field()

    paprika_setup = paprika.Setup(host=host, guest=guest, directory_path=base_directory)

    host_mol2_path = str(paprika_setup.benchmark_path.joinpath(
        paprika_setup.host_yaml['structure']))

    host_smiles = mol2_to_smiles(host_mol2_path)
    guest_smiles = None

    if guest is not None:

        guest_mol2_path = str(paprika_setup.benchmark_path.joinpath(
            paprika_setup.guest).joinpath(
            paprika_setup.guest_yaml['structure']))

        guest_smiles = mol2_to_smiles(guest_mol2_path)

    substance = build_substance(guest_smiles, host_smiles)

    filter_solvent = miscellaneous.FilterSubstanceByRole('filter_solvent')
    filter_solvent.input_substance = substance  # ProtocolPath('substance', 'global')
    filter_solvent.component_role = Substance.ComponentRole.Solvent

    filter_solvent.execute(base_directory, None)

    window_system_xml_paths = {}
    window_coordinate_paths = {}

    reference_structure_path = None

    for index, window_file_path in enumerate(paprika_setup.desolvated_window_paths):

        window_directory = os.path.dirname(window_file_path)

        if not os.path.isdir(window_directory):
            os.mkdir(window_directory)

        solvate_complex = coordinates.SolvateExistingStructure('solvate_window')
        solvate_complex.max_molecules = 2000
        solvate_complex.box_aspect_ratio = [1.0, 1.0, 2.0]
        solvate_complex.center_solute_in_box = False

        solvate_complex.substance = filter_solvent.filtered_substance
        solvate_complex.solute_coordinate_file = window_file_path

        solvate_complex.execute(window_directory, None)

        # Assign force field parameters to the solvated complex system.
        build_solvated_complex_system = forcefield.BuildSmirnoffSystem('build_solvated_window_system')

        build_solvated_complex_system.force_field_path = force_field_path  # ProtocolPath('force_field_path', 'global')

        build_solvated_complex_system.coordinate_file_path = solvate_complex.coordinate_file_path  # ProtocolPath('coordinate_file_path', solvate_complex.id)
        build_solvated_complex_system.substance = substance

        build_solvated_complex_system.charged_molecule_paths = [host_mol2_path]

        build_solvated_complex_system.execute(window_directory, None)

        # Add the aligning dummy atoms to the solvated pdb files.
        window_system_xml_paths[index] = os.path.join(window_directory, 'restrained.xml')
        window_coordinate_paths[index] = os.path.join(window_directory, 'restrained.pdb')

        if index == 0:
            reference_structure_path = solvate_complex.coordinate_file_path

        paprika_setup.add_dummy_atoms(reference_structure_path,
                                      solvate_complex.coordinate_file_path,
                                      build_solvated_complex_system.system_path,
                                      window_coordinate_paths[index],
                                      window_system_xml_paths[index])

        logging.info(f'Set up window {index}')

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

            logging.info(f'Submitting window {index} {window_coordinate_paths[index]} {window_system_xml_paths[index]}')

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


def main():
    """An integrated test of calculating the gradients of observables with
    respect to force field parameters using the property estimator"""

    # worker_resources = ComputeResources(number_of_threads=1)
    #
    # calculation_backend = DaskLocalClusterBackend(number_of_workers=4,
    #                                               resources_per_worker=worker_resources)

    # worker_resources = ComputeResources(number_of_gpus=1, preferred_gpu_toolkit=ComputeResources.GPUToolkit.CUDA)
    #
    # calculation_backend = DaskLocalClusterBackend(number_of_workers=8,
    #                                               resources_per_worker=worker_resources)

    # calculation_backend = None

    queue_resources = QueueWorkerResources(number_of_threads=1,
                                           number_of_gpus=1,
                                           preferred_gpu_toolkit=QueueWorkerResources.GPUToolkit.CUDA,
                                           per_thread_memory_limit=8 * (unit.giga * unit.byte),
                                           wallclock_time_limit="05:59")

    worker_script_commands = [
        'export OE_LICENSE="/home/boothros/oe_license.txt"',
        '. /home/boothros/miniconda3/etc/profile.d/conda.sh',
        'conda activate propertyestimator',
        'module load cuda/9.2'
    ]

    calculation_backend = DaskLSFBackend(minimum_number_of_workers=1,
                                         maximum_number_of_workers=20,
                                         resources_per_worker=queue_resources,
                                         queue_name='gpuqueue',
                                         setup_script_commands=worker_script_commands,
                                         adaptive_interval='1000ms')

    calculation_backend.start()

    host_guest_directory = 'paprika_ap'

    # if os.path.isdir(host_guest_directory):
    #     shutil.rmtree(host_guest_directory)

    if not os.path.isdir(host_guest_directory):
        os.mkdir(host_guest_directory)

    host_guest_future = None

    try:
        host_guest_future = run_paprika(host="cb6", guest="but",
                                        base_directory=host_guest_directory,
                                        calculation_backend=calculation_backend)
    except Exception as e:
        formatted_exception = traceback.format_exception(None, e, e.__traceback__)
        logging.info(f'Failed to setup attach / pull calculations: {formatted_exception}')

    host_directory = 'paprika_r'

    # if os.path.isdir(host_directory):
    #     shutil.rmtree(host_directory)

    if not os.path.isdir(host_directory):
        os.mkdir(host_directory)

    host_future = None

    try:
        host_future = run_paprika(host="cb6", guest=None,
                                  base_directory=host_directory,
                                  calculation_backend=calculation_backend)
    except Exception as e:
        formatted_exception = traceback.format_exception(None, e, e.__traceback__)
        logging.info(f'Failed to setup release calculations: {formatted_exception}')

    attach_free_energy = None
    pull_free_energy = None
    release_free_energy = None
    reference_free_energy = None
    host_guest_error = None
    host_error = None

    if host_guest_future is not None:

        attach_free_energy, pull_free_energy, _, reference_free_energy, error = host_guest_future.result()

        if error is not None:
            logging.info(f'The attach, pull calculations failed with error: {error}')

        else:
            logging.info(f'The attach, pull calculation yielded free energies of '
                         f'{attach_free_energy} and {pull_free_energy} respectively '
                         f'and a reference correction free energy of {reference_free_energy}.')

    if host_future is not None:

        _, _, release_free_energy, _, error = host_future.result()

        if error is not None:
            logging.info(f'The release calculation failed with error: {error}')
        else:
            logging.info(f'The release calculation yielded a free energy of {release_free_energy}.')

    logging.info(attach_free_energy, pull_free_energy, release_free_energy,
                 reference_free_energy, host_guest_error, host_error)


if __name__ == "__main__":
    main()

