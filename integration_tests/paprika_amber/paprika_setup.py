#!/usr/bin/env python
import glob as glob
import logging
import os as os
import subprocess
import traceback

import numpy as np
import paprika
from paprika.io import save_restraints
from simtk import unit

from propertyestimator.protocols import miscellaneous, coordinates
from propertyestimator.substances import Substance
from propertyestimator.utils import setup_timestamp_logging
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.quantities import EstimatedQuantity


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


def analyse_run(host_name, guest_name, setup_directory):

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

        results = paprika.analyze(host=host_name, guest=guest_name, topology_file='cb6-but-dum.prmtop',
                                  trajectory_mask='production*.nc', directory_path=setup_directory).results

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


def setup_amber_files(guest, base_directory, paprika_setup):

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

        # Add the aligning dummy atoms to the solvated pdb files.
        window_coordinate_paths[index] = os.path.join(window_directory, 'restrained.pdb')

        if index == 0:
            reference_structure_path = solvate_complex.coordinate_file_path

        paprika_setup.add_dummy_atoms(reference_structure_path,
                                      solvate_complex.coordinate_file_path,
                                      None,
                                      window_coordinate_paths[index],
                                      None)

        logging.info(f'Set up window {index}')

    if len(window_coordinate_paths) == 0:
        raise ValueError('There were no defined windows to pull the guest along.')

    # Setup the actual restraints.
    paprika_setup.static_restraints, \
    paprika_setup.conformational_restraints, \
    paprika_setup.wall_restraints, \
    paprika_setup.guest_restraints = paprika_setup.initialize_restraints(window_coordinate_paths[0])

    from paprika.restraints import amber_restraints

    for index, window in enumerate(paprika_setup.window_list):

        window_directory = os.path.join(paprika_setup.directory, 'windows', window)

        if not os.path.isdir(window_directory):
            os.makedirs(window_directory)

        from paprika.tleap import System
        system = System()
        system.output_path = window_directory
        system.pbc_type = None
        system.neutralize = False

        system.template_lines = [
            "source leaprc.gaff",
            "source leaprc.water.tip3p",
            "source leaprc.protein.ff14SB",
            "loadamberparams ../../../../../cb6.frcmod",
            "loadamberparams ../../../../../dummy.frcmod",
            "CB6 = loadmol2 ../../../../../cb6.mol2",
            "BUT = loadmol2 ../../../../../but.mol2",
            "DM1 = loadmol2 ../../../../../dm1.mol2",
            "DM2 = loadmol2 ../../../../../dm2.mol2",
            "DM3 = loadmol2 ../../../../../dm3.mol2",
            f"model = loadpdb ../../../../../{window_coordinate_paths[index]}",
            f"setBox model \"centers\"",
            "check model",
            "saveamberparm model cb6-but-dum.prmtop cb6-but-dum.rst7"
        ]

        system.build()

        from paprika.utils import index_from_mask
        import parmed as pmd

        build_pdb_file = pmd.load_file(f'{window_directory}/build.pdb', structure=True)

        with open(f'{window_directory}/disang.rest', 'w') as file:

            value = ''

            for restraint in paprika_setup.static_restraints + paprika_setup.conformational_restraints + \
                             paprika_setup.wall_restraints + paprika_setup.guest_restraints:

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

    save_restraints(restraint_list=paprika_setup.static_restraints +
                                   paprika_setup.conformational_restraints +
                                   paprika_setup.wall_restraints +
                                   paprika_setup.guest_restraints,
                    filepath=os.path.join(paprika_setup.directory, "restraints.json"))

    from paprika.amber import Simulation

    for index, window in enumerate(paprika_setup.window_list):

        window_directory = os.path.join(paprika_setup.directory, 'windows', window)

        simulation = Simulation()

        simulation.path = f"{window_directory}/"
        simulation.prefix = "minimize"

        simulation.inpcrd = "cb6-but-dum.rst7"
        simulation.ref = "cb6-but-dum.rst7"
        simulation.topology = "cb6-but-dum.prmtop"
        simulation.restraint_file = "disang.rest"

        simulation.config_pbc_min()
        simulation.cntrl["ntr"] = 1
        simulation.cntrl["restraint_wt"] = 50.0
        simulation.cntrl["restraintmask"] = "'@DUM'"

        simulation._amber_write_input_file()

        #Equilibration
        simulation = Simulation()
        simulation.executable = "pmemd.cuda"

        simulation.path = f"{window_directory}/"
        simulation.prefix = "equilibration"

        simulation.inpcrd = "minimize.rst7"
        simulation.ref = "cb6-but-dum.rst7"
        simulation.topology = "cb6-but-dum.prmtop"
        simulation.restraint_file = "disang.rest"

        simulation.config_pbc_md()
        simulation.cntrl["ntr"] = 1
        simulation.cntrl["restraint_wt"] = 50.0
        simulation.cntrl["restraintmask"] = "'@DUM'"
        simulation.cntrl["dt"] = 0.001
        simulation.cntrl["nstlim"] = 1500
        simulation.cntrl["ntwx"] = 5000
        simulation.cntrl["barostat"] = 2

        simulation._amber_write_input_file()

        #Production
        simulation = Simulation()
        simulation.executable = "pmemd.cuda"

        simulation.path = f"{window_directory}/"
        simulation.prefix = "production"

        simulation.inpcrd = "equilibration.rst7"
        simulation.ref = "cb6-but-dum.rst7"
        simulation.topology = "cb6-but-dum.prmtop"
        simulation.restraint_file = "disang.rest"

        simulation.config_pbc_md()
        simulation.cntrl["ntr"] = 1
        simulation.cntrl["restraint_wt"] = 50.0
        simulation.cntrl["restraintmask"] = "'@DUM'"
        simulation.cntrl["dt"] = 0.001
        simulation.cntrl["nstlim"] = 1000000
        simulation.cntrl["ntwx"] = 5000
        simulation.cntrl["barostat"] = 2

        simulation._amber_write_input_file()


def run_paprika(host, guest, base_directory):
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

    paprika_setup = paprika.Setup(host=host, guest=guest, directory_path=base_directory)
    setup_amber_files(guest, base_directory, paprika_setup)

    directories = glob.glob(f"{base_directory}/*/*/windows/*")
    directories = [i for i in directories if os.path.isdir(i)]

    full_multiples = int(np.floor(len(directories) / 8))
    chunks = [[i * 8, i * 8 + 8] for i in range(full_multiples)] + [
        [full_multiples * 8, len(directories)]
    ]

    counter = 0
    open_processes = []

    for chunk in chunks:

        for directory in sorted(directories)[chunk[0]: chunk[1]]:

            if not os.path.isfile(os.path.join(directory, "production.nc")):

                print(f"Running in {directory}")
                # open_processes.append(subprocess.Popen(['../run.sh']))

                my_env = os.environ
                my_env["CUDA_VISIBLE_DEVICES"] = f"{counter}"

                subprocess.call('echo $CUDA_VISIBLE_DEVICES', shell=True, env=my_env)
                subprocess.call('pwd', shell=True, cwd=directory)
                open_processes.append(subprocess.Popen(['../../../../../run.sh'], cwd=directory, env=my_env))

                counter += 1

            if len(open_processes) == 8:

                while len(open_processes) > 0:

                    open_processes[-1].wait()
                    open_processes.pop()

                counter = 0

    while len(open_processes) > 0:

        open_processes[-1].wait()
        open_processes.pop()

    return analyse_run(host_name=host,
                       guest_name=guest,
                       setup_directory=paprika_setup.directory)


def main():
    """An integrated test of calculating the gradients of observables with
    respect to force field parameters using the property estimator"""

    host_guest_directory = 'paprika_ap'

    if not os.path.isdir(host_guest_directory):
        os.mkdir(host_guest_directory)

    attach_free_energy = None
    pull_free_energy = None
    release_free_energy = None
    reference_free_energy = None
    host_guest_error = None
    host_error = None

    # try:
    attach_free_energy, \
    pull_free_energy, _, \
    reference_free_energy, \
    host_guest_error = run_paprika(host="cb6", guest="but",
                                   base_directory=host_guest_directory)
    # except Exception as e:
    #     formatted_exception = traceback.format_exception(None, e, e.__traceback__)
    #     logging.info(f'Failed to setup attach / pull calculations: {formatted_exception}')

    host_directory = 'paprika_r'

    if not os.path.isdir(host_directory):
        os.mkdir(host_directory)

    # try:
    _, _, release_free_energy, _, host_error = run_paprika(host="cb6", guest=None,
                                                           base_directory=host_directory)
    # except Exception as e:
    #     formatted_exception = traceback.format_exception(None, e, e.__traceback__)
    #     logging.info(f'Failed to setup release calculations: {formatted_exception}')

    logging.info(attach_free_energy, pull_free_energy, release_free_energy,
                 reference_free_energy, host_guest_error, host_error)


if __name__ == "__main__":
    main()
