import os
import shutil

import numpy as np
import matplotlib.pyplot as plt

from propertyestimator import unit
from propertyestimator.backends import ComputeResources
from propertyestimator.protocols import coordinates, forcefield, simulation
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.thermodynamics import Ensemble, ThermodynamicState
from propertyestimator.utils import setup_timestamp_logging
from propertyestimator.utils.exceptions import PropertyEstimatorException


def _setup_dummy_system(directory, coordinte_file_name, system_file_name):

    force_field_path = os.path.join(directory, 'ff.json')

    with open(force_field_path, 'w') as file:
        file.write(build_tip3p_smirnoff_force_field().json())

    substance = Substance.from_components('CCC(=O)O')

    build_coordinates = coordinates.BuildCoordinatesPackmol('build_coordinates')
    build_coordinates.max_molecules = 1
    build_coordinates.mass_density = 0.001 * unit.grams / unit.milliliters
    build_coordinates.substance = substance
    build_coordinates.execute(directory, None)

    assign_parameters = forcefield.BuildSmirnoffSystem(f'assign_parameters')
    assign_parameters.force_field_path = force_field_path
    assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
    assign_parameters.substance = substance
    assign_parameters.execute(directory, None)

    shutil.copyfile(build_coordinates.coordinate_file_path, coordinte_file_name)
    shutil.copyfile(assign_parameters.system_path, system_file_name)


def main():

    import mdtraj

    setup_timestamp_logging()

    compute_resources = ComputeResources(number_of_threads=1)

    coordinate_path = 'input.pdb'
    system_path = 'system.xml'

    if not os.path.isfile(coordinate_path) or not os.path.isfile(system_path):

        os.makedirs('setup', exist_ok=True)
        _setup_dummy_system('setup', coordinate_path, system_path)

    if os.path.isdir('run'):
        shutil.rmtree('run')

    os.makedirs('run', exist_ok=True)

    protocol = simulation.OpenMMParallelTempering('protocol')
    protocol.total_number_of_iterations = 2000
    protocol.steps_per_iteration = 100
    protocol.input_coordinate_file = coordinate_path
    protocol.system_path = system_path
    protocol.ensemble = Ensemble.NVT
    protocol.enable_pbc = False
    protocol.allow_gpu_platforms = False
    protocol.high_precision = True

    protocol.thermodynamic_state = ThermodynamicState(298 * unit.kelvin)
    protocol.maximum_temperature = 1243 * unit.kelvin
    # protocol.number_of_replicas = 4

    protocol.replica_temperatures = [313 * unit.kelvin,
                                     343 * unit.kelvin,
                                     403 * unit.kelvin,
                                     523 * unit.kelvin,
                                     763 * unit.kelvin]

    result = protocol.execute('run', compute_resources)
    assert not isinstance(result, PropertyEstimatorException)

    trajectory = mdtraj.load_dcd('trajectory.dcd', top=coordinate_path)

    dihedrals = mdtraj.compute_dihedrals(trajectory, indices=np.array([[3, 2, 4, 10]]))[:, 0]
    dihedrals = dihedrals / np.pi * 180.0 + 180

    num_bins = 180

    plt.hist(dihedrals, num_bins, range=[0.0, 360])
    plt.show()


if __name__ == "__main__":
    main()
