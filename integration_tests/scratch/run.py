import os
import shutil

from propertyestimator import unit
from propertyestimator.backends import ComputeResources
from propertyestimator.protocols import coordinates, forcefield, simulation
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.thermodynamics import Ensemble, ThermodynamicState
from propertyestimator.utils import setup_timestamp_logging
from propertyestimator.utils.exceptions import PropertyEstimatorException


def _setup_dummy_system(directory):

    force_field_path = os.path.join(directory, 'ff.json')

    with open(force_field_path, 'w') as file:
        file.write(build_tip3p_smirnoff_force_field().json())

    substance = Substance.from_components('C')

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

    return build_coordinates.coordinate_file_path, assign_parameters.system_path


def main():

    setup_timestamp_logging()

    thermodynamic_state = ThermodynamicState(298 * unit.kelvin)

    compute_resources = ComputeResources(number_of_threads=1)

    if os.path.isdir('setup'):
        shutil.rmtree('setup')
    if os.path.isdir('run'):
        shutil.rmtree('run')

    os.makedirs('setup', exist_ok=True)
    os.makedirs('run', exist_ok=True)

    coordinate_path, system_path = _setup_dummy_system('setup')

    protocol = simulation.OpenMMParallelTempering('protocol')
    protocol.total_number_of_iterations = 333
    protocol.steps_per_iteration = 3000
    protocol.thermodynamic_state = thermodynamic_state
    protocol.input_coordinate_file = coordinate_path
    protocol.system_path = system_path
    protocol.ensemble = Ensemble.NVT
    protocol.enable_pbc = False
    protocol.allow_gpu_platforms = False
    protocol.high_precision = True

    protocol.maximum_temperature = 450 * unit.kelvin
    protocol.number_of_replicas = 4

    result = protocol.execute('run', compute_resources)
    assert not isinstance(result, PropertyEstimatorException)

    # for iterations in range(1, 11):
    #
    #     equilibration_simulation.total_number_of_iterations = iterations
    #     assert isinstance(equilibration_simulation.execute('run', compute_resources), dict)
    #     print(iterations)


if __name__ == "__main__":
    main()
