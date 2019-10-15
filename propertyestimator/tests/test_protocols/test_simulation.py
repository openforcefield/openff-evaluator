"""
Units tests for propertyestimator.protocols.simulation
"""
import tempfile
from os import path

from propertyestimator import unit
from propertyestimator.backends import ComputeResources
from propertyestimator.protocols.coordinates import BuildCoordinatesPackmol
from propertyestimator.protocols.forcefield import BuildSmirnoffSystem
from propertyestimator.protocols.simulation import RunEnergyMinimisation, RunOpenMMSimulation
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.exceptions import PropertyEstimatorException


def _setup_dummy_system(directory):

    force_field_path = path.join(directory, 'ff.json')

    with open(force_field_path, 'w') as file:
        file.write(build_tip3p_smirnoff_force_field().json())

    substance = Substance.from_components('C')

    build_coordinates = BuildCoordinatesPackmol('build_coordinates')
    build_coordinates.max_molecules = 1
    build_coordinates.mass_density = 0.001 * unit.grams / unit.milliliters
    build_coordinates.substance = substance
    build_coordinates.execute(directory, None)

    assign_parameters = BuildSmirnoffSystem(f'assign_parameters')
    assign_parameters.force_field_path = force_field_path
    assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
    assign_parameters.substance = substance
    assign_parameters.execute(directory, None)

    return build_coordinates.coordinate_file_path, assign_parameters.system_path


def test_run_energy_minimisation():

    with tempfile.TemporaryDirectory() as directory:

        coordinate_path, system_path = _setup_dummy_system(directory)

        energy_minimisation = RunEnergyMinimisation('energy_minimisation')
        energy_minimisation.input_coordinate_file = coordinate_path
        energy_minimisation.system_path = system_path

        result = energy_minimisation.execute(directory, ComputeResources())

        assert not isinstance(result, PropertyEstimatorException)
        assert path.isfile(energy_minimisation.output_coordinate_file)


def test_run_openmm_simulation():

    thermodynamic_state = ThermodynamicState(298 * unit.kelvin,
                                             1.0 * unit.atmosphere)

    with tempfile.TemporaryDirectory() as directory:

        coordinate_path, system_path = _setup_dummy_system(directory)

        npt_equilibration = RunOpenMMSimulation('npt_equilibration')
        npt_equilibration.steps = 2
        npt_equilibration.output_frequency = 1
        npt_equilibration.thermodynamic_state = thermodynamic_state
        npt_equilibration.input_coordinate_file = coordinate_path
        npt_equilibration.system_path = system_path

        result = npt_equilibration.execute(directory, ComputeResources())

        assert not isinstance(result, PropertyEstimatorException)
        assert path.isfile(npt_equilibration.output_coordinate_file)
        assert path.isfile(npt_equilibration.trajectory_file_path)
        assert path.isfile(npt_equilibration.statistics_file_path)
