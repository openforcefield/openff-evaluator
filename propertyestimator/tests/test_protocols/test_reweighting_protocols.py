import random
import tempfile
from os import path

import numpy as np

from propertyestimator import unit
from propertyestimator.backends import ComputeResources
from propertyestimator.protocols.coordinates import BuildCoordinatesPackmol
from propertyestimator.protocols.forcefield import BuildSmirnoffSystem
from propertyestimator.protocols.reweighting import (
    CalculateReducedPotentialOpenMM,
    ConcatenateStatistics,
    ConcatenateTrajectories,
    ReweightStatistics,
)
from propertyestimator.substances import Substance
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import get_data_filename
from propertyestimator.utils.statistics import ObservableType, StatisticsArray


def test_concatenate_trajectories():

    import mdtraj

    coordinate_path = get_data_filename("test/trajectories/water.pdb")
    trajectory_path = get_data_filename("test/trajectories/water.dcd")

    original_trajectory = mdtraj.load(trajectory_path, top=coordinate_path)

    with tempfile.TemporaryDirectory() as temporary_directory:

        concatenate_protocol = ConcatenateTrajectories("concatenate_protocol")
        concatenate_protocol.input_coordinate_paths = [coordinate_path, coordinate_path]
        concatenate_protocol.input_trajectory_paths = [trajectory_path, trajectory_path]
        concatenate_protocol.execute(temporary_directory, ComputeResources())

        final_trajectory = mdtraj.load(
            concatenate_protocol.output_trajectory_path, top=coordinate_path
        )
        assert len(final_trajectory) == len(original_trajectory) * 2


def test_concatenate_statistics():

    statistics_path = get_data_filename("test/statistics/stats_pandas.csv")
    original_array = StatisticsArray.from_pandas_csv(statistics_path)

    with tempfile.TemporaryDirectory() as temporary_directory:

        concatenate_protocol = ConcatenateStatistics("concatenate_protocol")
        concatenate_protocol.input_statistics_paths = [statistics_path, statistics_path]
        concatenate_protocol.execute(temporary_directory, ComputeResources())

        final_array = StatisticsArray.from_pandas_csv(
            concatenate_protocol.output_statistics_path
        )
        assert len(final_array) == len(original_array) * 2


def test_calculate_reduced_potential_openmm():

    substance = Substance.from_components("O")
    thermodynamic_state = ThermodynamicState(298 * unit.kelvin, 1.0 * unit.atmosphere)

    with tempfile.TemporaryDirectory() as directory:
        force_field_path = path.join(directory, "ff.json")

        with open(force_field_path, "w") as file:
            file.write(build_tip3p_smirnoff_force_field().json())

        build_coordinates = BuildCoordinatesPackmol("build_coordinates")
        build_coordinates.max_molecules = 10
        build_coordinates.mass_density = 0.05 * unit.grams / unit.milliliters
        build_coordinates.substance = substance
        build_coordinates.execute(directory, None)

        assign_parameters = BuildSmirnoffSystem(f"assign_parameters")
        assign_parameters.force_field_path = force_field_path
        assign_parameters.coordinate_file_path = build_coordinates.coordinate_file_path
        assign_parameters.substance = substance
        assign_parameters.execute(directory, None)

        reduced_potentials = CalculateReducedPotentialOpenMM(f"reduced_potentials")
        reduced_potentials.substance = substance
        reduced_potentials.thermodynamic_state = thermodynamic_state
        reduced_potentials.reference_force_field_paths = [force_field_path]
        reduced_potentials.system_path = assign_parameters.system_path
        reduced_potentials.trajectory_file_path = get_data_filename(
            "test/trajectories/water.dcd"
        )
        reduced_potentials.coordinate_file_path = get_data_filename(
            "test/trajectories/water.pdb"
        )
        reduced_potentials.kinetic_energies_path = get_data_filename(
            "test/statistics/stats_pandas.csv"
        )
        reduced_potentials.high_precision = False
        reduced_potentials.execute(directory, ComputeResources())

        assert path.isfile(reduced_potentials.statistics_file_path)

        final_array = StatisticsArray.from_pandas_csv(
            reduced_potentials.statistics_file_path
        )
        assert ObservableType.ReducedPotential in final_array


def test_reweight_statistics():

    number_of_frames = 10

    reduced_potentials = (
        np.ones(number_of_frames) * random.random() * unit.dimensionless
    )
    potentials = (
        np.ones(number_of_frames) * random.random() * unit.kilojoule / unit.mole
    )

    with tempfile.TemporaryDirectory() as directory:

        statistics_path = path.join(directory, "stats.csv")

        statistics_array = StatisticsArray()
        statistics_array[ObservableType.ReducedPotential] = reduced_potentials
        statistics_array[ObservableType.PotentialEnergy] = potentials
        statistics_array.to_pandas_csv(statistics_path)

        reweight_protocol = ReweightStatistics(f"reduced_potentials")
        reweight_protocol.statistics_type = ObservableType.PotentialEnergy
        reweight_protocol.statistics_paths = statistics_path
        reweight_protocol.reference_reduced_potentials = statistics_path
        reweight_protocol.target_reduced_potentials = statistics_path
        reweight_protocol.bootstrap_uncertainties = True
        reweight_protocol.required_effective_samples = 0
        reweight_protocol.execute(directory, ComputeResources())
