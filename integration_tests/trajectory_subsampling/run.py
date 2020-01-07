"""An integrated test to ensure that streamed subsampling of trajectories
yields correct results relative to subsampling a trajectory fully loaded
into memory.
"""
import logging
import time

import numpy as np

from propertyestimator import unit
from propertyestimator.backends import ComputeResources
from propertyestimator.protocols.analysis import ExtractUncorrelatedTrajectoryData
from propertyestimator.protocols.forcefield import BuildSmirnoffSystem
from propertyestimator.protocols.openmm import OpenMMSimulation
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import setup_timestamp_logging


def generate_trajectories():
    """Generates trajectories to subsample.
    """

    setup_timestamp_logging()

    logger = logging.getLogger()

    substance = Substance.from_components('C(C(C(C(C(F)(F)Br)(F)F)(F)F)(F)F)(C(C(C(F)(F)F)(F)F)(F)F)(F)F')

    logger.info('Building system.')

    build_system = BuildSmirnoffSystem('build_system')
    build_system.coordinate_file_path = 'coords.pdb'
    build_system.substance = substance
    build_system.force_field_path = 'smirnoff99Frosst-1.1.0.offxml'
    build_system.execute('', None)

    logger.info('System built.')

    production_simulation = OpenMMSimulation(f'production_simulation')
    production_simulation.steps_per_iteration = 500
    production_simulation.output_frequency = 1
    production_simulation.timestep = 2.0 * unit.femtosecond
    production_simulation.thermodynamic_state = ThermodynamicState(temperature=298.15*unit.kelvin,
                                                                   pressure=1.0*unit.atmosphere)
    production_simulation.input_coordinate_file = 'coords.pdb'
    production_simulation.system_path = 'system.xml'

    compute_resources = ComputeResources(number_of_threads=4)

    logger.info(f'Simulation started.')
    production_simulation_schema = production_simulation.schema
    production_simulation.execute('', compute_resources)
    production_simulation.schema = production_simulation_schema
    logger.info(f'Simulation finished.')


def subsample_trajectory_memory(output_name, equilibration_index=0, stride=5):
    """Subsamples a trajectory by fully loading it into memory.
    """

    import mdtraj

    trajectory = mdtraj.load_dcd(filename='trajectory.dcd', top='coords.pdb')
    trajectory = trajectory[equilibration_index:]

    uncorrelated_indices = [index for index in range(0, trajectory.n_frames, stride)]
    uncorrelated_trajectory = trajectory[uncorrelated_indices]

    uncorrelated_trajectory.save_dcd(output_name)


def main():

    import mdtraj
    generate_trajectories()

    subsample_inputs = [
        (0, 1),
        (0, 5),
        (0, 10),
        (1, 1),
        (3, 5),
        (9, 10)
    ]

    for equilibration_index, stride, in subsample_inputs:

        start_time = time.perf_counter()
        subsample_trajectory = ExtractUncorrelatedTrajectoryData('stream_subsample')
        subsample_trajectory.input_coordinate_file = 'coords.pdb'
        subsample_trajectory.input_trajectory_path = 'trajectory.dcd'
        subsample_trajectory.equilibration_index = equilibration_index
        subsample_trajectory.statistical_inefficiency = stride
        subsample_trajectory.execute('', None)
        protocol_total_time = (time.perf_counter() - start_time) * 1000

        start_time = time.perf_counter()
        subsample_trajectory_memory('memory.dcd', equilibration_index, stride)
        memory_total_time = (time.perf_counter() - start_time) * 1000

        print(f'Eq Index={equilibration_index} '
              f'Stride={stride} '
              f'Protocol Time={protocol_total_time}s '
              f'Memory Time={memory_total_time}s')

        stream_trajectory = mdtraj.load_dcd(filename=subsample_trajectory.output_trajectory_path, top='coords.pdb')
        memory_trajectory = mdtraj.load_dcd(filename='memory.dcd', top='coords.pdb')

        assert len(stream_trajectory) == len(memory_trajectory)
        assert np.allclose(stream_trajectory.xyz, memory_trajectory.xyz)
        assert np.allclose(stream_trajectory.unitcell_lengths, memory_trajectory.unitcell_lengths)
        assert np.allclose(stream_trajectory.unitcell_angles, memory_trajectory.unitcell_angles)


if __name__ == '__main__':
    main()
