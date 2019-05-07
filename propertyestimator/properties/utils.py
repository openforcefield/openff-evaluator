"""
A set of utilities for setting up property estimation workflows.
"""

from collections import namedtuple

from propertyestimator.protocols import analysis, forcefield, reweighting
from propertyestimator.workflow.schemas import ProtocolReplicator
from propertyestimator.workflow.utils import ProtocolPath, ReplicatorValue

BaseReweightingProtocols = namedtuple('BaseReweightingProtocols', 'unpack_stored_data '
                                                                  'analysis_protocol '
                                                                  'decorrelate_trajectory '
                                                                  'concatenate_trajectories '
                                                                  'build_reference_system '
                                                                  'reduced_reference_potential '
                                                                  'build_target_system '
                                                                  'reduced_target_potential '
                                                                  'mbar_protocol ')


def generate_base_reweighting_protocols(analysis_protocol, replicator_id='data_repl', id_suffix=''):
    """Constructs a set of protocols which, when combined in a workflow schema,
    may be executed to reweight a set of existing data to estimate a particular
    property. The reweighted observable of interest will be calculated by
    following the passed in `analysis_protocol`.

    Parameters
    ----------
    analysis_protocol: AveragePropertyProtocol
        The protocol which will take input from the stored data,
        and generate a set of observables to reweight.
    replicator_id: str
        The id to use for the data replicator.
    id_suffix: str
        A string suffix to append to each of the protocol ids.

    Returns
    -------
    BaseReweightingProtocols:
        A named tuple of the protocol which should form the bulk of
        a property estimation workflow.
    ProtocolReplicator:
        A replicator which will clone the workflow for each piece of
        stored data.
    """

    assert isinstance(analysis_protocol, analysis.AveragePropertyProtocol)

    replicator_suffix = '_$({}){}'.format(replicator_id, id_suffix)

    # Unpack all the of the stored data.
    unpack_stored_data = reweighting.UnpackStoredSimulationData('unpack_data{}'.format(replicator_suffix))
    unpack_stored_data.simulation_data_path = ReplicatorValue(replicator_id)

    # The autocorrelation time of each of the stored files will be calculated for this property
    # using the passed in analysis protocol.

    # Decorrelate the frames of the concatenated trajectory.
    decorrelate_trajectory = analysis.ExtractUncorrelatedTrajectoryData('decorrelate_traj{}'.format(replicator_suffix))

    decorrelate_trajectory.statistical_inefficiency = ProtocolPath('statistical_inefficiency',
                                                                   analysis_protocol.id)
    decorrelate_trajectory.equilibration_index = ProtocolPath('equilibration_index',
                                                              analysis_protocol.id)
    decorrelate_trajectory.input_coordinate_file = ProtocolPath('coordinate_file_path',
                                                                unpack_stored_data.id)
    decorrelate_trajectory.input_trajectory_path = ProtocolPath('trajectory_file_path',
                                                                unpack_stored_data.id)

    # Stitch together all of the trajectories
    concatenate_trajectories = reweighting.ConcatenateTrajectories('concat_traj' + id_suffix)

    concatenate_trajectories.input_coordinate_paths = [ProtocolPath('coordinate_file_path',
                                                                    unpack_stored_data.id)]

    concatenate_trajectories.input_trajectory_paths = [ProtocolPath('output_trajectory_path',
                                                                    decorrelate_trajectory.id)]

    # Calculate the reduced potentials for each of the reference states.
    build_reference_system = forcefield.BuildSmirnoffSystem('build_system{}'.format(replicator_suffix))

    build_reference_system.force_field_path = ProtocolPath('force_field_path', unpack_stored_data.id)
    build_reference_system.substance = ProtocolPath('substance', unpack_stored_data.id)
    build_reference_system.coordinate_file_path = ProtocolPath('coordinate_file_path',
                                                               unpack_stored_data.id)

    reduced_reference_potential = reweighting.CalculateReducedPotentialOpenMM('reduced_potential{}'.format(
                                                                              replicator_suffix))

    reduced_reference_potential.system_path = ProtocolPath('system_path', build_reference_system.id)
    reduced_reference_potential.thermodynamic_state = ProtocolPath('thermodynamic_state',
                                                                   unpack_stored_data.id)
    reduced_reference_potential.coordinate_file_path = ProtocolPath('coordinate_file_path',
                                                                    unpack_stored_data.id)
    reduced_reference_potential.trajectory_file_path = ProtocolPath('output_trajectory_path',
                                                                    concatenate_trajectories.id)

    # Calculate the reduced potential of the target state.
    build_target_system = forcefield.BuildSmirnoffSystem('build_system_target' + id_suffix)

    build_target_system.force_field_path = ProtocolPath('force_field_path', 'global')
    build_target_system.substance = ProtocolPath('substance', 'global')
    build_target_system.coordinate_file_path = ProtocolPath('output_coordinate_path',
                                                            concatenate_trajectories.id)

    reduced_target_potential = reweighting.CalculateReducedPotentialOpenMM('reduced_potential_target' + id_suffix)

    reduced_target_potential.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')
    reduced_target_potential.system_path = ProtocolPath('system_path', build_target_system.id)
    reduced_target_potential.coordinate_file_path = ProtocolPath('output_coordinate_path',
                                                                 concatenate_trajectories.id)
    reduced_target_potential.trajectory_file_path = ProtocolPath('output_trajectory_path',
                                                                 concatenate_trajectories.id)

    # Finally, apply MBAR to get the reweighted value.
    mbar_protocol = reweighting.ReweightWithMBARProtocol('mbar' + id_suffix)

    mbar_protocol.reference_reduced_potentials = [ProtocolPath('reduced_potentials',
                                                               reduced_reference_potential.id)]

    mbar_protocol.reference_observables = [ProtocolPath('uncorrelated_values', analysis_protocol.id)]
    mbar_protocol.target_reduced_potentials = [ProtocolPath('reduced_potentials', reduced_target_potential.id)]

    base_protocols = BaseReweightingProtocols(unpack_stored_data,
                                              analysis_protocol,
                                              decorrelate_trajectory,
                                              concatenate_trajectories,
                                              build_reference_system,
                                              reduced_reference_potential,
                                              build_target_system,
                                              reduced_target_potential,
                                              mbar_protocol)

    # Create the replicator object.
    component_replicator = ProtocolReplicator(replicator_id=replicator_id)
    component_replicator.protocols_to_replicate = []

    # Pass it paths to the protocols to be replicated.
    for protocol in base_protocols:

        if protocol.id.find('$({})'.format(replicator_id)) < 0:
            continue

        component_replicator.protocols_to_replicate.append(ProtocolPath('', protocol.id))

    component_replicator.template_values = ProtocolPath('full_system_data', 'global')

    return base_protocols, component_replicator
