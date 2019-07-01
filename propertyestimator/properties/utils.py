"""
A set of utilities for setting up property estimation workflows.
"""
import copy
from collections import namedtuple

from propertyestimator.protocols import analysis, forcefield, gradients, groups, reweighting
from propertyestimator.workflow import WorkflowOptions
from propertyestimator.workflow.plugins import available_protocols
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


def generate_base_reweighting_protocols(analysis_protocol, workflow_options,
                                        replicator_id='data_repl', id_suffix=''):
    """Constructs a set of protocols which, when combined in a workflow schema,
    may be executed to reweight a set of existing data to estimate a particular
    property. The reweighted observable of interest will be calculated by
    following the passed in `analysis_protocol`.

    Parameters
    ----------
    analysis_protocol: AveragePropertyProtocol
        The protocol which will take input from the stored data,
        and generate a set of observables to reweight.
    workflow_options: WorkflowOptions
        The options being used to generate a workflow.
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

    # Decorrelate the frames of the stored trajectory.
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

    mbar_protocol.reference_reduced_potentials = [ProtocolPath('statistics_file_path',
                                                               reduced_reference_potential.id)]

    mbar_protocol.reference_observables = [ProtocolPath('uncorrelated_values', analysis_protocol.id)]
    mbar_protocol.target_reduced_potentials = [ProtocolPath('statistics_file_path', reduced_target_potential.id)]

    # TODO: Implement a cleaner way to handle this.
    if workflow_options.convergence_mode == WorkflowOptions.ConvergenceMode.NoChecks:
        mbar_protocol.required_effective_samples = -1

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


def _get_default_gradient_reweighting_schema(observable_values):
    """Generates the schema for a `ReweightWithMBARProtocol` with settings
    appropriate for use in reweighting observables for direct gradients
    evaluations.

    Parameters
    ----------
    observable_values: ProtocolPath, optional
        The path to the observables whose gradient is to be determined.

    Returns
    -------
    ProtocolSchema
        The created schema.

    """
    mbar_protocol = reweighting.ReweightWithMBARProtocol(f'mbar')
    mbar_protocol.reference_observables = [observable_values]
    mbar_protocol.required_effective_samples = 0
    mbar_protocol.bootstrap_uncertainties = False

    return mbar_protocol.schema


def generate_gradient_protocol_group(reference_force_field_paths,
                                     target_force_field_path,
                                     coordinate_file_path,
                                     trajectory_file_path,
                                     replicator_id='repl',
                                     observable_values=None,
                                     template_reweighting_schema=None,
                                     perturbation_scale=1.0e-4):
    """Constructs a set of protocols which, when combined in a workflow schema,
    may be executed to reweight a set of existing data to estimate a particular
    property. The reweighted observable of interest will be calculated by
    following the passed in `analysis_protocol`.

    Parameters
    ----------
    reference_force_field_paths: list of ProtocolPath
        The paths to the force field parameters which were used to generate
        the trajectories from which the observables of interest were calculated.
    target_force_field_path: ProtocolPath
        The path to the force field parameters which the observables are being
         estimated at (this is mainly only useful when estimating the gradients
         of reweighted observables).
    coordinate_file_path: ProtocolPath
        A path to the initial coordinates of the simulation trajectory which
        was used to estimate the observable of interest.
    trajectory_file_path: ProtocolPath
        A path to the simulation trajectory which was used
        to estimate the observable of interest.
    replicator_id: str
        A unique id which will be used for the protocol replicator which will
        replicate this group for every parameter of interest.
    observable_values: ProtocolPath, optional
        The path to the observables whose gradient is to be determined. This
        parameter is mutually exclusive with `template_reweighting_schema`.
    template_reweighting_schema: ReweightWithMBARProtocol
        Not yet implemented.
    perturbation_scale: float
        The default amount to perturb parameters by.

    Returns
    -------
    ProtocolGroup
        The protocol group which will estimate the gradient of
        an observable with respect to one parameter.
    ProtocolReplicator
        The replicator which will copy the gradient group for
        every parameter of interest.
    ProtocolPath
        A protocol path which points to the final gradient value.
    """

    if ((observable_values is not None and template_reweighting_schema is not None) or
        (observable_values is None and template_reweighting_schema is None)):

        raise ValueError('Either `observable_values` or `template_reweighting_schema` '
                         'must not be `None` (but not both).')

    # Define the protocol which will evaluate the reduced potentials of the
    # reference, forward and reverse states using only a subset of the full
    # force field.
    reduced_potentials = gradients.GradientReducedPotentials(f'gradient_reduced_potentials_$({replicator_id})')

    reduced_potentials.substance = ProtocolPath('substance', 'global')
    reduced_potentials.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')
    reduced_potentials.reference_force_field_paths = reference_force_field_paths
    reduced_potentials.force_field_path = target_force_field_path

    reduced_potentials.trajectory_file_path = trajectory_file_path
    reduced_potentials.coordinate_file_path = coordinate_file_path

    reduced_potentials.parameter_key = ReplicatorValue(replicator_id)
    reduced_potentials.perturbation_scale = perturbation_scale

    reduced_potentials.use_subset_of_force_field = True

    # Set up the protocols which will actually reweight the value of the
    # observable to the forward and reverse states.
    if template_reweighting_schema is None:
        template_reweighting_schema = _get_default_gradient_reweighting_schema(observable_values)

    reverse_mbar_schema = copy.deepcopy(template_reweighting_schema)
    reverse_mbar_schema.id = f'reverse_{reverse_mbar_schema.id}_$({replicator_id})'

    reverse_mbar = available_protocols[reverse_mbar_schema.type](reverse_mbar_schema.id)
    reverse_mbar.schema = reverse_mbar_schema
    reverse_mbar.reference_reduced_potentials = ProtocolPath('reference_potential_paths', reduced_potentials.id)
    reverse_mbar.target_reduced_potentials = [ProtocolPath('reverse_potentials_path', reduced_potentials.id)]
    reverse_mbar.bootstrap_iterations = min(1, reverse_mbar.bootstrap_iterations)

    forward_mbar_schema = copy.deepcopy(template_reweighting_schema)
    forward_mbar_schema.id = f'forward_{forward_mbar_schema.id}_$({replicator_id})'

    forward_mbar = available_protocols[forward_mbar_schema.type](forward_mbar_schema.id)
    forward_mbar.schema = forward_mbar_schema
    forward_mbar.reference_reduced_potentials = ProtocolPath('reference_potential_paths', reduced_potentials.id)
    forward_mbar.target_reduced_potentials = [ProtocolPath('forward_potentials_path', reduced_potentials.id)]
    forward_mbar.bootstrap_iterations = min(1, reverse_mbar.bootstrap_iterations)

    # Set up the protocol which will actually evaluate the parameter gradient
    # using the central difference method.
    central_difference = gradients.CentralDifferenceGradient(f'central_difference_$({replicator_id})')
    central_difference.parameter_key = ReplicatorValue(replicator_id)
    central_difference.reverse_observable_value = ProtocolPath('value', reverse_mbar.id)
    central_difference.forward_observable_value = ProtocolPath('value', forward_mbar.id)
    central_difference.reverse_parameter_value = ProtocolPath('reverse_parameter_value', reduced_potentials.id)
    central_difference.forward_parameter_value = ProtocolPath('forward_parameter_value', reduced_potentials.id)

    # Assemble all of the protocols into a convenient group wrapper.
    gradient_group = groups.ProtocolGroup(f'gradient_group_$({replicator_id})')
    gradient_group.add_protocols(reduced_potentials, reverse_mbar, forward_mbar, central_difference)

    protocols_to_replicate = [ProtocolPath('', gradient_group.id)]

    protocols_to_replicate.extend([ProtocolPath('', gradient_group.id, protocol_id) for
                                   protocol_id in gradient_group.protocols])

    # Create the replicator which will copy the group for each parameter gradient
    # which will be calculated.
    parameter_replicator = ProtocolReplicator(replicator_id=replicator_id)
    parameter_replicator.protocols_to_replicate = protocols_to_replicate
    parameter_replicator.template_values = ProtocolPath('parameter_gradient_keys', 'global')

    return gradient_group, parameter_replicator, ProtocolPath('gradient', gradient_group.id, central_difference.id)
