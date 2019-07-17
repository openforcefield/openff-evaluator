"""
A set of utilities for setting up property estimation workflows.
"""
import copy
from collections import namedtuple

from propertyestimator.protocols import analysis, forcefield, gradients, groups, reweighting, coordinates, simulation
from propertyestimator.thermodynamics import Ensemble
from propertyestimator.workflow import WorkflowOptions
from propertyestimator.workflow.plugins import available_protocols
from propertyestimator.workflow.schemas import ProtocolReplicator, WorkflowSimulationDataToStore
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


BaseSimulationProtocols = namedtuple('BaseSimulationProtocols', 'build_coordinates '
                                                                'assign_parameters '
                                                                'energy_minimisation '
                                                                'equilibration_simulation '
                                                                'production_simulation '
                                                                'analysis_protocol '
                                                                'converge_uncertainty '
                                                                'extract_uncorrelated_trajectory '
                                                                'extract_uncorrelated_statistics ')


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


def generate_base_simulation_protocols(analysis_protocol, workflow_options, id_suffix='',
                                       conditional_group=None):
    """Constructs a set of protocols which, when combined in a workflow schema,
    may be executed to run a single simulation to estimate a particular
    property. The observable of interest to extract from the simulation is determined
    by the passed in `analysis_protocol`.

    The protocols returned will:

        1) Build a set of liquid coordinates for the
           property substance using packmol.

        2) Assign a set of smirnoff force field parameters
           to the system.

        3) Perform an energy minimisation on the system.

        4) Run a short NPT equilibration simulation for 100000 steps
           using a timestep of 1fs.

        5) Within a conditional group (up to a maximum of 100 times):

            5a) Run a longer NPT production simulation for 1000000 steps
           using a timestep of 1fs

            5b) Extract the average value of an observable and
                it's uncertainty.

            5c) If a convergence mode is set by the options,
                check if the target uncertainty has been met.
                If not, repeat steps 5a), 5b) and 5c).

        6) Extract uncorrelated configurations from a generated production
           simulation.

        7) Extract uncorrelated statistics from a generated production
           simulation.

    Parameters
    ----------
    analysis_protocol: AveragePropertyProtocol
        The protocol which will extract the observable of
        interest from the generated simulation data.
    workflow_options: WorkflowOptions
        The options being used to generate a workflow.
    id_suffix: str
        A string suffix to append to each of the protocol ids.
    conditional_group: ProtocolGroup, optional
        A custom group to wrap the main simulation / extraction
        protocols within. It is up to the caller of this method to
        manually add the convergence conditions to this group.
        If `None`, a default group with uncertainty convergence
        conditions is automatically constructed.

    Returns
    -------
    BaseSimulationProtocols
        A named tuple of the generated protocols.
    ProtocolPath
        A reference to the final value of the estimated observable
        and its uncertainty (an `EstimatedQuantity`).
    WorkflowSimulationDataToStore
        An object which describes the default data from a simulation to store,
        such as the uncorrelated statistics and configurations.
    """

    assert isinstance(analysis_protocol, analysis.AveragePropertyProtocol)

    build_coordinates = coordinates.BuildCoordinatesPackmol(f'build_coordinates{id_suffix}')
    build_coordinates.substance = ProtocolPath('substance', 'global')

    assign_parameters = forcefield.BuildSmirnoffSystem(f'assign_parameters{id_suffix}')
    assign_parameters.force_field_path = ProtocolPath('force_field_path', 'global')
    assign_parameters.coordinate_file_path = ProtocolPath('coordinate_file_path', build_coordinates.id)
    assign_parameters.substance = ProtocolPath('substance', 'global')

    # Equilibration
    energy_minimisation = simulation.RunEnergyMinimisation(f'energy_minimisation{id_suffix}')
    energy_minimisation.input_coordinate_file = ProtocolPath('coordinate_file_path', build_coordinates.id)
    energy_minimisation.system_path = ProtocolPath('system_path', assign_parameters.id)

    equilibration_simulation = simulation.RunOpenMMSimulation(f'equilibration_simulation{id_suffix}')
    equilibration_simulation.ensemble = Ensemble.NPT
    equilibration_simulation.steps = 100000
    equilibration_simulation.output_frequency = 5000
    equilibration_simulation.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')
    equilibration_simulation.input_coordinate_file = ProtocolPath('output_coordinate_file', energy_minimisation.id)
    equilibration_simulation.system_path = ProtocolPath('system_path', assign_parameters.id)

    # Production
    production_simulation = simulation.RunOpenMMSimulation(f'production_simulation{id_suffix}')
    production_simulation.ensemble = Ensemble.NPT
    production_simulation.steps = 1000000
    production_simulation.output_frequency = 10000
    production_simulation.thermodynamic_state = ProtocolPath('thermodynamic_state', 'global')
    production_simulation.input_coordinate_file = ProtocolPath('output_coordinate_file', equilibration_simulation.id)
    production_simulation.system_path = ProtocolPath('system_path', assign_parameters.id)

    # Set up a conditional group to ensure convergence of uncertainty
    if conditional_group is None:

        conditional_group = groups.ConditionalGroup(f'conditional_group{id_suffix}')
        conditional_group.max_iterations = 100

        if workflow_options.convergence_mode != WorkflowOptions.ConvergenceMode.NoChecks:

            condition = groups.ConditionalGroup.Condition()

            condition.left_hand_value = ProtocolPath('value.uncertainty',
                                                     conditional_group.id,
                                                     analysis_protocol.id)

            condition.right_hand_value = ProtocolPath('target_uncertainty', 'global')

            condition.condition_type = groups.ConditionalGroup.ConditionType.LessThan

            conditional_group.add_condition(condition)

    conditional_group.add_protocols(production_simulation, analysis_protocol)

    # Point the analyse protocol to the correct data source
    if isinstance(analysis_protocol, analysis.AverageTrajectoryProperty):
        analysis_protocol.input_coordinate_file = ProtocolPath('coordinate_file_path', build_coordinates.id)
        analysis_protocol.trajectory_path = ProtocolPath('trajectory_file_path', production_simulation.id)

    elif isinstance(analysis_protocol, analysis.ExtractAverageStatistic):
        analysis_protocol.statistics_path = ProtocolPath('statistics_file_path', production_simulation.id)

    else:
        raise ValueError('The analysis protocol must inherit from either the '
                         'AverageTrajectoryProperty or ExtractAverageStatistic '
                         'protocols.')

    # Finally, extract uncorrelated data
    statistical_inefficiency = ProtocolPath('statistical_inefficiency', conditional_group.id, analysis_protocol.id)
    equilibration_index = ProtocolPath('equilibration_index', conditional_group.id, analysis_protocol.id)
    coordinate_file = ProtocolPath('output_coordinate_file', conditional_group.id, production_simulation.id)
    trajectory_path = ProtocolPath('trajectory_file_path', conditional_group.id, production_simulation.id)
    statistics_path = ProtocolPath('statistics_file_path', conditional_group.id, production_simulation.id)

    extract_uncorrelated_trajectory = analysis.ExtractUncorrelatedTrajectoryData(f'extract_traj{id_suffix}')
    extract_uncorrelated_trajectory.statistical_inefficiency = statistical_inefficiency
    extract_uncorrelated_trajectory.equilibration_index = equilibration_index
    extract_uncorrelated_trajectory.input_coordinate_file = coordinate_file
    extract_uncorrelated_trajectory.input_trajectory_path = trajectory_path

    extract_uncorrelated_statistics = analysis.ExtractUncorrelatedStatisticsData(f'extract_stats{id_suffix}')
    extract_uncorrelated_statistics.statistical_inefficiency = statistical_inefficiency
    extract_uncorrelated_statistics.equilibration_index = equilibration_index
    extract_uncorrelated_statistics.input_statistics_path = statistics_path

    # Build the object which defines which pieces of simulation data to store.
    output_to_store = WorkflowSimulationDataToStore()

    output_to_store.total_number_of_molecules = ProtocolPath('final_number_of_molecules', build_coordinates.id)
    output_to_store.statistical_inefficiency = statistical_inefficiency
    output_to_store.statistics_file_path = ProtocolPath('output_statistics_path', extract_uncorrelated_statistics.id)
    output_to_store.trajectory_file_path = ProtocolPath('output_trajectory_path', extract_uncorrelated_trajectory.id)
    output_to_store.coordinate_file_path = coordinate_file

    # Define where the final values come from.
    final_value_source = ProtocolPath('value', conditional_group.id, analysis_protocol.id)

    base_protocols = BaseSimulationProtocols(build_coordinates,
                                             assign_parameters,
                                             energy_minimisation,
                                             equilibration_simulation,
                                             production_simulation,
                                             analysis_protocol,
                                             conditional_group,
                                             extract_uncorrelated_trajectory,
                                             extract_uncorrelated_statistics)

    return base_protocols, final_value_source, output_to_store


def _get_default_gradient_reweighting_schema(observable_values, id_prefix=''):
    """Generates the schema for a `ReweightWithMBARProtocol` with settings
    appropriate for use in reweighting observables for direct gradients
    evaluations.

    Parameters
    ----------
    observable_values: ProtocolPath, optional
        The path to the observables whose gradient is to be determined.
    id_prefix: str
        An optional string to prepend to the id of the protocol.

    Returns
    -------
    ProtocolSchema
        The created schema.
    """

    mbar_protocol = reweighting.ReweightWithMBARProtocol(f'{id_prefix}gradient_mbar')
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
                                     perturbation_scale=1.0e-4,
                                     substance_source=None,
                                     id_prefix=''):
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
    substance_source: ProtocolPath, optional
        An optional protocol path to the substance whose gradient
        is being estimated. If None, the global property substance
        is used.
    id_prefix: str
        An optional string to prepend to the beginning of each of the
        protocol ids.

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
    reduced_potentials = gradients.GradientReducedPotentials(f'{id_prefix}gradient_reduced_potentials_'
                                                             f'$({replicator_id})')

    reduced_potentials.substance = (ProtocolPath('substance', 'global') if substance_source is None
                                                                        else substance_source)
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
        template_reweighting_schema = _get_default_gradient_reweighting_schema(observable_values, id_prefix)

    reverse_mbar_schema = copy.deepcopy(template_reweighting_schema)
    reverse_mbar_schema.id = f'{id_prefix}reverse_{reverse_mbar_schema.id}_$({replicator_id})'

    reverse_mbar = available_protocols[reverse_mbar_schema.type](reverse_mbar_schema.id)
    reverse_mbar.schema = reverse_mbar_schema
    reverse_mbar.reference_reduced_potentials = ProtocolPath('reference_potential_paths', reduced_potentials.id)
    reverse_mbar.target_reduced_potentials = [ProtocolPath('reverse_potentials_path', reduced_potentials.id)]
    reverse_mbar.bootstrap_iterations = min(1, reverse_mbar.bootstrap_iterations)

    forward_mbar_schema = copy.deepcopy(template_reweighting_schema)
    forward_mbar_schema.id = f'{id_prefix}forward_{forward_mbar_schema.id}_$({replicator_id})'

    forward_mbar = available_protocols[forward_mbar_schema.type](forward_mbar_schema.id)
    forward_mbar.schema = forward_mbar_schema
    forward_mbar.reference_reduced_potentials = ProtocolPath('reference_potential_paths', reduced_potentials.id)
    forward_mbar.target_reduced_potentials = [ProtocolPath('forward_potentials_path', reduced_potentials.id)]
    forward_mbar.bootstrap_iterations = min(1, reverse_mbar.bootstrap_iterations)

    # Set up the protocol which will actually evaluate the parameter gradient
    # using the central difference method.
    central_difference = gradients.CentralDifferenceGradient(f'{id_prefix}central_difference_$({replicator_id})')
    central_difference.parameter_key = ReplicatorValue(replicator_id)
    central_difference.reverse_observable_value = ProtocolPath('value', reverse_mbar.id)
    central_difference.forward_observable_value = ProtocolPath('value', forward_mbar.id)
    central_difference.reverse_parameter_value = ProtocolPath('reverse_parameter_value', reduced_potentials.id)
    central_difference.forward_parameter_value = ProtocolPath('forward_parameter_value', reduced_potentials.id)

    # Assemble all of the protocols into a convenient group wrapper.
    gradient_group = groups.ProtocolGroup(f'{id_prefix}gradient_group_$({replicator_id})')
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
