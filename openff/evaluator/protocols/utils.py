"""
A set of utilities for setting up property estimation workflows.
"""
import copy
from collections import namedtuple

from openff.evaluator import unit
from openff.evaluator.attributes import UNDEFINED, PlaceholderValue
from openff.evaluator.datasets import PropertyPhase
from openff.evaluator.protocols import (
    analysis,
    coordinates,
    forcefield,
    gradients,
    groups,
    openmm,
    reweighting,
    storage,
)
from openff.evaluator.storage.data import StoredSimulationData
from openff.evaluator.thermodynamics import Ensemble
from openff.evaluator.utils.statistics import ObservableType
from openff.evaluator.workflow import registered_workflow_protocols
from openff.evaluator.workflow.schemas import ProtocolReplicator
from openff.evaluator.workflow.utils import ProtocolPath, ReplicatorValue

BaseReweightingProtocols = namedtuple(
    "BaseReweightingProtocols",
    "unpack_stored_data "
    "analysis_protocol "
    "decorrelate_statistics "
    "decorrelate_trajectory "
    "concatenate_trajectories "
    "concatenate_statistics "
    "build_reference_system "
    "reduced_reference_potential "
    "build_target_system "
    "reduced_target_potential "
    "mbar_protocol ",
)


BaseSimulationProtocols = namedtuple(
    "BaseSimulationProtocols",
    "build_coordinates "
    "assign_parameters "
    "energy_minimisation "
    "equilibration_simulation "
    "production_simulation "
    "analysis_protocol "
    "converge_uncertainty "
    "extract_uncorrelated_trajectory "
    "extract_uncorrelated_statistics ",
)


def generate_base_reweighting_protocols(
    analysis_protocol, mbar_protocol, replicator_id="data_repl", id_suffix="",
):
    """Constructs a set of protocols which, when combined in a workflow schema,
    may be executed to reweight a set of existing data to estimate a particular
    property. The reweighted observable of interest will be calculated by
    following the passed in `analysis_protocol`.

    Parameters
    ----------
    analysis_protocol: AveragePropertyProtocol
        The protocol which will take input from the stored data,
        and generate a set of observables to reweight.
    mbar_protocol: BaseReweightingProtocol
        A template mbar reweighting protocol, which has it's reference
        observables already set. This method will automatically set the
        reduced potentials on this object.
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

    assert f"$({replicator_id})" in analysis_protocol.id
    assert f"$({replicator_id})" not in mbar_protocol.id

    replicator_suffix = "_$({}){}".format(replicator_id, id_suffix)

    # Unpack all the of the stored data.
    unpack_stored_data = storage.UnpackStoredSimulationData(
        "unpack_data{}".format(replicator_suffix)
    )
    unpack_stored_data.simulation_data_path = ReplicatorValue(replicator_id)

    # The autocorrelation time of each of the stored files will be calculated for this property
    # using the passed in analysis protocol.
    if isinstance(analysis_protocol, analysis.ExtractAverageStatistic):

        analysis_protocol.statistics_path = ProtocolPath(
            "statistics_file_path", unpack_stored_data.id
        )

    elif isinstance(analysis_protocol, analysis.AverageTrajectoryProperty):

        analysis_protocol.input_coordinate_file = ProtocolPath(
            "coordinate_file_path", unpack_stored_data.id
        )
        analysis_protocol.trajectory_path = ProtocolPath(
            "trajectory_file_path", unpack_stored_data.id
        )

    # Decorrelate the frames of the stored trajectory and statistics arrays.
    decorrelate_statistics = analysis.ExtractUncorrelatedStatisticsData(
        "decorrelate_stats{}".format(replicator_suffix)
    )
    decorrelate_statistics.statistical_inefficiency = ProtocolPath(
        "statistical_inefficiency", analysis_protocol.id
    )
    decorrelate_statistics.equilibration_index = ProtocolPath(
        "equilibration_index", analysis_protocol.id
    )
    decorrelate_statistics.input_statistics_path = ProtocolPath(
        "statistics_file_path", unpack_stored_data.id
    )

    decorrelate_trajectory = analysis.ExtractUncorrelatedTrajectoryData(
        "decorrelate_traj{}".format(replicator_suffix)
    )
    decorrelate_trajectory.statistical_inefficiency = ProtocolPath(
        "statistical_inefficiency", analysis_protocol.id
    )
    decorrelate_trajectory.equilibration_index = ProtocolPath(
        "equilibration_index", analysis_protocol.id
    )
    decorrelate_trajectory.input_coordinate_file = ProtocolPath(
        "coordinate_file_path", unpack_stored_data.id
    )
    decorrelate_trajectory.input_trajectory_path = ProtocolPath(
        "trajectory_file_path", unpack_stored_data.id
    )

    # Stitch together all of the trajectories
    join_trajectories = reweighting.ConcatenateTrajectories("concat_traj" + id_suffix)
    join_trajectories.input_coordinate_paths = ProtocolPath(
        "coordinate_file_path", unpack_stored_data.id
    )
    join_trajectories.input_trajectory_paths = ProtocolPath(
        "output_trajectory_path", decorrelate_trajectory.id
    )

    join_statistics = reweighting.ConcatenateStatistics("concat_stats" + id_suffix)
    join_statistics.input_statistics_paths = ProtocolPath(
        "output_statistics_path", decorrelate_statistics.id
    )

    # Calculate the reduced potentials for each of the reference states.
    build_reference_system = forcefield.BaseBuildSystem(
        "build_system{}".format(replicator_suffix)
    )
    build_reference_system.force_field_path = ProtocolPath(
        "force_field_path", unpack_stored_data.id
    )
    build_reference_system.substance = ProtocolPath("substance", unpack_stored_data.id)
    build_reference_system.coordinate_file_path = ProtocolPath(
        "coordinate_file_path", unpack_stored_data.id
    )

    reduced_reference_potential = openmm.OpenMMReducedPotentials(
        "reduced_potential{}".format(replicator_suffix)
    )
    reduced_reference_potential.system_path = ProtocolPath(
        "system_path", build_reference_system.id
    )
    reduced_reference_potential.thermodynamic_state = ProtocolPath(
        "thermodynamic_state", unpack_stored_data.id
    )
    reduced_reference_potential.coordinate_file_path = ProtocolPath(
        "coordinate_file_path", unpack_stored_data.id
    )
    reduced_reference_potential.trajectory_file_path = ProtocolPath(
        "output_trajectory_path", join_trajectories.id
    )
    reduced_reference_potential.kinetic_energies_path = ProtocolPath(
        "output_statistics_path", join_statistics.id
    )

    # Calculate the reduced potential of the target state.
    build_target_system = forcefield.BaseBuildSystem("build_system_target" + id_suffix)
    build_target_system.force_field_path = ProtocolPath("force_field_path", "global")
    build_target_system.substance = ProtocolPath("substance", "global")
    build_target_system.coordinate_file_path = ProtocolPath(
        "output_coordinate_path", join_trajectories.id
    )

    reduced_target_potential = openmm.OpenMMReducedPotentials(
        "reduced_potential_target" + id_suffix
    )
    reduced_target_potential.thermodynamic_state = ProtocolPath(
        "thermodynamic_state", "global"
    )
    reduced_target_potential.system_path = ProtocolPath(
        "system_path", build_target_system.id
    )
    reduced_target_potential.coordinate_file_path = ProtocolPath(
        "output_coordinate_path", join_trajectories.id
    )
    reduced_target_potential.trajectory_file_path = ProtocolPath(
        "output_trajectory_path", join_trajectories.id
    )
    reduced_target_potential.kinetic_energies_path = ProtocolPath(
        "output_statistics_path", join_statistics.id
    )

    # Finally, apply MBAR to get the reweighted value.
    mbar_protocol.reference_reduced_potentials = ProtocolPath(
        "statistics_file_path", reduced_reference_potential.id
    )
    mbar_protocol.target_reduced_potentials = ProtocolPath(
        "statistics_file_path", reduced_target_potential.id
    )

    if (
        isinstance(mbar_protocol, reweighting.ReweightStatistics)
        and mbar_protocol.statistics_type != ObservableType.PotentialEnergy
        and mbar_protocol.statistics_type != ObservableType.TotalEnergy
        and mbar_protocol.statistics_type != ObservableType.Enthalpy
        and mbar_protocol.statistics_type != ObservableType.ReducedPotential
    ):

        mbar_protocol.statistics_paths = ProtocolPath(
            "output_statistics_path", decorrelate_statistics.id
        )

    elif isinstance(mbar_protocol, reweighting.ReweightStatistics):

        mbar_protocol.statistics_paths = [
            ProtocolPath("statistics_file_path", reduced_target_potential.id)
        ]
        mbar_protocol.frame_counts = ProtocolPath(
            "number_of_uncorrelated_samples", decorrelate_statistics.id
        )

    base_protocols = BaseReweightingProtocols(
        unpack_stored_data,
        analysis_protocol,
        decorrelate_statistics,
        decorrelate_trajectory,
        join_trajectories,
        join_statistics,
        build_reference_system,
        reduced_reference_potential,
        build_target_system,
        reduced_target_potential,
        mbar_protocol,
    )

    # Create the replicator object.
    component_replicator = ProtocolReplicator(replicator_id=replicator_id)
    component_replicator.template_values = ProtocolPath("full_system_data", "global")

    return base_protocols, component_replicator


def generate_base_simulation_protocols(
    analysis_protocol,
    use_target_uncertainty,
    id_suffix="",
    conditional_group=None,
    n_molecules=1000,
):
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
           using a timestep of 2fs.

        5) Within a conditional group (up to a maximum of 100 times):

            5a) Run a longer NPT production simulation for 1000000 steps using a timestep of 2fs

            5b) Extract the average value of an observable and it's uncertainty.

            5c) If a convergence mode is set by the options, check if the target uncertainty has been met.
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
    use_target_uncertainty: bool
        Whether to run the simulation until the observable is
        estimated to within the target uncertainty.
    id_suffix: str
        A string suffix to append to each of the protocol ids.
    conditional_group: ProtocolGroup, optional
        A custom group to wrap the main simulation / extraction
        protocols within. It is up to the caller of this method to
        manually add the convergence conditions to this group.
        If `None`, a default group with uncertainty convergence
        conditions is automatically constructed.
    n_molecules: int
        The number of molecules to use in the workflow.

    Returns
    -------
    BaseSimulationProtocols
        A named tuple of the generated protocols.
    ProtocolPath
        A reference to the final value of the estimated observable
        and its uncertainty (a `pint.Measurement`).
    StoredSimulationData
        An object which describes the default data from a simulation to store,
        such as the uncorrelated statistics and configurations.
    """

    assert isinstance(analysis_protocol, analysis.AveragePropertyProtocol)

    build_coordinates = coordinates.BuildCoordinatesPackmol(
        f"build_coordinates{id_suffix}"
    )
    build_coordinates.substance = ProtocolPath("substance", "global")
    build_coordinates.max_molecules = n_molecules

    assign_parameters = forcefield.BaseBuildSystem(f"assign_parameters{id_suffix}")
    assign_parameters.force_field_path = ProtocolPath("force_field_path", "global")
    assign_parameters.coordinate_file_path = ProtocolPath(
        "coordinate_file_path", build_coordinates.id
    )
    assign_parameters.substance = ProtocolPath("output_substance", build_coordinates.id)

    # Equilibration
    energy_minimisation = openmm.OpenMMEnergyMinimisation(
        f"energy_minimisation{id_suffix}"
    )
    energy_minimisation.input_coordinate_file = ProtocolPath(
        "coordinate_file_path", build_coordinates.id
    )
    energy_minimisation.system_path = ProtocolPath("system_path", assign_parameters.id)

    equilibration_simulation = openmm.OpenMMSimulation(
        f"equilibration_simulation{id_suffix}"
    )
    equilibration_simulation.ensemble = Ensemble.NPT
    equilibration_simulation.steps_per_iteration = 100000
    equilibration_simulation.output_frequency = 5000
    equilibration_simulation.timestep = 2.0 * unit.femtosecond
    equilibration_simulation.thermodynamic_state = ProtocolPath(
        "thermodynamic_state", "global"
    )
    equilibration_simulation.input_coordinate_file = ProtocolPath(
        "output_coordinate_file", energy_minimisation.id
    )
    equilibration_simulation.system_path = ProtocolPath(
        "system_path", assign_parameters.id
    )

    # Production
    production_simulation = openmm.OpenMMSimulation(f"production_simulation{id_suffix}")
    production_simulation.ensemble = Ensemble.NPT
    production_simulation.steps_per_iteration = 1000000
    production_simulation.output_frequency = 2000
    production_simulation.timestep = 2.0 * unit.femtosecond
    production_simulation.thermodynamic_state = ProtocolPath(
        "thermodynamic_state", "global"
    )
    production_simulation.input_coordinate_file = ProtocolPath(
        "output_coordinate_file", equilibration_simulation.id
    )
    production_simulation.system_path = ProtocolPath(
        "system_path", assign_parameters.id
    )

    # Set up a conditional group to ensure convergence of uncertainty
    if conditional_group is None:

        conditional_group = groups.ConditionalGroup(f"conditional_group{id_suffix}")
        conditional_group.max_iterations = 100

        if use_target_uncertainty:

            condition = groups.ConditionalGroup.Condition()
            condition.right_hand_value = ProtocolPath("target_uncertainty", "global")
            condition.type = groups.ConditionalGroup.Condition.Type.LessThan
            condition.left_hand_value = ProtocolPath(
                "value.error", conditional_group.id, analysis_protocol.id
            )

            conditional_group.add_condition(condition)

            # Make sure the simulation gets extended after each iteration.
            production_simulation.total_number_of_iterations = ProtocolPath(
                "current_iteration", conditional_group.id
            )

    conditional_group.add_protocols(production_simulation, analysis_protocol)

    # Point the analyse protocol to the correct data source
    if isinstance(analysis_protocol, analysis.AverageTrajectoryProperty):
        analysis_protocol.input_coordinate_file = ProtocolPath(
            "coordinate_file_path", build_coordinates.id
        )
        analysis_protocol.trajectory_path = ProtocolPath(
            "trajectory_file_path", production_simulation.id
        )

    elif isinstance(analysis_protocol, analysis.ExtractAverageStatistic):
        analysis_protocol.statistics_path = ProtocolPath(
            "statistics_file_path", production_simulation.id
        )

    else:
        raise ValueError(
            "The analysis protocol must inherit from either the "
            "AverageTrajectoryProperty or ExtractAverageStatistic "
            "protocols."
        )

    # Finally, extract uncorrelated data
    statistical_inefficiency = ProtocolPath(
        "statistical_inefficiency", conditional_group.id, analysis_protocol.id
    )
    equilibration_index = ProtocolPath(
        "equilibration_index", conditional_group.id, analysis_protocol.id
    )
    coordinate_file = ProtocolPath(
        "output_coordinate_file", conditional_group.id, production_simulation.id
    )
    trajectory_path = ProtocolPath(
        "trajectory_file_path", conditional_group.id, production_simulation.id
    )
    statistics_path = ProtocolPath(
        "statistics_file_path", conditional_group.id, production_simulation.id
    )

    extract_uncorrelated_trajectory = analysis.ExtractUncorrelatedTrajectoryData(
        f"extract_traj{id_suffix}"
    )
    extract_uncorrelated_trajectory.statistical_inefficiency = statistical_inefficiency
    extract_uncorrelated_trajectory.equilibration_index = equilibration_index
    extract_uncorrelated_trajectory.input_coordinate_file = coordinate_file
    extract_uncorrelated_trajectory.input_trajectory_path = trajectory_path

    extract_uncorrelated_statistics = analysis.ExtractUncorrelatedStatisticsData(
        f"extract_stats{id_suffix}"
    )
    extract_uncorrelated_statistics.statistical_inefficiency = statistical_inefficiency
    extract_uncorrelated_statistics.equilibration_index = equilibration_index
    extract_uncorrelated_statistics.input_statistics_path = statistics_path

    # Build the object which defines which pieces of simulation data to store.
    output_to_store = StoredSimulationData()

    output_to_store.thermodynamic_state = ProtocolPath("thermodynamic_state", "global")
    output_to_store.property_phase = PropertyPhase.Liquid

    output_to_store.force_field_id = PlaceholderValue()

    output_to_store.number_of_molecules = ProtocolPath(
        "output_number_of_molecules", build_coordinates.id
    )
    output_to_store.substance = ProtocolPath("output_substance", build_coordinates.id)
    output_to_store.statistical_inefficiency = statistical_inefficiency
    output_to_store.statistics_file_name = ProtocolPath(
        "output_statistics_path", extract_uncorrelated_statistics.id
    )
    output_to_store.trajectory_file_name = ProtocolPath(
        "output_trajectory_path", extract_uncorrelated_trajectory.id
    )
    output_to_store.coordinate_file_name = coordinate_file

    output_to_store.source_calculation_id = PlaceholderValue()

    # Define where the final values come from.
    final_value_source = ProtocolPath(
        "value", conditional_group.id, analysis_protocol.id
    )

    base_protocols = BaseSimulationProtocols(
        build_coordinates,
        assign_parameters,
        energy_minimisation,
        equilibration_simulation,
        production_simulation,
        analysis_protocol,
        conditional_group,
        extract_uncorrelated_trajectory,
        extract_uncorrelated_statistics,
    )

    return base_protocols, final_value_source, output_to_store


def generate_gradient_protocol_group(
    template_reweighting_protocol,
    force_field_path,
    coordinate_file_path,
    trajectory_file_path,
    statistics_file_path,
    replicator_id="repl",
    substance_source=None,
    id_suffix="",
    enable_pbc=True,
    effective_sample_indices=None,
):
    """Constructs a set of protocols which, when combined in a workflow schema,
    may be executed to reweight a set of existing data to estimate a particular
    property. The reweighted observable of interest will be calculated by
    following the passed in `analysis_protocol`.

    Parameters
    ----------
    template_reweighting_protocol: BaseMBARProtocol
        A template protocol which will be used to reweight the observable of
        interest to small perturbations to the parameter of interest. These
        will then be used to calculate the finite difference gradient.

        The template *must* have it's `reference_reduced_potentials` input set.
        The `target_reduced_potentials` input will be set automatically by this
        function.

        In the case that the template is of type `ReweightStatistics` and the
        observable is an energy, the statistics path will automatically be pointed
        to the energies evaluated using the perturbed parameter as opposed to the
        energy measured during the reference simulation.
    force_field_path: ProtocolPath
        The path to the force field parameters which the observables are being
         estimated at.
    coordinate_file_path: ProtocolPath
        A path to the initial coordinates of the simulation trajectory which
        was used to estimate the observable of interest.
    trajectory_file_path: ProtocolPath
        A path to the simulation trajectory which was used
        to estimate the observable of interest.
    statistics_file_path: ProtocolPath, optional
        A path to the statistics which were generated alongside
        the trajectory passed to the `trajectory_file_path`. These
        should have been generated using the passed `force_field_path`.
    replicator_id: str
        A unique id which will be used for the protocol replicator which will
        replicate this group for every parameter of interest.
    substance_source: PlaceholderValue, optional
        An optional protocol path to the substance whose gradient
        is being estimated. If None, the global property substance
        is used.
    id_suffix: str
        An optional string to append to the end of each of the
        protocol ids.
    enable_pbc: bool
        If true, periodic boundary conditions are employed when recalculating
        the reduced potentials.
    effective_sample_indices: ProtocolPath, optional
        A placeholder variable which in future will ensure that only samples
        with a non-zero weight are included in the gradient calculation.

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

    assert isinstance(template_reweighting_protocol, reweighting.BaseMBARProtocol)
    assert template_reweighting_protocol.reference_reduced_potentials is not None
    assert template_reweighting_protocol.reference_reduced_potentials != UNDEFINED

    id_suffix = f"_$({replicator_id}){id_suffix}"

    # Set values of the optional parameters.
    substance_source = (
        ProtocolPath("substance", "global")
        if substance_source is None
        else substance_source
    )
    effective_sample_indices = (
        effective_sample_indices if effective_sample_indices is not None else []
    )

    # Define the protocol which will evaluate the reduced potentials of the
    # reference, forward and reverse states using only a subset of the full
    # force field.
    reduced_potentials = openmm.OpenMMGradientPotentials(
        f"gradient_reduced_potentials{id_suffix}"
    )
    reduced_potentials.substance = substance_source
    reduced_potentials.thermodynamic_state = ProtocolPath(
        "thermodynamic_state", "global"
    )
    reduced_potentials.force_field_path = force_field_path
    reduced_potentials.statistics_path = statistics_file_path
    reduced_potentials.trajectory_file_path = trajectory_file_path
    reduced_potentials.coordinate_file_path = coordinate_file_path
    reduced_potentials.parameter_key = ReplicatorValue(replicator_id)
    reduced_potentials.enable_pbc = enable_pbc
    reduced_potentials.effective_sample_indices = effective_sample_indices

    # Set up the protocols which will actually reweight the value of the
    # observable to the forward and reverse states.
    template_reweighting_protocol.bootstrap_iterations = 1
    template_reweighting_protocol.required_effective_samples = 0

    # We need to make sure we use the observable evaluated at the target state
    # if the observable depends on the parameter being reweighted.
    use_target_state_energies = isinstance(
        template_reweighting_protocol, reweighting.ReweightStatistics
    ) and (
        template_reweighting_protocol.statistics_type == ObservableType.PotentialEnergy
        or template_reweighting_protocol.statistics_type
        == ObservableType.ReducedPotential
        or template_reweighting_protocol.statistics_type == ObservableType.TotalEnergy
        or template_reweighting_protocol.statistics_type == ObservableType.Enthalpy
    )

    template_reweighting_schema = template_reweighting_protocol.schema

    # Create the reweighting protocols from the template schema.
    reverse_mbar_schema = copy.deepcopy(template_reweighting_schema)
    reverse_mbar_schema.id = f"reverse_reweight{id_suffix}"
    reverse_mbar = registered_workflow_protocols[reverse_mbar_schema.type](
        reverse_mbar_schema.id
    )
    reverse_mbar.schema = reverse_mbar_schema
    reverse_mbar.target_reduced_potentials = ProtocolPath(
        "reverse_potentials_path", reduced_potentials.id
    )

    forward_mbar_schema = copy.deepcopy(template_reweighting_schema)
    forward_mbar_schema.id = f"forward_reweight{id_suffix}"
    forward_mbar = registered_workflow_protocols[forward_mbar_schema.type](
        forward_mbar_schema.id
    )
    forward_mbar.schema = forward_mbar_schema
    forward_mbar.target_reduced_potentials = ProtocolPath(
        "forward_potentials_path", reduced_potentials.id
    )

    if use_target_state_energies:

        reverse_mbar.statistics_paths = [
            ProtocolPath("reverse_potentials_path", reduced_potentials.id)
        ]
        forward_mbar.statistics_paths = [
            ProtocolPath("forward_potentials_path", reduced_potentials.id)
        ]

    # Set up the protocol which will actually evaluate the parameter gradient
    # using the central difference method.
    central_difference = gradients.CentralDifferenceGradient(
        f"central_difference{id_suffix}"
    )
    central_difference.parameter_key = ReplicatorValue(replicator_id)
    central_difference.reverse_observable_value = ProtocolPath("value", reverse_mbar.id)
    central_difference.forward_observable_value = ProtocolPath("value", forward_mbar.id)
    central_difference.reverse_parameter_value = ProtocolPath(
        "reverse_parameter_value", reduced_potentials.id
    )
    central_difference.forward_parameter_value = ProtocolPath(
        "forward_parameter_value", reduced_potentials.id
    )

    # Assemble all of the protocols into a convenient group.
    gradient_group = groups.ProtocolGroup(f"gradient_group{id_suffix}")
    gradient_group.add_protocols(
        reduced_potentials, reverse_mbar, forward_mbar, central_difference
    )

    # Create the replicator which will copy the group for each parameter gradient
    # which will be calculated.
    parameter_replicator = ProtocolReplicator(replicator_id=replicator_id)
    parameter_replicator.template_values = ProtocolPath(
        "parameter_gradient_keys", "global"
    )

    return (
        gradient_group,
        parameter_replicator,
        ProtocolPath("gradient", gradient_group.id, central_difference.id),
    )
