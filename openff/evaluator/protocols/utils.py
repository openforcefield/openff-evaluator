"""
A set of utilities for setting up property estimation workflows.
"""

from dataclasses import astuple, dataclass
from typing import Generic, Optional, Tuple, TypeVar

from openff.units import unit

from openff.evaluator.attributes import UNDEFINED, PlaceholderValue
from openff.evaluator.datasets import PropertyPhase
from openff.evaluator.layers.equilibration import EquilibrationProperty
from openff.evaluator.protocols import (
    analysis,
    coordinates,
    forcefield,
    gradients,
    groups,
    miscellaneous,
    openmm,
    reweighting,
    storage,
)
from openff.evaluator.protocols.groups import ConditionalGroup
from openff.evaluator.protocols.miscellaneous import (
    AbsoluteValue,
    MaximumValue,
    MultiplyValue,
)
from openff.evaluator.storage.data import StoredEquilibrationData, StoredSimulationData
from openff.evaluator.thermodynamics import Ensemble
from openff.evaluator.utils.observables import ObservableType
from openff.evaluator.workflow import ProtocolGroup
from openff.evaluator.workflow.attributes import ConditionAggregationBehavior
from openff.evaluator.workflow.schemas import ProtocolReplicator
from openff.evaluator.workflow.utils import ProtocolPath, ReplicatorValue

S = TypeVar("S", bound=analysis.BaseAverageObservable)
T = TypeVar("T", bound=reweighting.BaseMBARProtocol)


@dataclass
class EquilibrationProtocols:
    """The common set of protocols for equilibration"""

    build_coordinates: coordinates.BuildCoordinatesPackmol
    assign_parameters: forcefield.BaseBuildSystem
    energy_minimisation: openmm.OpenMMEnergyMinimisation
    converge_uncertainty: ProtocolGroup

    def __iter__(self):
        yield from astuple(self)


@dataclass
class SimulationProtocols(Generic[S]):
    """The common set of protocols which would be required to estimate an observable
    by running a new molecule simulation."""

    build_coordinates: coordinates.BuildCoordinatesPackmol
    assign_parameters: forcefield.BaseBuildSystem
    energy_minimisation: openmm.OpenMMEnergyMinimisation
    equilibration_simulation: openmm.OpenMMSimulation
    production_simulation: openmm.OpenMMSimulation
    analysis_protocol: S
    converge_uncertainty: ProtocolGroup
    decorrelate_trajectory: analysis.DecorrelateTrajectory
    decorrelate_observables: analysis.DecorrelateObservables

    def __iter__(self):
        yield from astuple(self)


@dataclass
class ReweightingProtocols(Generic[S, T]):
    """The common set of protocols which would be required to re-weight an observable
    from cached simulation data."""

    unpack_stored_data: storage.UnpackStoredSimulationData

    join_trajectories: reweighting.ConcatenateTrajectories
    join_observables: reweighting.ConcatenateObservables

    build_reference_system: forcefield.BaseBuildSystem
    evaluate_reference_potential: reweighting.BaseEvaluateEnergies

    build_target_system: forcefield.BaseBuildSystem
    evaluate_target_potential: reweighting.BaseEvaluateEnergies

    statistical_inefficiency: S
    replicate_statistics: miscellaneous.DummyProtocol

    decorrelate_reference_potential: analysis.DecorrelateObservables
    decorrelate_target_potential: analysis.DecorrelateObservables

    decorrelate_observable: analysis.DecorrelateObservables
    zero_gradients: Optional[gradients.ZeroGradients]

    reweight_observable: T

    def __iter__(self):
        yield from astuple(self)


def generate_base_reweighting_protocols(
    statistical_inefficiency: S,
    reweight_observable: T,
    replicator_id: str = "data_replicator",
    id_suffix: str = "",
) -> Tuple[ReweightingProtocols[S, T], ProtocolReplicator]:
    """Constructs a set of protocols which, when combined in a workflow schema, may be
    executed to reweight a set of cached simulation data to estimate the average
    value of an observable.

    Parameters
    ----------
    statistical_inefficiency
        The protocol which will be used to compute the statistical inefficiency and
        equilibration time of the observable of interest. This information will be
        used to decorrelate the cached data prior to reweighting.
    reweight_observable
        The MBAR reweighting protocol to use to reweight the observable to the target
        state. This method will automatically set the reduced potentials on the
        object.
    replicator_id: str
        The id to use for the cached data replicator.
    id_suffix: str
        A string suffix to append to each of the protocol ids.

    Returns
    -------
        The protocols to add to the workflow, a reference to the average value of the
        estimated observable (an ``Observable`` object), and the replicator which will
        clone the workflow for each piece of cached simulation data.
    """

    # Create the replicator which will apply these protocol once for each piece of
    # cached simulation data.
    data_replicator = ProtocolReplicator(replicator_id=replicator_id)
    data_replicator.template_values = ProtocolPath("full_system_data", "global")

    # Validate the inputs.
    assert isinstance(statistical_inefficiency, analysis.BaseAverageObservable)

    assert data_replicator.placeholder_id in statistical_inefficiency.id
    assert data_replicator.placeholder_id not in reweight_observable.id

    replicator_suffix = f"_{data_replicator.placeholder_id}{id_suffix}"

    # Unpack all the of the stored data.
    unpack_stored_data = storage.UnpackStoredSimulationData(
        "unpack_data{}".format(replicator_suffix)
    )
    unpack_stored_data.simulation_data_path = ReplicatorValue(replicator_id)

    # Join the individual trajectories together.
    join_trajectories = reweighting.ConcatenateTrajectories(
        f"join_trajectories{id_suffix}"
    )
    join_trajectories.input_coordinate_paths = ProtocolPath(
        "coordinate_file_path", unpack_stored_data.id
    )
    join_trajectories.input_trajectory_paths = ProtocolPath(
        "trajectory_file_path", unpack_stored_data.id
    )
    join_observables = reweighting.ConcatenateObservables(
        f"join_observables{id_suffix}"
    )
    join_observables.input_observables = ProtocolPath(
        "observables", unpack_stored_data.id
    )

    # Calculate the reduced potentials for each of the reference states.
    build_reference_system = forcefield.BaseBuildSystem(
        f"build_system{replicator_suffix}"
    )
    build_reference_system.force_field_path = ProtocolPath(
        "force_field_path", unpack_stored_data.id
    )
    build_reference_system.coordinate_file_path = ProtocolPath(
        "coordinate_file_path", unpack_stored_data.id
    )
    build_reference_system.substance = ProtocolPath("substance", unpack_stored_data.id)

    reduced_reference_potential = openmm.OpenMMEvaluateEnergies(
        f"reduced_potential{replicator_suffix}"
    )
    reduced_reference_potential.parameterized_system = ProtocolPath(
        "parameterized_system", build_reference_system.id
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

    # Calculate the reduced potential of the target state.
    build_target_system = forcefield.BaseBuildSystem(f"build_system_target{id_suffix}")
    build_target_system.force_field_path = ProtocolPath("force_field_path", "global")
    build_target_system.substance = ProtocolPath("substance", "global")
    build_target_system.coordinate_file_path = ProtocolPath(
        "output_coordinate_path", join_trajectories.id
    )

    reduced_target_potential = openmm.OpenMMEvaluateEnergies(
        f"reduced_potential_target{id_suffix}"
    )
    reduced_target_potential.thermodynamic_state = ProtocolPath(
        "thermodynamic_state", "global"
    )
    reduced_target_potential.parameterized_system = ProtocolPath(
        "parameterized_system", build_target_system.id
    )
    reduced_target_potential.coordinate_file_path = ProtocolPath(
        "output_coordinate_path", join_trajectories.id
    )
    reduced_target_potential.trajectory_file_path = ProtocolPath(
        "output_trajectory_path", join_trajectories.id
    )
    reduced_target_potential.gradient_parameters = ProtocolPath(
        "parameter_gradient_keys", "global"
    )

    # Compute the observable gradients.
    zero_gradients = gradients.ZeroGradients(f"zero_gradients{id_suffix}")
    zero_gradients.force_field_path = ProtocolPath("force_field_path", "global")
    zero_gradients.gradient_parameters = ProtocolPath(
        "parameter_gradient_keys", "global"
    )

    # Decorrelate the target potentials and observables.
    if not isinstance(statistical_inefficiency, analysis.BaseAverageObservable):
        raise NotImplementedError()

    decorrelate_target_potential = analysis.DecorrelateObservables(
        f"decorrelate_target_potential{id_suffix}"
    )
    decorrelate_target_potential.time_series_statistics = ProtocolPath(
        "time_series_statistics", statistical_inefficiency.id
    )
    decorrelate_target_potential.input_observables = ProtocolPath(
        "output_observables", reduced_target_potential.id
    )

    decorrelate_observable = analysis.DecorrelateObservables(
        f"decorrelate_observable{id_suffix}"
    )
    decorrelate_observable.time_series_statistics = ProtocolPath(
        "time_series_statistics", statistical_inefficiency.id
    )
    decorrelate_observable.input_observables = ProtocolPath(
        "output_observables", zero_gradients.id
    )

    # Decorrelate the reference potentials. Due to a quirk of how workflow replicators
    # work the time series statistics need to be passed via a dummy protocol first.
    #
    # Because the `statistical_inefficiency` and `decorrelate_reference_potential`
    # protocols are replicated by the same replicator the `time_series_statistics`
    # input of `decorrelate_reference_potential_X` will take its value from
    # the `time_series_statistics` output of `statistical_inefficiency_X` rather than
    # as a list of of [statistical_inefficiency_0.time_series_statistics...
    # statistical_inefficiency_N.time_series_statistics]. Passing the statistics via
    # an un-replicated intermediate resolves this.
    replicate_statistics = miscellaneous.DummyProtocol(
        f"replicated_statistics{id_suffix}"
    )
    replicate_statistics.input_value = ProtocolPath(
        "time_series_statistics", statistical_inefficiency.id
    )

    decorrelate_reference_potential = analysis.DecorrelateObservables(
        f"decorrelate_reference_potential{replicator_suffix}"
    )
    decorrelate_reference_potential.time_series_statistics = ProtocolPath(
        "output_value", replicate_statistics.id
    )
    decorrelate_reference_potential.input_observables = ProtocolPath(
        "output_observables", reduced_reference_potential.id
    )

    # Finally, apply MBAR to get the reweighted value.
    reweight_observable.reference_reduced_potentials = ProtocolPath(
        "output_observables[ReducedPotential]", decorrelate_reference_potential.id
    )
    reweight_observable.target_reduced_potentials = ProtocolPath(
        "output_observables[ReducedPotential]", decorrelate_target_potential.id
    )
    reweight_observable.observable = ProtocolPath(
        "output_observables", decorrelate_observable.id
    )
    reweight_observable.frame_counts = ProtocolPath(
        "time_series_statistics.n_uncorrelated_points", statistical_inefficiency.id
    )

    protocols = ReweightingProtocols(
        unpack_stored_data,
        #
        join_trajectories,
        join_observables,
        #
        build_reference_system,
        reduced_reference_potential,
        #
        build_target_system,
        reduced_target_potential,
        #
        statistical_inefficiency,
        replicate_statistics,
        #
        decorrelate_reference_potential,
        decorrelate_target_potential,
        #
        decorrelate_observable,
        zero_gradients,
        #
        reweight_observable,
    )

    return protocols, data_replicator


def generate_reweighting_protocols(
    observable_type: ObservableType,
    replicator_id: str = "data_replicator",
    id_suffix: str = "",
) -> Tuple[
    ReweightingProtocols[analysis.AverageObservable, reweighting.ReweightObservable],
    ProtocolReplicator,
]:
    assert observable_type not in [
        ObservableType.KineticEnergy,
        ObservableType.TotalEnergy,
        ObservableType.Enthalpy,
    ]

    statistical_inefficiency = analysis.AverageObservable(
        f"observable_inefficiency_$({replicator_id}){id_suffix}"
    )
    statistical_inefficiency.bootstrap_iterations = 1

    reweight_observable = reweighting.ReweightObservable(
        f"reweight_observable{id_suffix}"
    )

    protocols, data_replicator = generate_base_reweighting_protocols(
        statistical_inefficiency, reweight_observable, replicator_id, id_suffix
    )
    protocols.statistical_inefficiency.observable = ProtocolPath(
        f"observables[{observable_type.value}]", protocols.unpack_stored_data.id
    )

    if (
        observable_type != ObservableType.PotentialEnergy
        and observable_type != ObservableType.TotalEnergy
        and observable_type != ObservableType.Enthalpy
        and observable_type != ObservableType.ReducedPotential
    ):
        protocols.zero_gradients.input_observables = ProtocolPath(
            f"output_observables[{observable_type.value}]",
            protocols.join_observables.id,
        )

    else:
        protocols.zero_gradients = None
        protocols.decorrelate_observable = protocols.decorrelate_target_potential
        protocols.reweight_observable.observable = ProtocolPath(
            f"output_observables[{observable_type.value}]",
            protocols.decorrelate_observable.id,
        )

    return protocols, data_replicator


def generate_equilibration_protocols(
    id_suffix: str = "",
    n_molecules: int = 1000,
    error_tolerances: list[EquilibrationProperty] = [],
    condition_aggregation_behavior: ConditionAggregationBehavior = ConditionAggregationBehavior.All,
    error_on_failure: bool = True,
    max_iterations: int = 100,
) -> Tuple[EquilibrationProtocols, ProtocolPath, StoredSimulationData]:
    """
    Constructs a set of protocols which, when combined in a workflow schema, may be
    executed to equilibrate a system for further simulation to collect production data.

    The protocols returned will:

        1) Build a set of liquid coordinates for the
           property substance using packmol.

        2) Assign a set of smirnoff force field parameters
           to the system.

        3) Perform an energy minimisation on the system.

        4) Run an NPT equilibration until properties converge.

    Parameters
    ----------
    id_suffix: str
        A string suffix to append to each of the protocol ids.
    conditional_group
    """

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
    energy_minimisation.parameterized_system = ProtocolPath(
        "parameterized_system", assign_parameters.id
    )

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
    equilibration_simulation.parameterized_system = ProtocolPath(
        "parameterized_system", assign_parameters.id
    )

    # Set up a conditional group to ensure convergence of uncertainty
    conditional_group = groups.ConditionalGroup(f"conditional_group{id_suffix}")
    conditional_group.max_iterations = max_iterations
    conditional_group.condition_aggregation_behavior = condition_aggregation_behavior
    conditional_group.error_on_failure = error_on_failure

    analysis_protocols = []
    multiplication_protocols = []

    for i, equilibration_property in enumerate(error_tolerances):
        observable_type = equilibration_property.observable_type.value

        # construct analysis protocol
        analysis_protocol = analysis.AverageObservable(
            f"extract_{i}_{observable_type}{id_suffix}"
        )

        analysis_protocol.thermodynamic_state = ProtocolPath(
            "thermodynamic_state", "global"
        )
        analysis_protocol.potential_energies = ProtocolPath(
            f"observables[{ObservableType.PotentialEnergy.value}]",
            equilibration_simulation.id,
        )
        analysis_protocol.observable = ProtocolPath(
            f"observables[{observable_type}]",
            equilibration_simulation.id,
        )
        analysis_protocols.append(analysis_protocol)

        if equilibration_property.n_uncorrelated_samples != UNDEFINED:
            condition = groups.ConditionalGroup.Condition()
            condition.right_hand_value = equilibration_property.n_uncorrelated_samples
            condition.type = groups.ConditionalGroup.Condition.Type.GreaterThan
            condition.left_hand_value = ProtocolPath(
                "time_series_statistics.n_uncorrelated_points",
                conditional_group.id,
                analysis_protocol.id,
            )
            conditional_group.add_condition(condition)

        if equilibration_property.tolerance != UNDEFINED:
            condition = groups.ConditionalGroup.Condition()
            # add checking of absolute error tolerance
            if equilibration_property.absolute_tolerance != UNDEFINED:
                tolerance = equilibration_property.absolute_tolerance.to(
                    equilibration_property.observable_unit
                )
            elif equilibration_property.relative_tolerance != UNDEFINED:
                # add error tolerance
                multiplication_protocol = MultiplyValue(
                    f"multiply_{i}_{observable_type}{id_suffix}"
                )
                multiplication_protocol.value = ProtocolPath(
                    "value.value", conditional_group.id, analysis_protocol.id
                )
                multiplication_protocol.multiplier = (
                    equilibration_property.relative_tolerance
                )

                absolute_protocol = AbsoluteValue(
                    f"absolute_{i}_{observable_type}{id_suffix}"
                )
                absolute_protocol.value = ProtocolPath(
                    "result", conditional_group.id, multiplication_protocol.id
                )
                tolerance = ProtocolPath(
                    "result", conditional_group.id, absolute_protocol.id
                )
                multiplication_protocols.extend(
                    [multiplication_protocol, absolute_protocol]
                )
            else:
                # should never get here
                continue
            condition.right_hand_value = tolerance
            condition.type = groups.ConditionalGroup.Condition.Type.LessThan
            condition.left_hand_value = ProtocolPath(
                "value.error", conditional_group.id, analysis_protocol.id
            )
            conditional_group.add_condition(condition)

        # Make sure the simulation gets extended after each iteration.
        equilibration_simulation.total_number_of_iterations = ProtocolPath(
            "current_iteration", conditional_group.id
        )

    # get the highest? statistical inefficiency for each protocol
    statistical_inefficiency_protocol = MaximumValue(
        f"get_maximum_statistical_inefficiency{id_suffix}"
    )
    statistical_inefficiency_protocol.values = [
        ProtocolPath(
            "time_series_statistics.statistical_inefficiency",
            conditional_group.id,
            analysis_protocol.id,
        )
        for analysis_protocol in analysis_protocols
    ]

    conditional_group.add_protocols(
        equilibration_simulation,
        *analysis_protocols,
        *multiplication_protocols,
        statistical_inefficiency_protocol,
    )

    # Finally, extract uncorrelated data
    # time_series_statistics = ProtocolPath(
    #     "time_series_statistics", conditional_group.id, analysis_protocol.id
    # )
    coordinate_file = ProtocolPath(
        "output_coordinate_file", conditional_group.id, equilibration_simulation.id
    )
    # trajectory_path = ProtocolPath(
    #     "trajectory_file_path", conditional_group.id, equilibration_simulation.id
    # )
    observables = ProtocolPath(
        "observables", conditional_group.id, equilibration_simulation.id
    )

    # Build the object which defines which pieces of simulation data to store.
    output_to_store = StoredEquilibrationData()
    output_to_store.thermodynamic_state = ProtocolPath("thermodynamic_state", "global")
    output_to_store.property_phase = PropertyPhase.Liquid
    output_to_store.force_field_id = PlaceholderValue()
    output_to_store.number_of_molecules = ProtocolPath(
        "output_number_of_molecules", build_coordinates.id
    )
    output_to_store.max_number_of_molecules = n_molecules
    output_to_store.substance = ProtocolPath("output_substance", build_coordinates.id)
    output_to_store.statistical_inefficiency = ProtocolPath(
        "result",
        conditional_group.id,
        statistical_inefficiency_protocol.id,
    )
    output_to_store.observables = observables
    # output_to_store.trajectory_file_name = trajectory_path
    output_to_store.coordinate_file_name = coordinate_file
    output_to_store.source_calculation_id = PlaceholderValue()
    output_to_store.calculation_layer = "EquilibrationLayer"

    # Define where the final values come from.
    # final_value_source = ProtocolPath(
    #     "value", conditional_group.id, analysis_protocol.id
    # )
    final_value_source = UNDEFINED

    protocols = EquilibrationProtocols(
        build_coordinates,
        assign_parameters,
        energy_minimisation,
        # equilibration_simulation,
        # analysis_protocol,
        conditional_group,
    )
    return protocols, final_value_source, output_to_store


def generate_simulation_protocols(
    analysis_protocol: S,
    use_target_uncertainty: bool,
    id_suffix: str = "",
    conditional_group: Optional[ConditionalGroup] = None,
    n_molecules: int = 1000,
) -> Tuple[SimulationProtocols[S], ProtocolPath, StoredSimulationData]:
    """Constructs a set of protocols which, when combined in a workflow schema, may be
    executed to run a single simulation to estimate the average value of an observable.

    The protocols returned will:

        1) Build a set of liquid coordinates for the
           property substance using packmol.

        2) Assign a set of smirnoff force field parameters
           to the system.

        3) Perform an energy minimisation on the system.

        4) Run a short NPT equilibration simulation for 100000 steps
           using a timestep of 2fs.

        5) Within a conditional group (up to a maximum of 100 times):

            5a) Run a longer NPT production simulation for 1000000 steps using a
                timestep of 2fs

            5b) Extract the average value of an observable and it's uncertainty.

            5c) If a convergence mode is set by the options, check if the target
                uncertainty has been met. If not, repeat steps 5a), 5b) and 5c).

        6) Extract uncorrelated configurations from a generated production
           simulation.

        7) Extract uncorrelated statistics from a generated production
           simulation.

    Parameters
    ----------
    analysis_protocol
        The protocol which will extract the observable of
        interest from the generated simulation data.
    use_target_uncertainty
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
        The protocols to add to the workflow, a reference to the average value of the
        estimated observable (an ``Observable`` object), and an object which describes
        the default data from a simulation to store, such as the uncorrelated statistics
        and configurations.
    """

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
    energy_minimisation.parameterized_system = ProtocolPath(
        "parameterized_system", assign_parameters.id
    )

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
    equilibration_simulation.parameterized_system = ProtocolPath(
        "parameterized_system", assign_parameters.id
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
    production_simulation.parameterized_system = ProtocolPath(
        "parameterized_system", assign_parameters.id
    )
    production_simulation.gradient_parameters = ProtocolPath(
        "parameter_gradient_keys", "global"
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

    # Point the analyse protocol to the correct data sources
    if not isinstance(analysis_protocol, analysis.BaseAverageObservable):
        raise ValueError(
            "The analysis protocol must inherit from either the "
            "AverageTrajectoryObservable or BaseAverageObservable "
            "protocols."
        )

    analysis_protocol.thermodynamic_state = ProtocolPath(
        "thermodynamic_state", "global"
    )
    analysis_protocol.potential_energies = ProtocolPath(
        f"observables[{ObservableType.PotentialEnergy.value}]",
        production_simulation.id,
    )

    # Finally, extract uncorrelated data
    time_series_statistics = ProtocolPath(
        "time_series_statistics", conditional_group.id, analysis_protocol.id
    )
    coordinate_file = ProtocolPath(
        "output_coordinate_file", conditional_group.id, production_simulation.id
    )
    trajectory_path = ProtocolPath(
        "trajectory_file_path", conditional_group.id, production_simulation.id
    )
    observables = ProtocolPath(
        "observables", conditional_group.id, production_simulation.id
    )

    decorrelate_trajectory = analysis.DecorrelateTrajectory(
        f"decorrelate_trajectory{id_suffix}"
    )
    decorrelate_trajectory.time_series_statistics = time_series_statistics
    decorrelate_trajectory.input_coordinate_file = coordinate_file
    decorrelate_trajectory.input_trajectory_path = trajectory_path

    decorrelate_observables = analysis.DecorrelateObservables(
        f"decorrelate_observables{id_suffix}"
    )
    decorrelate_observables.time_series_statistics = time_series_statistics
    decorrelate_observables.input_observables = observables

    # Build the object which defines which pieces of simulation data to store.
    output_to_store = StoredSimulationData()

    output_to_store.thermodynamic_state = ProtocolPath("thermodynamic_state", "global")
    output_to_store.property_phase = PropertyPhase.Liquid

    output_to_store.force_field_id = PlaceholderValue()

    output_to_store.number_of_molecules = ProtocolPath(
        "output_number_of_molecules", build_coordinates.id
    )
    output_to_store.substance = ProtocolPath("output_substance", build_coordinates.id)
    output_to_store.statistical_inefficiency = ProtocolPath(
        "time_series_statistics.statistical_inefficiency",
        conditional_group.id,
        analysis_protocol.id,
    )
    output_to_store.observables = ProtocolPath(
        "output_observables", decorrelate_observables.id
    )
    output_to_store.trajectory_file_name = ProtocolPath(
        "output_trajectory_path", decorrelate_trajectory.id
    )
    output_to_store.coordinate_file_name = coordinate_file

    output_to_store.source_calculation_id = PlaceholderValue()

    # Define where the final values come from.
    final_value_source = ProtocolPath(
        "value", conditional_group.id, analysis_protocol.id
    )

    base_protocols = SimulationProtocols(
        build_coordinates,
        assign_parameters,
        energy_minimisation,
        equilibration_simulation,
        production_simulation,
        analysis_protocol,
        conditional_group,
        decorrelate_trajectory,
        decorrelate_observables,
    )

    return base_protocols, final_value_source, output_to_store
