"""
Units tests for propertyestimator.workflow
"""
import tempfile
from os import path

import pytest

from propertyestimator import unit
from propertyestimator.backends import ComputeResources
from propertyestimator.properties.dielectric import ExtractAverageDielectric
from propertyestimator.protocols.analysis import (
    ExtractAverageStatistic,
    ExtractUncorrelatedStatisticsData,
    ExtractUncorrelatedTrajectoryData,
)
from propertyestimator.protocols.coordinates import BuildCoordinatesPackmol
from propertyestimator.protocols.forcefield import BuildSmirnoffSystem
from propertyestimator.protocols.miscellaneous import AddValues
from propertyestimator.protocols.simulation import (
    RunEnergyMinimisation,
    RunOpenMMSimulation,
)
from propertyestimator.substances import Substance
from propertyestimator.tests.test_workflow.utils import DummyInputOutputProtocol
from propertyestimator.tests.utils import build_tip3p_smirnoff_force_field
from propertyestimator.thermodynamics import Ensemble, ThermodynamicState
from propertyestimator.utils.exceptions import EvaluatorException
from propertyestimator.utils.quantities import EstimatedQuantity
from propertyestimator.utils.statistics import ObservableType
from propertyestimator.workflow.plugins import registered_workflow_protocols
from propertyestimator.workflow.protocols import ProtocolGraph
from propertyestimator.workflow.utils import ProtocolPath


@pytest.mark.parametrize("available_protocol", registered_workflow_protocols)
def test_default_protocol_schemas(available_protocol):
    """A simple test to ensure that each available protocol
    can both create, and be created from a schema."""
    protocol = registered_workflow_protocols[available_protocol]("dummy_id")
    protocol_schema = protocol.schema

    recreated_protocol = registered_workflow_protocols[available_protocol]("dummy_id")
    recreated_protocol.schema = protocol_schema

    assert protocol.schema.json() == recreated_protocol.schema.json()


def test_nested_protocol_paths():

    value_protocol_a = DummyInputOutputProtocol("protocol_a")
    value_protocol_a.input_value = EstimatedQuantity(
        1 * unit.kelvin, 0.1 * unit.kelvin, "constant"
    )

    assert (
        value_protocol_a.get_value(ProtocolPath("input_value.value"))
        == value_protocol_a.input_value.value
    )

    value_protocol_a.set_value(ProtocolPath("input_value._value"), 0.5 * unit.kelvin)
    assert value_protocol_a.input_value.value == 0.5 * unit.kelvin

    value_protocol_b = DummyInputOutputProtocol("protocol_b")
    value_protocol_b.input_value = EstimatedQuantity(
        2 * unit.kelvin, 0.05 * unit.kelvin, "constant"
    )

    value_protocol_c = DummyInputOutputProtocol("protocol_c")
    value_protocol_c.input_value = EstimatedQuantity(
        4 * unit.kelvin, 0.01 * unit.kelvin, "constant"
    )

    add_values_protocol = AddValues("add_values")

    add_values_protocol.values = [
        ProtocolPath("output_value", value_protocol_a.id),
        ProtocolPath("output_value", value_protocol_b.id),
        ProtocolPath("output_value", value_protocol_b.id),
        5,
    ]

    with pytest.raises(ValueError):
        add_values_protocol.get_value(ProtocolPath("valus[string]"))

    with pytest.raises(ValueError):
        add_values_protocol.get_value(ProtocolPath("values[string]"))

    input_values = add_values_protocol.get_value_references(ProtocolPath("values"))
    assert isinstance(input_values, dict) and len(input_values) == 3

    for index, value_reference in enumerate(input_values):

        input_value = add_values_protocol.get_value(value_reference)
        assert input_value.full_path == add_values_protocol.values[index].full_path

        add_values_protocol.set_value(value_reference, index)

    assert set(add_values_protocol.values) == {0, 1, 2, 5}

    dummy_dict_protocol = DummyInputOutputProtocol("dict_protocol")

    dummy_dict_protocol.input_value = {
        "value_a": ProtocolPath("output_value", value_protocol_a.id),
        "value_b": ProtocolPath("output_value", value_protocol_b.id),
    }

    input_values = dummy_dict_protocol.get_value_references(ProtocolPath("input_value"))
    assert isinstance(input_values, dict) and len(input_values) == 2

    for index, value_reference in enumerate(input_values):

        input_value = dummy_dict_protocol.get_value(value_reference)

        dummy_dict_keys = list(dummy_dict_protocol.input_value.keys())
        assert (
            input_value.full_path
            == dummy_dict_protocol.input_value[dummy_dict_keys[index]].full_path
        )

        dummy_dict_protocol.set_value(value_reference, index)

    add_values_protocol_2 = AddValues("add_values")

    add_values_protocol_2.values = [
        [ProtocolPath("output_value", value_protocol_a.id)],
        [
            ProtocolPath("output_value", value_protocol_b.id),
            ProtocolPath("output_value", value_protocol_b.id),
        ],
    ]

    with pytest.raises(ValueError):
        add_values_protocol_2.get_value(ProtocolPath("valus[string]"))

    with pytest.raises(ValueError):
        add_values_protocol.get_value(ProtocolPath("values[string]"))

    pass


def test_base_simulation_protocols():
    """Tests that the commonly chain build coordinates, assigned topology,
    energy minimise and perform simulation are able to work together without
    raising an exception."""

    water_substance = Substance.from_components("O")
    thermodynamic_state = ThermodynamicState(298 * unit.kelvin, 1 * unit.atmosphere)

    with tempfile.TemporaryDirectory() as temporary_directory:

        force_field_source = build_tip3p_smirnoff_force_field()
        force_field_path = path.join(temporary_directory, "ff.offxml")

        with open(force_field_path, "w") as file:
            file.write(force_field_source.json())

        build_coordinates = BuildCoordinatesPackmol("")

        # Set the maximum number of molecules in the system.
        build_coordinates.max_molecules = 10
        # and the target density (the default 1.0 g/ml is normally fine)
        build_coordinates.mass_density = 0.05 * unit.grams / unit.milliliters
        # and finally the system which coordinates should be generated for.
        build_coordinates.substance = water_substance

        # Build the coordinates, creating a file called output.pdb
        result = build_coordinates.execute(temporary_directory, None)
        assert not isinstance(result, EvaluatorException)

        # Assign some smirnoff force field parameters to the
        # coordinates
        print("Assigning some parameters.")
        assign_force_field_parameters = BuildSmirnoffSystem("")

        assign_force_field_parameters.force_field_path = force_field_path
        assign_force_field_parameters.coordinate_file_path = path.join(
            temporary_directory, "output.pdb"
        )
        assign_force_field_parameters.substance = water_substance

        result = assign_force_field_parameters.execute(temporary_directory, None)
        assert not isinstance(result, EvaluatorException)

        # Do a simple energy minimisation
        print("Performing energy minimisation.")
        energy_minimisation = RunEnergyMinimisation("")

        energy_minimisation.input_coordinate_file = path.join(
            temporary_directory, "output.pdb"
        )
        energy_minimisation.system_path = assign_force_field_parameters.system_path

        result = energy_minimisation.execute(temporary_directory, ComputeResources())
        assert not isinstance(result, EvaluatorException)

        npt_equilibration = RunOpenMMSimulation("npt_equilibration")

        npt_equilibration.ensemble = Ensemble.NPT

        npt_equilibration.steps_per_iteration = 20  # Debug settings.
        npt_equilibration.output_frequency = 2  # Debug settings.

        npt_equilibration.thermodynamic_state = thermodynamic_state

        npt_equilibration.input_coordinate_file = path.join(
            temporary_directory, "minimised.pdb"
        )
        npt_equilibration.system_path = assign_force_field_parameters.system_path

        result = npt_equilibration.execute(temporary_directory, ComputeResources())
        assert not isinstance(result, EvaluatorException)

        extract_density = ExtractAverageStatistic("extract_density")

        extract_density.statistics_type = ObservableType.Density
        extract_density.statistics_path = path.join(
            temporary_directory, "statistics.csv"
        )

        result = extract_density.execute(temporary_directory, ComputeResources())
        assert not isinstance(result, EvaluatorException)

        extract_dielectric = ExtractAverageDielectric("extract_dielectric")

        extract_dielectric.thermodynamic_state = thermodynamic_state

        extract_dielectric.input_coordinate_file = path.join(
            temporary_directory, "input.pdb"
        )
        extract_dielectric.trajectory_path = path.join(
            temporary_directory, "trajectory.dcd"
        )
        extract_dielectric.system_path = assign_force_field_parameters.system_path

        result = extract_dielectric.execute(temporary_directory, ComputeResources())
        assert not isinstance(result, EvaluatorException)

        extract_uncorrelated_trajectory = ExtractUncorrelatedTrajectoryData(
            "extract_traj"
        )

        extract_uncorrelated_trajectory.statistical_inefficiency = (
            extract_density.statistical_inefficiency
        )
        extract_uncorrelated_trajectory.equilibration_index = (
            extract_density.equilibration_index
        )
        extract_uncorrelated_trajectory.input_coordinate_file = path.join(
            temporary_directory, "input.pdb"
        )
        extract_uncorrelated_trajectory.input_trajectory_path = path.join(
            temporary_directory, "trajectory.dcd"
        )

        result = extract_uncorrelated_trajectory.execute(
            temporary_directory, ComputeResources()
        )
        assert not isinstance(result, EvaluatorException)

        extract_uncorrelated_statistics = ExtractUncorrelatedStatisticsData(
            "extract_stats"
        )

        extract_uncorrelated_statistics.statistical_inefficiency = (
            extract_density.statistical_inefficiency
        )
        extract_uncorrelated_statistics.equilibration_index = (
            extract_density.equilibration_index
        )
        extract_uncorrelated_statistics.input_statistics_path = path.join(
            temporary_directory, "statistics.csv"
        )

        result = extract_uncorrelated_statistics.execute(
            temporary_directory, ComputeResources()
        )
        assert not isinstance(result, EvaluatorException)


def build_merge(prefix):

    # a - b \
    #       | - e - f
    # c - d /
    protocol_a = DummyInputOutputProtocol(prefix + 'protocol_a')
    protocol_a.input_value = 1
    protocol_b = DummyInputOutputProtocol(prefix + 'protocol_b')
    protocol_b.input_value = ProtocolPath("output_value", protocol_a.id)
    protocol_c = DummyInputOutputProtocol(prefix + 'protocol_c')
    protocol_c.input_value = 1
    protocol_d = DummyInputOutputProtocol(prefix + 'protocol_d')
    protocol_d.input_value = ProtocolPath("output_value", protocol_c.id)
    protocol_e = DummyInputOutputProtocol(prefix + 'protocol_e')
    protocol_e.input_value = [
        ProtocolPath("output_value", protocol_b.id),
        ProtocolPath("output_value", protocol_d.id)
    ]
    protocol_f = DummyInputOutputProtocol(prefix + 'protocol_f')
    protocol_f.input_value = ProtocolPath("output_value", protocol_e.id)

    return [
        protocol_a,
        protocol_b,
        protocol_c,
        protocol_d,
        protocol_e,
        protocol_f,
    ]


def build_fork(prefix):
    #          / i - j
    # g - h - |
    #          \ k - l
    protocol_g = DummyInputOutputProtocol(prefix + 'protocol_g')
    protocol_g.input_value = 1
    protocol_h = DummyInputOutputProtocol(prefix + 'protocol_h')
    protocol_h.input_value = ProtocolPath("output_value", protocol_g.id)
    protocol_i = DummyInputOutputProtocol(prefix + 'protocol_i')
    protocol_i.input_value = ProtocolPath("output_value", protocol_h.id)
    protocol_j = DummyInputOutputProtocol(prefix + 'protocol_j')
    protocol_j.input_value = ProtocolPath("output_value", protocol_i.id)
    protocol_k = DummyInputOutputProtocol(prefix + 'protocol_k')
    protocol_k.input_value = ProtocolPath("output_value", protocol_h.id)
    protocol_l = DummyInputOutputProtocol(prefix + 'protocol_l')
    protocol_l.input_value = ProtocolPath("output_value", protocol_k.id)

    return [
        protocol_g,
        protocol_h,
        protocol_i,
        protocol_j,
        protocol_k,
        protocol_l
    ]


def build_easy_graph():

    protocol_a = DummyInputOutputProtocol('protocol_a')
    protocol_a.input_value = 1
    protocol_b = DummyInputOutputProtocol('protocol_b')
    protocol_b.input_value = 1

    return [protocol_a], [protocol_b]


def build_medium_graph():

    # a - b \
    #       | - e - f
    # c - d /
    #
    #          / i - j
    # g - h - |
    #          \ k - l
    return [*build_merge("_a"), *build_fork("_a")], [*build_merge("_b"), *build_fork("_b")]


def build_hard_graph():

    # a - b \                    / i - j
    #       | - e - f - g - h - |
    # c - d /                   \ k - l

    def build_graph(prefix):

        merger = build_merge(prefix)
        fork = build_fork(prefix)

        fork[0].input_value = ProtocolPath("output_value", prefix + "protocol_f")
        return [*merger, *fork]

    return build_graph("a_"), build_graph("b_")


@pytest.mark.parametrize("protocols_a, protocols_b", [
    build_easy_graph(),
    build_medium_graph(),
    build_hard_graph()
])
def test_protocol_graph_simple(protocols_a, protocols_b):

    # Make sure that the graph can merge simple protocols
    # when they are added one after the other.
    protocol_graph = ProtocolGraph()
    protocol_graph.add_protocols(*protocols_a)

    assert len(protocol_graph.protocols) == len(protocols_a)
    assert len(protocol_graph.dependants_graph) == len(protocols_a)
    n_root_protocols = len(protocol_graph.root_protocols)

    protocol_graph.add_protocols(*protocols_b)

    assert len(protocol_graph.protocols) == len(protocols_a)
    assert len(protocol_graph.dependants_graph) == len(protocols_a)
    assert len(protocol_graph.root_protocols) == n_root_protocols

    # Currently the graph shouldn't merge with an
    # addition
    protocol_graph = ProtocolGraph()
    protocol_graph.add_protocols(*protocols_a, *protocols_b)

    assert len(protocol_graph.protocols) == len(protocols_a) + len(protocols_b)
    assert len(protocol_graph.dependants_graph) == len(protocols_a) + len(protocols_b)
    assert len(protocol_graph.root_protocols) == 2 * n_root_protocols
