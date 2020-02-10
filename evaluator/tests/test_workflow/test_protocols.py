"""
Units tests for evaluator.workflow
"""
import json
import tempfile

import pytest

from evaluator import unit
from evaluator.backends import ComputeResources
from evaluator.backends.dask import DaskLocalCluster
from evaluator.protocols.miscellaneous import AddValues
from evaluator.tests.test_workflow.utils import DummyInputOutputProtocol
from evaluator.utils.serialization import TypedJSONDecoder
from evaluator.workflow import workflow_protocol
from evaluator.workflow.protocols import Protocol, ProtocolGraph, ProtocolGroup
from evaluator.workflow.utils import ProtocolPath


@workflow_protocol()
class ExceptionProtocol(Protocol):
    def _execute(self, directory, available_resources):
        raise RuntimeError()


def test_nested_protocol_paths():

    value_protocol_a = DummyInputOutputProtocol("protocol_a")
    value_protocol_a.input_value = (1 * unit.kelvin).plus_minus(0.1 * unit.kelvin)

    assert (
        value_protocol_a.get_value(ProtocolPath("input_value.value"))
        == value_protocol_a.input_value.value
    )

    value_protocol_b = DummyInputOutputProtocol("protocol_b")
    value_protocol_b.input_value = (2 * unit.kelvin).plus_minus(0.05 * unit.kelvin)

    value_protocol_c = DummyInputOutputProtocol("protocol_c")
    value_protocol_c.input_value = (4 * unit.kelvin).plus_minus(0.01 * unit.kelvin)

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


def build_merge(prefix):

    # a - b \
    #       | - e - f
    # c - d /
    protocol_a = DummyInputOutputProtocol(prefix + "protocol_a")
    protocol_a.input_value = 1
    protocol_b = DummyInputOutputProtocol(prefix + "protocol_b")
    protocol_b.input_value = ProtocolPath("output_value", protocol_a.id)
    protocol_c = DummyInputOutputProtocol(prefix + "protocol_c")
    protocol_c.input_value = 1
    protocol_d = DummyInputOutputProtocol(prefix + "protocol_d")
    protocol_d.input_value = ProtocolPath("output_value", protocol_c.id)
    protocol_e = DummyInputOutputProtocol(prefix + "protocol_e")
    protocol_e.input_value = [
        ProtocolPath("output_value", protocol_b.id),
        ProtocolPath("output_value", protocol_d.id),
    ]
    protocol_f = DummyInputOutputProtocol(prefix + "protocol_f")
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
    protocol_g = DummyInputOutputProtocol(prefix + "protocol_g")
    protocol_g.input_value = 1
    protocol_h = DummyInputOutputProtocol(prefix + "protocol_h")
    protocol_h.input_value = ProtocolPath("output_value", protocol_g.id)
    protocol_i = DummyInputOutputProtocol(prefix + "protocol_i")
    protocol_i.input_value = ProtocolPath("output_value", protocol_h.id)
    protocol_j = DummyInputOutputProtocol(prefix + "protocol_j")
    protocol_j.input_value = ProtocolPath("output_value", protocol_i.id)
    protocol_k = DummyInputOutputProtocol(prefix + "protocol_k")
    protocol_k.input_value = ProtocolPath("output_value", protocol_h.id)
    protocol_l = DummyInputOutputProtocol(prefix + "protocol_l")
    protocol_l.input_value = ProtocolPath("output_value", protocol_k.id)

    return [protocol_g, protocol_h, protocol_i, protocol_j, protocol_k, protocol_l]


def build_easy_graph():

    protocol_a = DummyInputOutputProtocol("protocol_a")
    protocol_a.input_value = 1
    protocol_b = DummyInputOutputProtocol("protocol_b")
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
    return (
        [*build_merge("_a"), *build_fork("_a")],
        [*build_merge("_b"), *build_fork("_b")],
    )


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


@pytest.mark.parametrize(
    "protocols_a, protocols_b",
    [build_easy_graph(), build_medium_graph(), build_hard_graph()],
)
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


@pytest.mark.parametrize(
    "calculation_backend, compute_resources",
    [(DaskLocalCluster(), None), (None, ComputeResources())],
)
def test_protocol_graph_execution(calculation_backend, compute_resources):

    if calculation_backend is not None:
        calculation_backend.start()

    protocol_a = DummyInputOutputProtocol("protocol_a")
    protocol_a.input_value = 1
    protocol_b = DummyInputOutputProtocol("protocol_b")
    protocol_b.input_value = ProtocolPath("output_value", protocol_a.id)

    protocol_graph = ProtocolGraph()
    protocol_graph.add_protocols(protocol_a, protocol_b)

    with tempfile.TemporaryDirectory() as directory:

        results = protocol_graph.execute(
            directory, calculation_backend, compute_resources
        )

        final_result = results[protocol_b.id]

        if calculation_backend is not None:
            final_result = final_result.result()

        with open(final_result[1]) as file:
            results_b = json.load(file, cls=TypedJSONDecoder)

    assert results_b[".output_value"] == protocol_a.input_value

    if compute_resources is not None:
        assert protocol_b.output_value == protocol_a.input_value

    if calculation_backend is not None:
        calculation_backend.stop()


def test_protocol_group_merging():
    def build_protocols(prefix):

        #     .-------------------.
        #     |          / i - j -|- b
        # a - | g - h - |         |
        #     |          \ k - l -|- c
        #     .-------------------.
        protocol_a = DummyInputOutputProtocol(prefix + "protocol_a")
        protocol_a.input_value = 1
        fork_protocols = build_fork(prefix)
        fork_protocols[0].input_value = ProtocolPath("output_value", protocol_a.id)
        protocol_group = ProtocolGroup(prefix + "protocol_group")
        protocol_group.add_protocols(*fork_protocols)
        protocol_b = DummyInputOutputProtocol(prefix + "protocol_b")
        protocol_b.input_value = ProtocolPath(
            "output_value", protocol_group.id, "protocol_j"
        )
        protocol_c = DummyInputOutputProtocol(prefix + "protocol_c")
        protocol_c.input_value = ProtocolPath(
            "output_value", protocol_group.id, "protocol_l"
        )

        return [protocol_a, protocol_group, protocol_b, protocol_c]

    protocols_a = build_protocols("a_")
    protocols_b = build_protocols("b_")

    protocol_graph = ProtocolGraph()
    protocol_graph.add_protocols(*protocols_a)
    protocol_graph.add_protocols(*protocols_b)

    assert len(protocol_graph.protocols) == len(protocols_a)
    assert "a_protocol_group" in protocol_graph.protocols

    original_protocol_group = protocols_a[1]
    merged_protocol_group = protocol_graph.protocols["a_protocol_group"]

    assert original_protocol_group.schema.json() == merged_protocol_group.schema.json()


def test_protocol_group_execution():

    protocol_a = DummyInputOutputProtocol("protocol_a")
    protocol_a.input_value = 1
    protocol_b = DummyInputOutputProtocol("protocol_b")
    protocol_b.input_value = ProtocolPath("output_value", protocol_a.id)

    protocol_group = ProtocolGroup("protocol_group")
    protocol_group.add_protocols(protocol_a, protocol_b)

    with tempfile.TemporaryDirectory() as directory:

        protocol_group.execute(directory, ComputeResources())

    value_path = ProtocolPath("output_value", protocol_group.id, protocol_b.id)
    final_value = protocol_group.get_value(value_path)

    assert final_value == protocol_a.input_value


def test_protocol_group_exceptions():

    exception_protocol = ExceptionProtocol("exception_protocol")

    protocol_group = ProtocolGroup("protocol_group")
    protocol_group.add_protocols(exception_protocol)

    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(RuntimeError):
            protocol_group.execute(directory, ComputeResources())
