"""
Units tests for openff.evaluator.layers.simulation
"""
import math
import os
import tempfile

import numpy
import pytest
from openff.toolkit.typing.engines.smirnoff import ForceField, VirtualSiteHandler
from openff.units import unit

from openff.evaluator.attributes import UNDEFINED
from openff.evaluator.backends import ComputeResources
from openff.evaluator.backends.dask import DaskLocalCluster
from openff.evaluator.forcefield import ParameterGradientKey, SmirnoffForceFieldSource
from openff.evaluator.properties import Density
from openff.evaluator.protocols.groups import ConditionalGroup
from openff.evaluator.protocols.miscellaneous import DummyProtocol
from openff.evaluator.substances import Substance
from openff.evaluator.tests.utils import create_dummy_property
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.workflow import (
    ProtocolGroup,
    Workflow,
    WorkflowResult,
    WorkflowSchema,
)
from openff.evaluator.workflow.schemas import ProtocolReplicator
from openff.evaluator.workflow.utils import ProtocolPath, ReplicatorValue


@pytest.mark.parametrize(
    "calculation_backend, compute_resources, exception",
    [
        (None, None, False),
        (None, ComputeResources(number_of_threads=1), False),
        (DaskLocalCluster(), None, False),
        (DaskLocalCluster(), ComputeResources(number_of_threads=1), True),
    ],
)
def test_simple_workflow_graph(calculation_backend, compute_resources, exception):
    expected_value = (1 * unit.kelvin).plus_minus(0.1 * unit.kelvin)

    protocol_a = DummyProtocol("protocol_a")
    protocol_a.input_value = expected_value
    protocol_b = DummyProtocol("protocol_b")
    protocol_b.input_value = ProtocolPath("output_value", protocol_a.id)

    schema = WorkflowSchema()
    schema.protocol_schemas = [protocol_a.schema, protocol_b.schema]
    schema.final_value_source = ProtocolPath("output_value", protocol_b.id)
    schema.validate()

    workflow = Workflow({})
    workflow.schema = schema

    workflow_graph = workflow.to_graph()

    with tempfile.TemporaryDirectory() as directory:
        if calculation_backend is not None:
            with DaskLocalCluster() as calculation_backend:
                if exception:
                    with pytest.raises(AssertionError):
                        workflow_graph.execute(
                            directory, calculation_backend, compute_resources
                        )

                    return

                else:
                    results_futures = workflow_graph.execute(
                        directory, calculation_backend, compute_resources
                    )

                assert len(results_futures) == 1
                result = results_futures[0].result()

        else:
            result = workflow_graph.execute(
                directory, calculation_backend, compute_resources
            )[0]

            if exception:
                with pytest.raises(AssertionError):
                    workflow_graph.execute(
                        directory, calculation_backend, compute_resources
                    )

                return

        assert isinstance(result, WorkflowResult)
        assert result.value.value == expected_value.value


def test_workflow_with_groups():
    expected_value = (1 * unit.kelvin).plus_minus(0.1 * unit.kelvin)

    protocol_a = DummyProtocol("protocol_a")
    protocol_a.input_value = expected_value
    protocol_b = DummyProtocol("protocol_b")
    protocol_b.input_value = ProtocolPath("output_value", protocol_a.id)

    conditional_group = ConditionalGroup("conditional_group")
    conditional_group.add_protocols(protocol_a, protocol_b)

    condition = ConditionalGroup.Condition()
    condition.right_hand_value = 2 * unit.kelvin
    condition.type = ConditionalGroup.Condition.Type.LessThan
    condition.left_hand_value = ProtocolPath(
        "output_value.value", conditional_group.id, protocol_b.id
    )
    conditional_group.add_condition(condition)

    schema = WorkflowSchema()
    schema.protocol_schemas = [conditional_group.schema]
    schema.final_value_source = ProtocolPath(
        "output_value", conditional_group.id, protocol_b.id
    )
    schema.validate()

    workflow = Workflow({})
    workflow.schema = schema

    workflow_graph = workflow.to_graph()

    with tempfile.TemporaryDirectory() as directory:
        with DaskLocalCluster() as calculation_backend:
            results_futures = workflow_graph.execute(directory, calculation_backend)
            assert len(results_futures) == 1

            result = results_futures[0].result()

        assert isinstance(result, WorkflowResult)
        assert result.value.value == expected_value.value


def test_nested_input():
    dict_protocol = DummyProtocol("dict_protocol")
    dict_protocol.input_value = {"a": ThermodynamicState(1.0 * unit.kelvin)}

    quantity_protocol = DummyProtocol("quantity_protocol")
    quantity_protocol.input_value = ProtocolPath(
        "output_value[a].temperature", dict_protocol.id
    )

    schema = WorkflowSchema()
    schema.protocol_schemas = [dict_protocol.schema, quantity_protocol.schema]
    schema.validate()

    workflow = Workflow({})
    workflow.schema = schema

    workflow_graph = workflow.to_graph()

    with tempfile.TemporaryDirectory() as temporary_directory:
        with DaskLocalCluster() as calculation_backend:
            results_futures = workflow_graph.execute(
                temporary_directory, calculation_backend
            )

            assert len(results_futures) == 1
            result = results_futures[0].result()

    assert isinstance(result, WorkflowResult)


def test_index_replicated_protocol():
    replicator = ProtocolReplicator("replicator")
    replicator.template_values = ["a", "b", "c", "d"]

    replicated_protocol = DummyProtocol(f"protocol_{replicator.placeholder_id}")
    replicated_protocol.input_value = ReplicatorValue(replicator.id)

    schema = WorkflowSchema()
    schema.protocol_replicators = [replicator]
    schema.protocol_schemas = [replicated_protocol.schema]

    for index in range(len(replicator.template_values)):
        indexing_protocol = DummyProtocol(f"indexing_protocol_{index}")
        indexing_protocol.input_value = ProtocolPath(
            "output_value", f"protocol_{index}"
        )
        schema.protocol_schemas.append(indexing_protocol.schema)

    schema.validate()

    workflow = Workflow({})
    workflow.schema = schema


def test_from_schema():
    protocol_a = DummyProtocol("protocol_a")
    protocol_a.input_value = 1 * unit.kelvin

    schema = WorkflowSchema()
    schema.protocol_schemas = [protocol_a.schema]

    workflow = Workflow.from_schema(schema, {}, unique_id="")

    assert workflow is not None

    rebuilt_schema = workflow.schema
    rebuilt_schema.outputs_to_store = UNDEFINED

    assert rebuilt_schema.json(format=True) == schema.json(format=True)


def test_unique_ids():
    protocol_a = DummyProtocol("protocol-a")
    protocol_a.input_value = 1

    group_a = ProtocolGroup("group-a")
    group_a.add_protocols(protocol_a)

    group_b = ProtocolGroup("group-b")
    group_b.add_protocols(protocol_a)

    schema = WorkflowSchema()
    schema.protocol_schemas = [group_a.schema, group_b.schema]

    with pytest.raises(ValueError) as error_info:
        schema.validate()

    assert "Several protocols in the schema have the same id" in str(error_info.value)
    assert "protocol-a" in str(error_info.value)


def test_replicated_ids():
    replicator = ProtocolReplicator("replicator-a")

    protocol_a = DummyProtocol("protocol-a")
    protocol_a.input_value = 1

    group_a = ProtocolGroup(f"group-a-{replicator.placeholder_id}")
    group_a.add_protocols(protocol_a)

    schema = WorkflowSchema()
    schema.protocol_schemas = [group_a.schema]
    schema.protocol_replicators = [replicator]

    with pytest.raises(ValueError) as error_info:
        schema.validate()

    assert (
        f"The children of replicated protocol {group_a.id} must also contain the "
        "replicators placeholder" in str(error_info.value)
    )


def test_find_relevant_gradient_keys(tmpdir):
    force_field = ForceField()

    vdw_handler = force_field.get_parameter_handler("vdW")
    vdw_handler.add_parameter(
        {
            "smirks": "[#1:1]",
            "epsilon": 0.0 * unit.kilocalorie_per_mole,
            "sigma": 1.0 * unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#17:1]",
            "epsilon": 0.0 * unit.kilocalorie_per_mole,
            "sigma": 1.0 * unit.angstrom,
        }
    )
    vdw_handler.add_parameter(
        {
            "smirks": "[#6:1]",
            "epsilon": 0.0 * unit.kilocalorie_per_mole,
            "sigma": 1.0 * unit.angstrom,
        }
    )
    vsite_handler = VirtualSiteHandler(version=0.3)
    vsite_handler.add_parameter(
        {
            "smirks": "[#1:1][#17:2]",
            "type": "BondCharge",
            "distance": 0.1 * unit.nanometers,
            "match": "all_permutations",
            "charge_increment1": 0.0 * unit.elementary_charge,
            "charge_increment2": 0.0 * unit.elementary_charge,
        }
    )
    force_field.register_parameter_handler(vsite_handler)

    force_field_path = os.path.join(tmpdir, "ff.json")
    SmirnoffForceFieldSource.from_object(force_field).json(force_field_path)

    expected_gradient_keys = {
        ParameterGradientKey(tag="vdW", smirks=None, attribute="scale14"),
        ParameterGradientKey(tag="vdW", smirks="[#1:1]", attribute="epsilon"),
        ParameterGradientKey(
            tag="VirtualSites", smirks="[#1:1][#17:2]", attribute="distance"
        ),
    }

    gradient_keys = Workflow._find_relevant_gradient_keys(
        Substance.from_components("[H]Cl"),
        force_field_path,
        [
            *expected_gradient_keys,
            ParameterGradientKey(tag="vdW", smirks="[#6:1]", attribute="sigma"),
        ],
    )

    assert len(gradient_keys) == len(expected_gradient_keys)
    assert {*gradient_keys} == expected_gradient_keys


def test_generate_default_metadata_defaults():
    dummy_property = create_dummy_property(Density)
    dummy_forcefield = "smirnoff99Frosst-1.1.0.offxml"

    data = Workflow.generate_default_metadata(dummy_property, dummy_forcefield)

    assert data["parameter_gradient_keys"] == []
    assert numpy.isclose(
        data["target_uncertainty"], math.inf * unit.gram / unit.milliliter
    )
    assert numpy.isclose(
        data["per_component_uncertainty"], math.inf * unit.gram / unit.milliliter
    )
