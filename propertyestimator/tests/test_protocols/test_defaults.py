import importlib
import pytest

from propertyestimator.workflow.plugins import registered_workflow_protocols


# Load the default protocols.
protocol_modules = [
    "analysis",
    "coordinates",
    "forcefield",
    "gradients",
    "groups",
    "miscellaneous",
    "reweighting",
    "simulation",
    "storage",
    "yank",
]

for protocol_module in protocol_modules:
    importlib.import_module(f"propertyestimator.protocols.{protocol_module}")


@pytest.mark.parametrize("available_protocol", registered_workflow_protocols)
def test_default_protocol_schemas(available_protocol):
    """A simple test to ensure that each available protocol
    can both create, and be created from a schema."""
    protocol = registered_workflow_protocols[available_protocol]("dummy_id")
    protocol_schema = protocol.schema

    recreated_protocol = registered_workflow_protocols[available_protocol]("dummy_id")
    recreated_protocol.schema = protocol_schema

    assert protocol.schema.json() == recreated_protocol.schema.json()
