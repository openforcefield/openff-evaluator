"""
Units tests for the openff.evaluator.plugins module.
"""
from openff.evaluator.layers import (
    registered_calculation_layers,
    registered_calculation_schemas,
)
from openff.evaluator.plugins import register_default_plugins, register_external_plugins
from openff.evaluator.workflow import registered_workflow_protocols


def test_register_default_plugins():
    register_default_plugins()

    assert len(registered_workflow_protocols) > 0
    assert len(registered_calculation_layers) > 0
    assert len(registered_calculation_schemas) > 0


def test_register_external_plugins(caplog):
    register_external_plugins()

    # Check that we could / couldn't load the correct plugins.
    assert "Could not load the Dummy1 plugin" not in caplog.text
    assert "Could not load the Dummy2 plugin" in caplog.text
