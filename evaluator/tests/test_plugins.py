"""
Units tests for the evaluator.plugins module.
"""
import pkg_resources

from evaluator.layers import (
    registered_calculation_layers,
    registered_calculation_schemas,
)
from evaluator.plugins import register_default_plugins, register_external_plugins
from evaluator.workflow import registered_workflow_protocols


def test_register_default_plugins():

    register_default_plugins()

    assert len(registered_workflow_protocols) > 0
    assert len(registered_calculation_layers) > 0
    assert len(registered_calculation_schemas) > 0


def test_register_external_plugins(caplog):
    """This test is based on `this stack overflow answer
    <https://stackoverflow.com/a/48666503/11808960>`_
    """

    # Create a fake distribution to insert into the global working_set
    distribution = pkg_resources.Distribution(__file__)

    # Create the fake entry point definitions
    valid_entry_point = pkg_resources.EntryPoint.parse(
        "dummy_1 = evaluator.properties", dist=distribution
    )
    bad_entry_point = pkg_resources.EntryPoint.parse(
        "dummy_2 = evaluator.propertis", dist=distribution
    )

    # Add the mapping to the fake EntryPoint
    distribution._ep_map = {
        "openff-evaluator.plugins": {
            "dummy_1": valid_entry_point,
            "dummy_2": bad_entry_point,
        }
    }

    # Add the fake distribution to the global working_set
    pkg_resources.working_set.add(distribution, "dummy_1")
    pkg_resources.working_set.add(distribution, "dummy_2")

    register_external_plugins()

    # Check that we could / couldn't load the correct plugins.
    assert "Could not load the dummy_1" not in caplog.text
    assert "Could not load the dummy_2" in caplog.text
