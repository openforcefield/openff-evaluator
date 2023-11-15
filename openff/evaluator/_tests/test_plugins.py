"""
Units tests for the openff.evaluator.plugins module.
"""
import sys

from openff.evaluator.layers import (
    registered_calculation_layers,
    registered_calculation_schemas,
)
from openff.evaluator.plugins import register_default_plugins, register_external_plugins
from openff.evaluator.workflow import registered_workflow_protocols

if sys.version_info[1] < 10:
    # Backport only for Python 3.9 - drop April 2024
    from importlib_metadata import entry_points
else:
    from importlib.metadata import entry_points


def test_register_default_plugins():
    register_default_plugins()

    assert len(registered_workflow_protocols) > 0
    assert len(registered_calculation_layers) > 0
    assert len(registered_calculation_schemas) > 0


def test_register_external_plugins(caplog):
    """This test wass based on `this stack overflow answer
    <https://stackoverflow.com/a/48666503/11808960>`_, but it relied on
    functionality not present in `importlib.metadata`.
    """
    register_external_plugins()
    names = {
        entry_point.name
        for entry_point in entry_points().select(group="openff_evaluator.plugins")
    }

    assert names == {"DummyPlugin"}
