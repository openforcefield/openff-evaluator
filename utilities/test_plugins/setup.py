"""
test-evaluator-plugins
A test package used to ensure that parameterhandler plugins are handled correctly
"""

from setuptools import setup

setup(
    name="test_evaluator_plugins",
    packages=["evaluator_plugins"],
    include_package_data=True,
    entry_points={
        "openff_evaluator.plugins": [
            "DummyPlugin = evaluator_plugins.plugins:DummyPlugin",
        ]
    },
)
