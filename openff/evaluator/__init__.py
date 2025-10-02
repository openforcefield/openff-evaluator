"""
openff-evaluator
A physical property evaluation toolkit from the Open Forcefield Consortium.
"""

from importlib.metadata import version

from openff.units import unit

from openff.evaluator.plugins import register_default_plugins, register_external_plugins

# Load the default plugins
register_default_plugins()
# Load in any found external plugins.
register_external_plugins()

__version__ = version("openff.evaluator")

__all__ = ("unit",)
