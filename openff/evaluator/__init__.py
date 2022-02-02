"""
openff-evaluator
A physical property evaluation toolkit from the Open Forcefield Consortium.
"""

from openff.units import unit

from ._version import get_versions
from .plugins import register_default_plugins, register_external_plugins

# Load the default plugins
register_default_plugins()
# Load in any found external plugins.
register_external_plugins()

# Handle versioneer
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
