"""
openff-evaluator
A physical property evaluation toolkit from the Open Forcefield Consortium.
"""

from ._version import get_versions
from .plugins import register_default_plugins, register_external_plugins
from .utils.units import DEFAULT_UNIT_REGISTRY

unit = DEFAULT_UNIT_REGISTRY

# Load the default plugins
register_default_plugins()
# Load in any found external plugins.
register_external_plugins()

# Handle versioneer
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
