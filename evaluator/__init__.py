"""
evaluator
A physical property evaulation toolkit from the Open Forcefield Consortium.
"""

import warnings

import pint

from ._version import get_versions
from .plugins import register_default_plugins, register_external_plugins

unit = pint.UnitRegistry()
unit.default_format = "~"
pint.set_application_registry(unit)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])

# Load the default plugins
register_default_plugins()
# Load in any found external plugins.
register_external_plugins()

# Handle versioneer
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
