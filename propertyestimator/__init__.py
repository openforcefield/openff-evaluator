"""
propertyestimator
Property calculation toolkit from the Open Forcefield Consortium.
"""

# Make Python 2 and 3 imports work the same
# Safe to remove with Python 3-only code
from __future__ import absolute_import

import pkg_resources

from ._version import get_versions

# Load in any found plugins.
for entry_point in pkg_resources.iter_entry_points('propertyestimator.plugins'):
    entry_point.load()

# Handle versioneer
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
