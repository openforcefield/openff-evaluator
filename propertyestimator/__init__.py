"""
propertyestimator
Property calculation toolkit from the Open Forcefield Consortium.
"""

import pint
import pkg_resources
import warnings

from ._version import get_versions

unit = pint.UnitRegistry()
pint.set_application_registry(unit)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pint.Quantity([])

# Load in any found plugins.
for entry_point in pkg_resources.iter_entry_points("propertyestimator.plugins"):

    try:
        entry_point.load()
    except ImportError as e:

        import logging
        import traceback

        formatted_exception = traceback.format_exception(None, e, e.__traceback__)
        logging.warning(
            f"Could not load the {entry_point} plugin: {formatted_exception}"
        )

# Handle versioneer
versions = get_versions()
__version__ = versions["version"]
__git_revision__ = versions["full-revisionid"]
del get_versions, versions
