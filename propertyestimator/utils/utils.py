"""
A collection of general utilities.
"""
import abc
import copy
import logging
import os
import sys


def find_types_with_decorator(class_type, decorator_type):
    """ A method to collect all attributes marked by a specified
    decorator type (e.g. InputProperty).

    Parameters
    ----------
    class_type: class
        The class to pull attributes from.
    decorator_type: str
        The type of decorator to search for.

    Returns
    ----------
    The names of the attributes decorated with the specified decorator.
    """
    inputs = []

    def get_bases(current_base_type):

        bases = [current_base_type]

        for base_type in current_base_type.__bases__:
            bases.extend(get_bases(base_type))

        return bases

    all_bases = get_bases(class_type)

    for base in all_bases:

        inputs.extend([attribute_name for attribute_name in base.__dict__ if
                       type(base.__dict__[attribute_name]).__name__ == decorator_type])

    return inputs


def get_data_filename(relative_path):
    """Get the full path to one of the reference files in data.

    In the source distribution, these files are in ``propertyestimator/data/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    relative_path : str
        The relative path of the file to load.
    """

    from pkg_resources import resource_filename
    fn = resource_filename('propertyestimator', os.path.join('data', relative_path))

    if not os.path.exists(fn):
        raise ValueError("Sorry! %s does not exist. If you just added it, you'll have to re-install" % fn)

    return fn


_cached_molecules = {}


def create_molecule_from_smiles(smiles):
    """
    Create an ``OEMol`` molecule from a smiles pattern.

    .. todo:: Replace with the toolkit function when finished.

    Parameters
    ----------
    smiles : str
        Smiles pattern

    Returns
    -------
    molecule : OEMol
        OEMol with 3D coordinates, but no charges
     """

    from openeye import oechem, oeomega

    # Check cache
    if smiles in _cached_molecules:
        return copy.deepcopy(_cached_molecules[smiles])

    # Create molecule from smiles.
    molecule = oechem.OEMol()
    parse_smiles_options = oechem.OEParseSmilesOptions(quiet=True)

    if not oechem.OEParseSmiles(molecule, smiles, parse_smiles_options):

        logging.warning('Could not parse SMILES: ' + smiles)
        return False

    # Normalize molecule
    oechem.OEAssignAromaticFlags(molecule, oechem.OEAroModelOpenEye)
    oechem.OEAddExplicitHydrogens(molecule)
    oechem.OETriposAtomNames(molecule)

    # Create configuration
    omega = oeomega.OEOmega()

    omega.SetMaxConfs(1)
    omega.SetIncludeInput(False)
    omega.SetCanonOrder(False)
    omega.SetSampleHydrogens(True)
    omega.SetStrictStereo(True)
    omega.SetStrictAtomTypes(False)

    status = omega(molecule)

    if not status:

        logging.warning('Could not generate a conformer for ' + smiles)
        return False

    _cached_molecules[smiles] = molecule

    return molecule


def setup_timestamp_logging():
    """Set up timestamp-based logging."""
    formatter = logging.Formatter(fmt='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                                  datefmt='%H:%M:%S')

    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(screen_handler)


class SubhookedABCMeta(metaclass=abc.ABCMeta):
    """Abstract class with an implementation of __subclasshook__.
    The __subclasshook__ method checks that the instance implement the
    abstract properties and methods defined by the abstract class. This
    allow classes to implement an abstraction without explicitly
    subclassing it.

    Notes
    -----
    This class is an extension of the SubhookedABCMeta class from
    `openmmtools`

    Examples
    --------
    >>> class MyInterface(SubhookedABCMeta):
    ...     @abc.abstractmethod
    ...     def my_method(self): pass
    >>> class Implementation(object):
    ...     def my_method(self): return True
    >>> isinstance(Implementation(), MyInterface)
    True
    """

    # Populated by the metaclass, defined here to help IDEs only
    __abstractmethods__ = frozenset()

    @classmethod
    def __subclasshook__(cls, subclass):

        for abstract_method in cls.__abstractmethods__:

            if not any(abstract_method in C.__dict__ for C in subclass.__mro__):
                return False

        return True
