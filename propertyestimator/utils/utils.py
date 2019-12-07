"""
A collection of general utilities.
"""
import abc
import contextlib
import copy
import logging
import os
import sys

from propertyestimator.utils.string import extract_variable_index_and_name


def find_types_with_decorator(class_type, decorator_type):
    """ A method to collect all attributes marked by a specified
    decorator type (e.g. InputProperty).

    Parameters
    ----------
    class_type: type
        The class to pull attributes from.
    decorator_type: type
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

        inputs.extend(
            [
                attribute_name
                for attribute_name in base.__dict__
                if isinstance(base.__dict__[attribute_name], decorator_type)
            ]
        )

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

    fn = resource_filename("propertyestimator", os.path.join("data", relative_path))

    if not os.path.exists(fn):
        raise ValueError(
            "Sorry! %s does not exist. If you just added it, you'll have to re-install"
            % fn
        )

    return fn


_cached_molecules = {}


def create_molecule_from_smiles(smiles, number_of_conformers=1):
    """
    Create an ``OEMol`` molecule from a smiles pattern.

    .. todo:: Replace with the toolkit function when finished.

    Parameters
    ----------
    smiles : str
        The smiles pattern to create the molecule from.
    number_of_conformers: int
        The number of conformers to generate for the molecule using Omega.

    Returns
    -------
    molecule : OEMol
        OEMol with no charges, and a number of conformers as specified
        by `number_of_conformers`
     """

    from openeye import oechem, oeomega

    # Check cache
    if (number_of_conformers, smiles) in _cached_molecules:
        return copy.deepcopy(_cached_molecules[(number_of_conformers, smiles)])

    # Create molecule from smiles.
    molecule = oechem.OEMol()
    parse_smiles_options = oechem.OEParseSmilesOptions(quiet=True)

    if not oechem.OEParseSmiles(molecule, smiles, parse_smiles_options):

        logging.warning("Could not parse SMILES: " + smiles)
        return None

    # Normalize molecule
    oechem.OEAssignAromaticFlags(molecule, oechem.OEAroModelOpenEye)
    oechem.OEAddExplicitHydrogens(molecule)

    # Create configuration
    if number_of_conformers > 0:

        omega = oeomega.OEOmega()

        omega.SetMaxConfs(number_of_conformers)
        omega.SetIncludeInput(False)
        omega.SetCanonOrder(False)
        omega.SetSampleHydrogens(True)
        omega.SetStrictStereo(True)
        omega.SetStrictAtomTypes(False)

        status = omega(molecule)

        if not status:

            logging.warning("Could not generate a conformer for " + smiles)
            return None

    _cached_molecules[(number_of_conformers, smiles)] = molecule

    return molecule


def setup_timestamp_logging(file_path=None):
    """Set up timestamp-based logging.

    Parameters
    ----------
    file_path: str, optional
        The file to write the log to. If none, the logger will
        print to the terminal.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s", datefmt="%H:%M:%S"
    )

    if file_path is None:
        logger_handler = logging.StreamHandler(stream=sys.stdout)
    else:
        logger_handler = logging.FileHandler(file_path)

    logger_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logger_handler)


def safe_unlink(file_path):
    """Attempts to remove the file at the given path,
    catching any file not found exceptions.

    Parameters
    ----------
    file_path: str
        The path to the file to remove.
    """
    try:
        os.unlink(file_path)
    except OSError:
        pass


def get_nested_attribute(containing_object, name):
    """A recursive version of getattr, which has full support
    for attribute names which contain list / dict indices

    Parameters
    ----------
    containing_object: Any
        The object which contains the nested attribute.
    name: str
        The name (/ path) of the nested attribute with successive
        attribute names separated by periods, for example:

            name = 'attribute_a.attribute_b[index].attribute_c'

    Returns
    -------
    Any
        The value of the attribute.
    """
    attribute_name_split = name.split(".")

    current_attribute = containing_object

    for index, full_attribute_name in enumerate(attribute_name_split):

        array_index = None
        attribute_name = full_attribute_name

        if attribute_name.find("[") >= 0 or attribute_name.find("]") >= 0:
            attribute_name, array_index = extract_variable_index_and_name(
                attribute_name
            )

        if current_attribute is None:
            return None

        if not isinstance(current_attribute, dict):

            if not hasattr(current_attribute, attribute_name):

                raise ValueError(
                    "This object does not have a {} "
                    "attribute.".format(".".join(attribute_name_split[: index + 1]))
                )

            else:

                current_attribute = getattr(current_attribute, attribute_name)

        else:

            if attribute_name not in current_attribute:

                raise ValueError(
                    "This object does not have a {} "
                    "attribute.".format(".".join(attribute_name_split[: index + 1]))
                )

            else:
                current_attribute = current_attribute[attribute_name]

        if array_index is not None:

            if isinstance(current_attribute, list):

                try:
                    array_index = int(array_index)
                except ValueError:

                    raise ValueError(
                        "List indices must be integer: "
                        "{}".format(".".join(attribute_name_split[: index + 1]))
                    )

            array_value = current_attribute[array_index]
            current_attribute = array_value

    return current_attribute


def set_nested_attribute(containing_object, name, value):
    """A recursive version of setattr, which has full support
    for attribute names which contain list / dict indices

    Parameters
    ----------
    containing_object: Any
        The object which contains the nested attribute.
    name: str
        The name (/ path) of the nested attribute with successive
        attribute names separated by periods, for example:

            name = 'attribute_a.attribute_b[index].attribute_c'
    value: Any
        The value to set on the attribute.
    """

    current_attribute = containing_object
    attribute_name = name

    if attribute_name.find(".") > 1:

        last_separator_index = attribute_name.rfind(".")

        current_attribute = get_nested_attribute(
            current_attribute, attribute_name[:last_separator_index]
        )
        attribute_name = attribute_name[last_separator_index + 1 :]

    if attribute_name.find("[") >= 0:

        attribute_name, array_index = extract_variable_index_and_name(attribute_name)

        if not hasattr(current_attribute, attribute_name):

            raise ValueError(
                "This object does not have a {} " "attribute.".format(attribute_name)
            )

        current_attribute = getattr(current_attribute, attribute_name)

        if isinstance(current_attribute, list):

            try:
                array_index = int(array_index)
            except ValueError:
                raise ValueError(
                    "List indices must be integer: " "{}".format(attribute_name)
                )

        current_attribute[array_index] = value

    else:

        if not hasattr(current_attribute, attribute_name):

            raise ValueError(
                "This object does not have a {} " "attribute.".format(attribute_name)
            )

        setattr(current_attribute, attribute_name, value)


@contextlib.contextmanager
def temporarily_change_directory(file_path):
    """A context to temporarily change the working directory.

    Parameters
    ----------
    file_path: str
        The file path to temporarily change into.
    """
    prev_dir = os.getcwd()
    os.chdir(os.path.abspath(file_path))

    try:
        yield
    finally:
        os.chdir(prev_dir)


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
