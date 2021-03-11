"""
A collection of general utilities.
"""
import contextlib
import logging
import os
import sys
from tempfile import TemporaryDirectory
from typing import Optional

from openff.evaluator.utils.string import extract_variable_index_and_name


def get_data_filename(relative_path):
    """Get the full path to one of the reference files in data.

    In the source distribution, these files are in ``evaluator/data/``,
    but on installation, they're moved to somewhere in the user's python
    site-packages directory.

    Parameters
    ----------
    relative_path : str
        The relative path of the file to load.
    """

    from pkg_resources import resource_filename

    fn = resource_filename("openff.evaluator", os.path.join("data", relative_path))

    if not os.path.exists(fn):
        raise ValueError(
            "Sorry! %s does not exist. If you just added it, you'll have to re-install"
            % fn
        )

    return fn


def timestamp_formatter() -> logging.Formatter:
    """Returns a logging formatter which outputs in the style of
    ``YEAR-MONTH-DAY HOUR:MINUTE:SECOND.MILLISECOND LEVEL MESSAGE``.
    """
    return logging.Formatter(
        fmt="%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def setup_timestamp_logging(file_path=None):
    """Set up timestamp based logging which outputs in the style of
    ``YEAR-MONTH-DAY HOUR:MINUTE:SECOND.MILLISECOND LEVEL MESSAGE``.

    Parameters
    ----------
    file_path: str, optional
        The file to write the log to. If none, the logger will
        print to the terminal.
    """
    formatter = timestamp_formatter()

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
                "This object does not have a {} attribute.".format(attribute_name)
            )

        current_attribute = getattr(current_attribute, attribute_name)

        if isinstance(current_attribute, list):

            try:
                array_index = int(array_index)
            except ValueError:
                raise ValueError(
                    "List indices must be integer: {}".format(attribute_name)
                )

        current_attribute[array_index] = value

    else:

        if not hasattr(current_attribute, attribute_name):

            raise ValueError(
                "This object does not have a {} attribute.".format(attribute_name)
            )

        setattr(current_attribute, attribute_name, value)


@contextlib.contextmanager
def temporarily_change_directory(directory_path: Optional[str] = None):
    """Temporarily move the current working directory to the path
    specified. If no path is given, a temporary directory will be
    created, moved into, and then destroyed when the context manager
    is closed.

    Parameters
    ----------
    directory_path
        The directory to change into. If None, a temporary directory will
        be created and changed into.
    """

    if directory_path is not None and len(directory_path) == 0:
        yield
        return

    old_directory = os.getcwd()

    try:

        if directory_path is None:

            with TemporaryDirectory() as new_directory:
                os.chdir(new_directory)
                yield

        else:

            os.chdir(directory_path)
            yield

    finally:
        os.chdir(old_directory)


def has_openeye():
    """Checks whether the `openeye` toolkits are available for use
    Returns
    -------
    bool
        True if the `openeye` toolkit can be imported and has a valid
        license.
    """

    try:

        from openeye import oechem

        available = True

        if not oechem.OEChemIsLicensed():
            available = False

    except ImportError:
        available = False

    return available


def is_file_and_not_empty(file_path):
    """Checks that a file both exists at the specified ``path`` and is not empty.

    Parameters
    ----------
    file_path: str
        The file path to check.

    Returns
    -------
    bool
        That a file both exists at the specified ``path`` and is not empty.
    """
    return os.path.isfile(file_path) and (os.path.getsize(file_path) != 0)


def is_number(s):
    """Returns True if string is a number."""
    try:
        float(s)
        return True
    except ValueError:
        return False
