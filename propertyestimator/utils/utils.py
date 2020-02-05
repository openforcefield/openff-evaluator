"""
A collection of general utilities.
"""
import contextlib
import logging
import os
import sys

from propertyestimator.utils.string import extract_variable_index_and_name


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
