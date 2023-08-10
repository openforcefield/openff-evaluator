"""A set of helper classes for manipulating and passing inputs between
buildings blocks in a property estimation workflow.
"""

from openff.evaluator.attributes import PlaceholderValue
from openff.evaluator.utils import graph


class ReplicatorValue(PlaceholderValue):
    """A placeholder value which will be set by a protocol replicator
    with the specified id.
    """

    def __init__(self, replicator_id=""):
        """Constructs a new ReplicatorValue object

        Parameters
        ----------
        replicator_id: str
            The id of the replicator which will set this value.
        """
        self.replicator_id = replicator_id

    def __getstate__(self):
        return {"replicator_id": self.replicator_id}

    def __setstate__(self, state):
        self.replicator_id = state["replicator_id"]


class ProtocolPath(PlaceholderValue):
    """Represents a pointer to the output of another protocol."""

    # The character which separates protocol ids.
    path_separator = "/"
    # The character which separates the property name from the path.
    property_separator = "."

    @property
    def property_name(self):
        """str: The property name pointed to by the path."""
        return self._property_name

    @property
    def protocol_ids(self):
        """tuple of str: The ids of the protocols referenced by this
        object."""
        return self._protocol_ids

    @property
    def start_protocol(self):
        """str: The leading protocol id of the path."""
        return None if len(self._protocol_ids) == 0 else self._protocol_ids[0]

    @property
    def last_protocol(self):
        """str: The end protocol id of the path."""
        return None if len(self._protocol_ids) == 0 else self._protocol_ids[-1]

    @property
    def protocol_path(self):
        """str: The full path referenced by this object excluding the
        property name."""
        return self._protocol_path

    @property
    def full_path(self):
        """str: The full path referenced by this object."""
        return self._full_path

    @property
    def is_global(self):
        return self.start_protocol == "global"

    def __init__(self, property_name="", *protocol_ids):
        """Constructs a new ProtocolPath object.

        Parameters
        ----------
        property_name: str
            The property name referenced by the path.
        protocol_ids: str
            An args list of protocol ids in the order in which they will appear in the path.
        """

        if property_name is None:
            property_name = ""

        self._property_name = property_name
        self._protocol_ids = tuple(protocol_ids)

        self._protocol_path = None
        self._full_path = None

        self._update_string_paths()

    def _update_string_paths(self):
        """Combines the property name and protocol ids into string representations
        and stores them on the object.
        """

        self._protocol_path = ""

        if len(self._protocol_ids) > 0:
            self._protocol_path = ProtocolPath.path_separator.join(self._protocol_ids)

        property_name = "" if self._property_name is None else self._property_name

        self._full_path = (
            f"{self._protocol_path}{ProtocolPath.property_separator}{property_name}"
        )

    @classmethod
    def from_string(cls, existing_path_string: str):
        property_name, protocol_ids = ProtocolPath._to_components(existing_path_string)

        if any(x is None or len(x) == 0 for x in protocol_ids):
            raise ValueError("An invalid protocol id (either None or empty) was found.")

        return ProtocolPath(property_name, *protocol_ids)

    @staticmethod
    def _to_components(path_string):
        """Splits a protocol path string into the property
        name, and the individual protocol ids.

        Parameters
        ----------
        path_string: str
            The protocol path to split.

        Returns
        -------
        str, list of str
            A tuple of the property name, and a list of the protocol ids in the path.
        """
        path_string = path_string.lstrip().rstrip()
        property_name_index = path_string.find(ProtocolPath.property_separator)

        if property_name_index < 0:
            raise ValueError(
                f"A protocol path must contain a {ProtocolPath.property_separator} "
                f"followed by the property name this path represents"
            )

        property_name_index = path_string.find(ProtocolPath.property_separator)
        property_name = path_string[property_name_index + 1 :]

        protocol_id_path = path_string[:property_name_index]
        protocol_ids = protocol_id_path.split(ProtocolPath.path_separator)

        if len(protocol_id_path) == 0:
            protocol_ids = tuple()

        return property_name, protocol_ids

    def prepend_protocol_id(self, id_to_prepend):
        """Prepend a new protocol id onto the front of the path.

        Parameters
        ----------
        id_to_prepend: str
            The protocol id to prepend to the path
        """

        if len(self._protocol_ids) > 0 and self._protocol_ids[0] == id_to_prepend:
            return

        self._protocol_ids = (id_to_prepend, *self._protocol_ids)
        self._update_string_paths()

    def pop_next_in_path(self):
        """Pops and then returns the leading protocol id from the path.

        Returns
        -------
        str:
            The previously leading protocol id.
        """

        if len(self._protocol_ids) == 0:
            return None

        next_in_path = self._protocol_ids[0]

        self._protocol_ids = self._protocol_ids[1:]
        self._update_string_paths()

        return next_in_path

    def append_uuid(self, uuid):
        """Appends a uuid to each of the protocol id's in the path

        Parameters
        ----------
        uuid: str
            The uuid to append.
        """

        if self.is_global:
            # Don't append uuids to global paths.
            return

        self._protocol_ids = tuple(
            graph.append_uuid(x, uuid) for x in self._protocol_ids
        )
        self._update_string_paths()

    def replace_protocol(self, old_id, new_id):
        """Redirect the input to point at a new protocol.

        The main use of this method is when merging multiple protocols
        into one.

        Parameters
        ----------
        old_id : str
            The id of the protocol to replace.
        new_id : str
            The id of the new protocol to use.
        """
        self._protocol_ids = tuple(
            new_id if x == old_id else x for x in self._protocol_ids
        )
        self._update_string_paths()

    def copy(self):
        """Returns a copy of this path."""
        return ProtocolPath(self._property_name, *self._protocol_ids)

    def __str__(self):
        return self._full_path

    def __repr__(self):
        return f"<ProtocolPath full_path={self._full_path}>"

    def __hash__(self):
        """Returns the hash key of this ProtocolPath."""
        return hash(self._full_path)

    def __eq__(self, other):
        return type(self) is type(other) and self._full_path == other.full_path

    def __ne__(self, other):
        return not (self == other)

    def __getstate__(self):
        return {"full_path": self._full_path}

    def __setstate__(self, state):
        self._property_name, self._protocol_ids = ProtocolPath._to_components(
            state["full_path"]
        )

        self._update_string_paths()
