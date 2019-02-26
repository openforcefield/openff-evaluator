"""A set of helper classes for manipulating and passing inputs between
buildings blocks in a property estimation workflow.
"""

from propertyestimator.utils import graph


class PlaceholderInput:
    """A class to act as a place holder for a protocols
    input value, for when the value of an input is not
    known a priori, and does not come from another protocol.
    """

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
        pass


class ReplicatorValue(PlaceholderInput):
    """A placeholder value which will be set by a protocol replicator
    with the specified id.
    """

    def __init__(self, non_default, replicator_id=''):
        """Constructs a new ReplicatorValue object

        Parameters
        ----------
        replicator_id: str
            The id of the replicator which will set this value.
        """
        self.replicator_id = replicator_id

    def __getstate__(self):
        return {
            'replicator_id': self.replicator_id
        }

    def __setstate__(self, state):
        self.replicator_id = state['replicator_id']


class ProtocolPath(PlaceholderInput):
    """Represents a pointer to the output of another protocol.
    """

    path_separator = '/'  # The character which separates protocol ids.
    property_separator = '.'  # The character which separates the property name from the path.

    @property
    def property_name(self):
        """str: The property name pointed to by the path."""
        property_name, protocol_ids = ProtocolPath.to_components(self._full_path)
        return property_name

    @property
    def start_protocol(self):
        """str: The leading protocol id of the path."""
        property_name, protocol_ids = ProtocolPath.to_components(self._full_path)
        return None if len(protocol_ids) == 0 else protocol_ids[0]

    @property
    def last_protocol(self):
        """str: The leading protocol id of the path."""
        property_name, protocol_ids = ProtocolPath.to_components(self._full_path)
        return None if len(protocol_ids) == 0 else protocol_ids[len(protocol_ids) - 1]

    @property
    def full_path(self):
        """str: The full path referenced by this object."""
        return self._full_path

    @property
    def is_global(self):
        return self.start_protocol == 'global'

    def __init__(self, property_name, *protocol_ids):
        """Constructs a new ProtocolPath object.

        Parameters
        ----------
        property_name: str
            The property name referenced by the path.
        protocol_ids: str
            An args list of protocol ids in the order in which they will appear in the path.
        """

        self._full_path = ''
        self._from_components(property_name, *protocol_ids)

    def _from_components(self, property_name, *protocol_ids):
        """Sets this components path from individual components.

        Parameters
        ----------
        property_name: str
            The property name referenced by the path.
        protocol_ids: str
            A list of protocol ids in the order in which they will appear in the path.
        """

        assert property_name is not None and isinstance(property_name, str)

        assert property_name.find(ProtocolPath.path_separator) < 0

        for protocol_id in protocol_ids:

            assert protocol_id is not None and isinstance(protocol_id, str)

            assert protocol_id.find(ProtocolPath.property_separator) < 0 and \
                   protocol_id.find(ProtocolPath.path_separator) < 0

        protocol_path = ProtocolPath.path_separator.join(protocol_ids)

        if len(protocol_ids) == 0:
            protocol_path = ''

        self._full_path = '{}{}{}'.format(protocol_path,
                                          ProtocolPath.property_separator,
                                          property_name)

    @classmethod
    def from_string(cls, existing_path_string: str):

        existing_path_string = existing_path_string.lstrip().rstrip()
        property_name_index = existing_path_string.find(ProtocolPath.property_separator)

        if property_name_index < 0:

            raise ValueError('A protocol path must contain a {} followed by the '
                             'property name this path represents'.format(ProtocolPath.property_separator))

        property_name, protocol_ids = ProtocolPath.to_components(existing_path_string)

        for protocol_id in protocol_ids:

            if protocol_id is not None and len(protocol_id) > 0:
                continue

            raise ValueError('An invalid protocol id (either None or empty) was found.')

        return ProtocolPath(property_name, *protocol_ids)

    @staticmethod
    def to_components(path_string):
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
        property_name_index = path_string.find(ProtocolPath.property_separator)
        property_name = path_string[property_name_index + 1:]

        protocol_id_path = path_string[:property_name_index]
        protocol_ids = protocol_id_path.split(ProtocolPath.path_separator)

        if len(protocol_id_path) == 0:
            protocol_ids = []

        return property_name, protocol_ids

    def prepend_protocol_id(self, id_to_prepend):
        """Prepend a new protocol id onto the front of the path.

        Parameters
        ----------
        id_to_prepend: str
            The protocol id to prepend to the path
        """
        property_name, protocol_ids = ProtocolPath.to_components(self._full_path)

        if len(protocol_ids) == 0 or (len(protocol_ids) > 0 and protocol_ids[0] != id_to_prepend):
            protocol_ids.insert(0, id_to_prepend)

        self._from_components(property_name, *protocol_ids)

    def pop_next_in_path(self):
        """Pops and then returns the leading protocol id from the path.

        Returns
        -------
        str:
            The previously leading protocol id.
        """
        property_name, protocol_ids = ProtocolPath.to_components(self._full_path)

        if len(protocol_ids) == 0:
            return None

        next_in_path = protocol_ids.pop(0)
        self._from_components(property_name, *protocol_ids)

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

        property_name, protocol_ids = ProtocolPath.to_components(self._full_path)
        appended_ids = []

        for protocol_id in protocol_ids:

            if protocol_id is None:
                continue

            appended_id = graph.append_uuid(protocol_id, uuid)
            appended_ids.append(appended_id)

        self._from_components(property_name, *appended_ids)

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
        self._full_path = self._full_path.replace(old_id, new_id)

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):

        if isinstance(v, str):
            return ProtocolPath.from_string(v)
        elif isinstance(v, dict):

            path_object = ProtocolPath('', *[])
            path_object.__setstate__(v)

            v = path_object

        return v

    def __str__(self):
        return self._full_path

    def __repr__(self):
        return '<ProtocolPath full_path={}>'.format(self._full_path)

    def __hash__(self):
        """Returns the hash key of this ProtocolPath."""
        return hash(self._full_path)

    def __eq__(self, other):
        """Returns true if the two inputs are equal."""
        return self._full_path == other.full_path

    def __ne__(self, other):
        """Returns true if the two inputs are not equal."""
        return not (self == other)

    def __getstate__(self):
        return {'full_path': self._full_path}

    def __setstate__(self, state):
        self._full_path = state['full_path']
