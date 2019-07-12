"""
A collection of classes representing data stored by a storage backend.
"""


class BaseStoredData:
    """A base representation of cached data to be stored by
    a storage backend.
    """

    def __init__(self):
        """Constructs a new BaseStoredData object"""
        self.unique_id = None

        self.thermodynamic_state = None

        self.source_calculation_id = None
        self.provenance = None

        self.force_field_id = None

    def can_collapse(self, other_data):
        """Checks whether this piece of data stores the same
        amount of compatible information (or more) than another
        piece of stored data, and hence whether the two can be
        collapsed together.

        Parameters
        ----------
        other_data: BaseStoredData
            The other stored data to compare against.

        Returns
        -------
        bool
            Returns `True` if this piece of data stores the same
            amount of information or more than another piece of
            data, or false if it contains less or incompatible data.
        """

        if other_data is None:
            return False

        if type(self) != type(other_data):
            return False

        if self.thermodynamic_state != other_data.thermodynamic_state:
            return False

        if self.force_field_id != other_data.force_field_id:
            return False

        return True

    @classmethod
    def collapse(cls, stored_data_1, stored_data_2):
        """Collapse two pieces of compatible stored data
        into one.

        Parameters
        ----------
        stored_data_1: BaseStoredData
            The first piece of stored data.
        stored_data_2: BaseStoredData
            The second piece of stored data.

        Returns
        -------
        BaseStoredData
            The collapsed stored data.
        """
        raise NotImplementedError()

    def __getstate__(self):
        return {
            'unique_id': self.unique_id,

            'thermodynamic_state': self.thermodynamic_state,

            'source_calculation_id': self.source_calculation_id,
            'provenance': self.provenance,

            'force_field_id': self.force_field_id,
        }

    def __setstate__(self, state):
        self.unique_id = state['unique_id']

        self.thermodynamic_state = state['thermodynamic_state']

        self.source_calculation_id = state['source_calculation_id']
        self.provenance = state['provenance']

        self.force_field_id = state['force_field_id']


class StoredSimulationData(BaseStoredData):
    """A representation of data which has been cached
    from a single previous simulation.
    """

    def __init__(self):
        """Constructs a new StoredSimulationData object"""
        super().__init__()

        self.substance = None

        self.coordinate_file_name = None
        self.trajectory_file_name = None

        self.statistics_file_name = None
        self.statistical_inefficiency = 0.0

        self.total_number_of_molecules = None

    def can_collapse(self, other_data):
        """
        Parameters
        ----------
        other_data: StoredSimulationData
            The other stored data to compare against.
        """

        if not super(StoredSimulationData, self).can_collapse(other_data):
            return False

        if self.substance != other_data.substance:
            return False

        return True

    @classmethod
    def collapse(cls, stored_data_1, stored_data_2):
        """Collapse two pieces of compatible stored data
        into one, by only retaining the data with the longest
        autocorrelation time.

        Parameters
        ----------
        stored_data_1: StoredSimulationData
            The first piece of stored data.
        stored_data_2: StoredSimulationData
            The second piece of stored data.

        Returns
        -------
        StoredSimulationData
            The collapsed stored data.
        """

        # Make sure the two objects can actually be collapsed.
        if not stored_data_1.can_collapse(stored_data_2):

            raise ValueError('The two pieces of data are incompatible and cannot '
                             'be collapsed into one.')

        if stored_data_1.statistical_inefficiency < stored_data_2.statistical_inefficiency:
            return stored_data_2

        return stored_data_1

    def __getstate__(self):
        base_state = super(StoredSimulationData, self).__getstate__()

        base_state.update({

            'substance': self.substance,

            'coordinate_file_name': self.coordinate_file_name,
            'trajectory_file_name': self.trajectory_file_name,

            'statistics_file_name': self.statistics_file_name,
            'statistical_inefficiency': self.statistical_inefficiency,

            'total_number_of_molecules': self.total_number_of_molecules
        })

        return base_state

    def __setstate__(self, state):
        super(StoredSimulationData, self).__setstate__(state)

        self.substance = state['substance']

        self.coordinate_file_name = state['coordinate_file_name']
        self.trajectory_file_name = state['trajectory_file_name']

        self.statistics_file_name = state['statistics_file_name']
        self.statistical_inefficiency = state['statistical_inefficiency']

        self.total_number_of_molecules = state['total_number_of_molecules']


class StoredDataCollection(BaseStoredData):
    """A collection of stored `StoredSimulationData` objects, all
    generated at the same state and using the same force field
    parameters.
    """

    def __init__(self):
        """Constructs a new StoredDataCollection object"""
        super().__init__()
        self.data = {}

    def can_collapse(self, other_data):
        """
        Parameters
        ----------
        other_data: StoredDataCollection
            The other stored data to compare against.
        """

        if not super(StoredDataCollection, self).can_collapse(other_data):
            return False

        if len(self.data) != len(other_data.data):
            return False

        for data_key in self.data:

            if data_key not in other_data.data:
                return False

            self_data = self.data[data_key]
            other_data = other_data.data[data_key]

            if self_data.can_collapse(other_data):
                continue

            return False

        return True

    @classmethod
    def collapse(cls, stored_data_1, stored_data_2):
        """Collapse two pieces of compatible stored data
        into one, by only retaining the data with the longest
        autocorrelation time.

        Parameters
        ----------
        stored_data_1: StoredDataCollection
            The first piece of stored data.
        stored_data_2: StoredDataCollection
            The second piece of stored data.

        Returns
        -------
        StoredDataCollection
            The collapsed stored data.
        """

        # Make sure the two objects can actually be collapsed.
        if not stored_data_1.can_collapse(stored_data_2):

            raise ValueError('The two pieces of data are incompatible and '
                             'cannot be collapsed into one.')

        collapsed_data = cls()
        collapsed_data.force_field_id = stored_data_1.force_field_id

        for data_key in stored_data_1.data:

            collapsed_data.data[data_key] = stored_data_1.data[data_key].collapse(stored_data_1.data[data_key],
                                                                                  stored_data_2.data[data_key])

        return collapsed_data

    def __getstate__(self):

        state = super(StoredDataCollection, self).__getstate__()
        state.update({'data': self.data})
        return state

    def __setstate__(self, state):
        super(StoredDataCollection, self).__setstate__(state)
        self.data = state['data']
