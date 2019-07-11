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
