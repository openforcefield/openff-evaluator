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
    from a previous simulation.
    """

    def __init__(self):
        """Constructs a new StoredSimulationData object"""
        super().__init__()

        self.substance = None

        self.coordinate_file_name = None
        self.trajectory_file_name = None

        self.statistics_file_name = None
        self.statistical_inefficiency = 0.0

    def __getstate__(self):

        base_state = super(StoredSimulationData, self).__getstate__()

        base_state.update({

            'substance': self.substance,

            'coordinate_file_name': self.coordinate_file_name,
            'trajectory_file_name': self.trajectory_file_name,

            'statistics_file_name': self.statistics_file_name,
            'statistical_inefficiency': self.statistical_inefficiency,
        })

        return base_state

    def __setstate__(self, state):

        super(StoredSimulationData, self).__setstate__(state)

        self.substance = state['substance']

        self.coordinate_file_name = state['coordinate_file_name']
        self.trajectory_file_name = state['trajectory_file_name']

        self.statistics_file_name = state['statistics_file_name']
        self.statistical_inefficiency = state['statistical_inefficiency']


class FreeEnergySimulationData(BaseStoredData):
    """A specialised storage class which represents data which has been
    cached from a previous free energy simulation.
    """

    def __init__(self):
        """Constructs a new FreeEnergySimulationData object"""

        super().__init__()

        self.substance_state_a = None
        self.substance_state_b = None

        self.coordinate_file_name_state_a = None
        self.coordinate_file_name_state_b = None

        self.trajectory_file_name_state_a = None
        self.trajectory_file_name_state_b = None

        self.statistics_file_name_state_a = None
        self.statistics_file_name_state_b = None

        self.statistical_inefficiency_state_a = 0.0
        self.statistical_inefficiency_state_b = 0.0

    def __getstate__(self):

        base_state = super(FreeEnergySimulationData, self).__getstate__()

        base_state.update({

            'substance_state_a': self.substance_state_a,
            'substance_state_b': self.substance_state_b,
    
            'coordinate_file_name_state_a': self.coordinate_file_name_state_a,
            'coordinate_file_name_state_b': self.coordinate_file_name_state_b,
    
            'trajectory_file_name_state_a': self.trajectory_file_name_state_a,
            'trajectory_file_name_state_b': self.trajectory_file_name_state_b,
    
            'statistics_file_name_state_a': self.statistics_file_name_state_a,
            'statistics_file_name_state_b': self.statistics_file_name_state_b,
    
            'statistical_inefficiency_state_a': self.statistical_inefficiency_state_a,
            'statistical_inefficiency_state_b': self.statistical_inefficiency_state_b
        })

        return base_state

    def __setstate__(self, state):

        super(FreeEnergySimulationData, self).__setstate__(state)

        self.substance_state_a = state['substance_state_a']
        self.substance_state_b = state['substance_state_b']

        self.coordinate_file_name_state_a = state['coordinate_file_name_state_a']
        self.coordinate_file_name_state_b = state['coordinate_file_name_state_b']

        self.trajectory_file_name_state_a = state['trajectory_file_name_state_a']
        self.trajectory_file_name_state_b = state['trajectory_file_name_state_b']

        self.statistics_file_name_state_a = state['statistics_file_name_state_a']
        self.statistics_file_name_state_b = state['statistics_file_name_state_b']

        self.statistical_inefficiency_state_a = state['statistical_inefficiency_state_a']
        self.statistical_inefficiency_state_b = state['statistical_inefficiency_state_b']
