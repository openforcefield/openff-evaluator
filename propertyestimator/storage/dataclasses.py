"""
A collection of classes representing data stored by a storage backend.
"""


class StoredSimulationData:
    """A class which describes a collection of data which has been cached
    from a previous simulation.
    """

    def __init__(self):
        """Constructs a new StoredSimulationData object"""
        self.unique_id = None

        self.substance = None
        self.thermodynamic_state = None

        self.source_calculation_id = None
        self.provenance = None

        self.coordinate_file_name = None
        self.trajectory_file_name = None

        self.statistics_file_name = None
        self.statistical_inefficiency = 0.0

        self.total_number_of_molecules = None

        self.force_field_id = None

    def __getstate__(self):

        return {
            'unique_id': self.unique_id,

            'substance': self.substance,
            'thermodynamic_state': self.thermodynamic_state,

            'source_calculation_id': self.source_calculation_id,
            'provenance': self.provenance,

            'coordinate_file_name': self.coordinate_file_name,
            'trajectory_file_name': self.trajectory_file_name,

            'statistics_file_name': self.statistics_file_name,
            'statistical_inefficiency': self.statistical_inefficiency,

            'total_number_of_molecules': self.total_number_of_molecules,

            'force_field_id': self.force_field_id,
        }

    def __setstate__(self, state):

        self.unique_id = state['unique_id']

        self.substance = state['substance']
        self.thermodynamic_state = state['thermodynamic_state']

        self.source_calculation_id = state['source_calculation_id']
        self.provenance = state['provenance']

        self.coordinate_file_name = state['coordinate_file_name']
        self.trajectory_file_name = state['trajectory_file_name']

        self.statistics_file_name = state['statistics_file_name']
        self.statistical_inefficiency = state['statistical_inefficiency']

        self.total_number_of_molecules = state['total_number_of_molecules']

        self.force_field_id = state['force_field_id']
