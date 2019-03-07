"""
A collection of classes representing data stored by a storage backend.
"""


class StoredSimulationData:
    """A container class for storing data from a previous simulation.
    """

    def __init__(self):

        self.unique_id = None

        self.substance = None
        self.thermodynamic_state = None

        self.source_calculation_id = None
        self.provenance = None

        self.statistical_inefficiency = 0.0

        self.trajectory_data = None
        self.statistics_data = None

        self.force_field_id = None
