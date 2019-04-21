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

        self.coordinate_path = None
        self.trajectory_path = None

        self.statistics_path = None
        self.statistical_inefficiency = 0.0

        self.force_field_id = None
