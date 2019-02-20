"""
A collection of classes representing data stored by a storage backend.
"""

from mdtraj import Trajectory

from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState


class StoredSimulationData:
    """A container class for storing data from a previous simulation.
    """

    def __init__(self):

        self.substance: Substance = None
        self.thermodynamic_state: ThermodynamicState = None

        self.source_calculation_id: str = None
        self.provenance: str = None

        self.autocorrelation_time: float = 0.0
        self.effective_samples: int = 0

        self.trajectory_data: Trajectory = None

        self.force_field_id: str = None

    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        return v
