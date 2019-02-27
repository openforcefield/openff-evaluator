"""
Properties base API.
"""

import uuid
from enum import IntFlag, unique
from typing import Optional, Dict, Any

from simtk import unit

from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils.serialization import TypedBaseModel


@unique
class PropertyPhase(IntFlag):
    """An enum describing the phase a property was collected in.
    """

    Undefined = 0x00
    Solid = 0x01
    Liquid = 0x02
    Gas = 0x04

    def __str__(self):
        """
        Returns
        ---
        str
            A string representation of the PropertyPhase enum
        """
        phases = '|'.join([phase.name for phase in PropertyPhase if self & phase])
        return phases

    def __repr__(self):
        """
        Returns
        ---
        str
            A string representation of the PropertyPhase enum
        """
        return str(self)


class Source(TypedBaseModel):
    """Container class for information about how a property was measured / calculated.

    .. todo:: Swap this out with a more general provenance class.
    """
    pass


class MeasurementSource(Source):
    """Contains any metadata about how a physical property was measured by experiment.

    This class contains either the DOI and/or the reference, but must contain at
    least one as the observable must have a source, even if it was measured in lab.

    Attributes
    ----------
    doi : str or None, default None
        The DOI for the source, preferred way to identify for source
    reference : str
        The long form description of the source if no DOI is available, or more
        information is needed or wanted.
    """

    doi: Optional[str] = None
    reference: Optional[str] = None


class CalculationSource(Source):
    """Contains any metadata about how a physical property was calculated.

    This includes at which fidelity the property was calculated at (e.g Direct
    simulation, reweighting, ...) in addition to the parameters which were
    used as part of the calculations.

    Attributes
    ----------
    fidelity : str
        The fidelity at which the property was calculated
    provenance : dict of str and Any
        A dictionary containing information about how the property was calculated.
    """

    fidelity: str = None
    provenance: Dict[str, Any] = None


class PhysicalProperty(TypedBaseModel):
    """Represents the value of any physical property and it's uncertainty.

    It additionally stores the thermodynamic state at which the property
    was collected, the phase it was collected in, information about
    the composition of the observed system, and metadata about how the
    property was collected.
    """

    thermodynamic_state: ThermodynamicState = None
    phase: PropertyPhase = PropertyPhase.Undefined

    substance: Any = None

    value: unit.Quantity = None
    uncertainty: unit.Quantity = None

    source: Any = None

    id: str = ''

    def __init__(self, **data):
        """
        Constructs a new PhysicalProperty object.
        """
        super().__init__(**data)
        self.id = str(uuid.uuid4())

    @property
    def temperature(self):
        """simtk.unit.Quantity or None: The temperature at which the property was collected."""
        return None if self.thermodynamic_state is None else self.thermodynamic_state.temperature

    @property
    def pressure(self):
        """simtk.unit.Quantity or None: The pressure at which the property was collected."""
        return None if self.thermodynamic_state is None else self.thermodynamic_state.pressure

    def set_value(self, value, uncertainty):
        """Set the value and uncertainty of this property.

        Parameters
        ----------
        value : simtk.unit.Quantity
            The value of the property.
        uncertainty : simtk.unit.Quantity
            The uncertainty in the properties value.
        """
        self.value = value
        self.uncertainty = uncertainty

    @staticmethod
    def get_default_workflow_schema():
        """Returns the set of steps needed to calculate
        this property by direct simulation methods.

        Returns
        -------
        propertyestimator.WorkflowSchema
            The calculation schema to follow.
        """
        raise NotImplementedError()
