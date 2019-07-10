"""
Properties base API.
"""

import uuid
from enum import IntFlag, unique

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

    def __getstate__(self): return {}

    def __setstate__(self, state): pass


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

    def __init__(self, doi='', reference=''):
        """Constructs a new MeasurementSource object.

        Parameters
        ----------
        doi : str or None, default None
            The DOI for the source, preferred way to identify for source
        reference : str
            The long form description of the source if no DOI is available, or more
            information is needed or wanted.
        """

        self.doi = doi
        self.reference = reference

    def __getstate__(self):

        return {
            'doi': self.doi,
            'reference': self.reference,
        }

    def __setstate__(self, state):

        self.doi = state['doi']
        self.reference = state['reference']


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

    def __init__(self, fidelity=None, provenance=None):
        """Constructs a new CalculationSource object.

        Parameters
        ----------
        fidelity : str
            The fidelity at which the property was calculated
        provenance : dict of str and Any
            A dictionary containing information about how the property was calculated.
        """

        self.fidelity = fidelity
        self.provenance = provenance

    def __getstate__(self):

        return {
            'fidelity': self.fidelity,
            'provenance': self.provenance,
        }

    def __setstate__(self, state):

        self.fidelity = state['fidelity']
        self.provenance = state['provenance']


class ParameterGradientKey:

    @property
    def tag(self):
        return self._tag

    @property
    def smirks(self):
        return self._smirks

    @property
    def attribute(self):
        return self._attribute

    def __init__(self, tag=None, smirks=None, attribute=None):

        self._tag = tag
        self._smirks = smirks
        self._attribute = attribute

    def __getstate__(self):

        return {
            'tag': self._tag,
            'smirks': self._smirks,
            'attribute': self._attribute
        }

    def __setstate__(self, state):

        self._tag = state['tag']
        self._smirks = state['smirks']
        self._attribute = state['attribute']

    def __str__(self):
        return f'tag={self._tag} smirks={self._smirks} attribute={self._attribute}'

    def __repr__(self):
        return f'<ParameterGradientKey {str(self)}>'

    def __hash__(self):
        return hash((self._tag, self._smirks, self._attribute))

    def __eq__(self, other):

        return (isinstance(other, ParameterGradientKey) and
                self._tag == other._tag and
                self._smirks == other._smirks and
                self._attribute == other._attribute)

    def __ne__(self, other):
        return not self.__eq__(other)


class ParameterGradient:

    @property
    def key(self):
        return self._key

    @property
    def value(self):
        return self._value

    def __init__(self, key=None, value=None):

        self._key = key
        self._value = value

    def __getstate__(self):

        return {
            'key': self._key,
            'value': self._value,
        }

    def __setstate__(self, state):

        self._key = state['key']
        self._value = state['value']

    def __str__(self):
        return f'key=({self._key}) value={self._value}'

    def __repr__(self):
        return f'<ParameterGradient key={self._key} value={self._value}>'


class PhysicalProperty(TypedBaseModel):
    """Represents the value of any physical property and it's uncertainty.

    It additionally stores the thermodynamic state at which the property
    was collected, the phase it was collected in, information about
    the composition of the observed system, and metadata about how the
    property was collected.
    """

    def __init__(self, thermodynamic_state=None, phase=PropertyPhase.Undefined,
                 substance=None, value=None, uncertainty=None, gradients=None, source=None):
        """Constructs a new PhysicalProperty object.

        Parameters
        ----------
        thermodynamic_state : ThermodynamicState
            The thermodynamic state that the property was measured in.
        phase : PropertyPhase
            The phase that the property was measured in.
        substance : Substance
            The composition of the substance that was measured.
        value: unit.Quantity
            The value of the measured physical property.
        uncertainty: unit.Quantity
            The uncertainty in the measured value.
        source: Source
            The source of this property.
        """
        self.id = str(uuid.uuid4())

        self.thermodynamic_state = thermodynamic_state
        self.phase = phase

        self.substance = substance

        self.value = value
        self.uncertainty = uncertainty

        self.gradients = []

        self.source = source

        self._metadata = {}

    def __getstate__(self):

        return {
            'id': self.id,

            'thermodynamic_state': self.thermodynamic_state,
            'phase': self.phase,
    
            'substance': self.substance,
    
            'value': self.value,
            'uncertainty': self.uncertainty,

            'gradients': self.gradients,
    
            'source': self.source,

            'metadata': self._metadata
        }

    def __setstate__(self, state):

        self.id = state['id']

        self.thermodynamic_state = state['thermodynamic_state']
        self.phase = state['phase']

        self.substance = state['substance']

        self.value = state['value']
        self.uncertainty = state['uncertainty']

        self.gradients = state['gradients']

        self.source = state['source']

        self._metadata = state['metadata']

    @property
    def temperature(self):
        """simtk.unit.Quantity or None: The temperature at which the property was collected."""
        return None if self.thermodynamic_state is None else self.thermodynamic_state.temperature

    @property
    def pressure(self):
        """simtk.unit.Quantity or None: The pressure at which the property was collected."""
        return None if self.thermodynamic_state is None else self.thermodynamic_state.pressure

    @property
    def metadata(self):
        """dict of str and Any: Additional metadata associated with this property, such as
        file paths to coordinate files or ...

        All property metadata will be made accessible to property estimation workflows.
        """
        return self._metadata

    @metadata.setter
    def metadata(self, value):
        self._metadata = value

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
    def get_default_workflow_schema(calculation_layer, options=None):
        """Returns the default workflow schema to use for
        a specific calculation layer.

        Parameters
        ----------
        calculation_layer: str
            The calculation layer which will attempt to execute the workflow
            defined by this schema.
        options: WorkflowOptions
            The options to use when setting up the default workflows.

        Returns
        -------
        WorkflowSchema
            The default workflow schema.
        """
        raise NotImplementedError()
