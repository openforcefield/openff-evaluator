"""A collection of classes to help track the provenance
of measured / estimated properties.
"""
from openff.evaluator.utils.serialization import TypedBaseModel


class Source(TypedBaseModel):
    """Container class for information about how a property was measured / calculated.

    .. todo:: Swap this out with a more general provenance class.
    """

    def __getstate__(self):
        return {}

    def __setstate__(self, state):
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

    def __init__(self, doi="", reference=""):
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
            "doi": self.doi,
            "reference": self.reference,
        }

    def __setstate__(self, state):
        self.doi = state["doi"]
        self.reference = state["reference"]


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
            "fidelity": self.fidelity,
            "provenance": self.provenance,
        }

    def __setstate__(self, state):
        self.fidelity = state["fidelity"]
        self.provenance = state["provenance"]
