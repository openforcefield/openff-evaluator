"""A collection of classes to help track the provenance
of measured / estimated properties.
"""
import abc

from evaluator.attributes import UNDEFINED, Attribute, AttributeClass


class Source(AttributeClass, abc.ABC):
    """The base class for objects which store information about how a
    property was measured experimentally / estimated for simulation
    data.
    """

    pass


class MeasurementSource(Source):
    """Contains metadata about how a physical property was measured by experiment.

    This class contains either the DOI and/or the reference, but must contain at
    least one.

    Attributes
    ----------
    doi : str or None, default None
        The DOI for the source, preferred way to identify for source
    reference : str
        The long form description of the source if no DOI is available, or more
        information is needed or wanted.
    """

    doi = Attribute(
        docstring="The digital object identifier of the source from which this "
        "measurement was obtained.",
        type_hint=str,
        default_value=UNDEFINED,
        optional=True,
    )
    reference = Attribute(
        docstring="An alternative identifier of the source from which this "
        "measurement was obtained, e.g. a URL.",
        type_hint=str,
        default_value=UNDEFINED,
        optional=True,
    )

    def __init__(self, doi=None, reference=None):
        """Constructs a new MeasurementSource object.

        Parameters
        ----------
        doi : str, optional
            The digital object identifier of the source from which this
            measurement was obtained.
        reference : str
            An alternative identifier of the source from which this
            measurement was obtained, e.g. a URL.
        """

        if doi is not None:
            self.doi = doi
        if reference is not None:
            self.reference = reference

    def validate(self, attribute_type=None):

        super(MeasurementSource, self).validate(attribute_type)
        assert self.doi != UNDEFINED or self.reference != UNDEFINED


class CalculationSource(Source):
    """Contains information about how a physical property was estimated.

    This includes provenance such as the unique id of the estimation request
    which produced this measurement, the unique id of the force field used in
    the request, and the fidelity at which the property was estimated (e.g
    direct simulation, cached simulation data reweighting, etc.).

    Attributes
    ----------
    request_id: str
        The unique id of the estimation request which produced this measurement
    force_field_id: str
        The unique id of the force field used in the request (as assigned by
        an `EvaluatorServer`.
    fidelity : str
        The fidelity at which the property was calculated.
    """

    request_id = Attribute(
        docstring="The unique id of the estimation request which produced this "
        "measurement.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    force_field_id = Attribute(
        docstring="The unique id of the force field used in the request (as assigned "
        "by an `EvaluatorServer`.",
        type_hint=str,
        default_value=UNDEFINED,
    )
    fidelity = Attribute(
        docstring="The fidelity at which the property was calculated.",
        type_hint=str,
        default_value=UNDEFINED,
    )

    def __init__(self, request_id=None, force_field_id=None, fidelity=None):
        """Constructs a new CalculationSource object.

        Parameters
        ----------
        request_id: str
            The unique id of the estimation request which produced this measurement
        force_field_id: str
            The unique id of the force field used in the request (as assigned by
            an `EvaluatorServer`.
        fidelity : str
            The fidelity at which the property was calculated
        """

        if request_id is not None:
            self.request_id = request_id
        if force_field_id is not None:
            self.force_field_id = force_field_id
        if fidelity is not None:
            self.fidelity = fidelity
