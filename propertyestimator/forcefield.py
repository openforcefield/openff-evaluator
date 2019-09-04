"""
A collection of wrappers around commonly employed force fields.
"""
import abc
from enum import Enum

from propertyestimator.utils.serialization import TypedBaseModel


class ForceFieldSource(TypedBaseModel):
    """A helper object to define the source of a force field
    and any associated meta data, such as version, file paths,
    or generation options.

    Notes
    -----
    It is likely that this class and classes based off of it will
    not be permanent fixtures of the framework, but rather will
    exist until the force fields can be stored in a uniform format /
    object model.
    """

    @abc.abstractmethod
    def __getstate__(self):
        pass

    @abc.abstractmethod
    def __setstate__(self, state):
        pass


class SmirnoffForceFieldSource(ForceFieldSource):
    """A wrapper around force fields based on the
    `SMIRks Native Open Force Field (SMIRNOFF) specification
    <https://open-forcefield-toolkit.readthedocs.io/en/latest/smirnoff.html>`_.
    """

    def __init__(self, inner_xml=None):
        """Constructs a new SmirnoffForceFieldSource object

        Parameters
        ----------
        inner_xml: str, optional
            A string containing the xml representation of the force field.
        """
        self._inner_xml = inner_xml

    def to_force_field(self):
        """Returns the SMIRNOFF force field created from this source.

        Returns
        -------
        openforcefield.typing.engines.smirnoff.ForceField
            The created force field.
        """
        from openforcefield.typing.engines import smirnoff
        return smirnoff.ForceField(self._inner_xml)

    @classmethod
    def from_object(cls, force_field):
        """Creates a new `SmirnoffForceFieldSource` from an existing
        `ForceField` object

        Notes
        -----
        All cosmetic attributes will be discarded.

        Parameters
        ----------
        force_field: openforcefield.typing.engines.smirnoff.ForceField
            The existing force field.

        Returns
        -------
        SmirnoffForceFieldSource
            The created object.
        """

        return cls(force_field.to_string('XML', True))

    @classmethod
    def from_path(cls, file_path):
        """Creates a new `SmirnoffForceFieldSource` from the file path to a
        `ForceField` object.

        Notes
        -----
        All cosmetic attributes will be discarded.

        Parameters
        ----------
        file_path: str
            The file path to the force field object. This may also be the
            name of a file which can be loaded via an entry point.

        Returns
        -------
        SmirnoffForceFieldSource
            The created object.
        """

        from openforcefield.typing.engines.smirnoff import ForceField

        force_field = ForceField(file_path, allow_cosmetic_attributes=True)
        return cls.from_object(force_field)

    def __getstate__(self):
        return {'inner_xml': self._inner_xml}

    def __setstate__(self, state):
        self._inner_xml = state['inner_xml']


class TLeapForceFieldSource(ForceFieldSource):
    """A wrapper around Amber force fields which may be
    applied via the `tleap` software package.

    Notes
    -----
    Currently this only supports force fields which are installed
    alongside `tleap`.
    """

    @property
    def leap_sources(self):
        """list of str: A list of the files which should be sourced by `leap`
        when applying the force field.
        """
        return self._leap_sources

    def __init__(self, leap_sources):
        """Constructs a new TLeapForceFieldSource object

        Parameters
        ----------
        leap_sources: list of str
            A list of the files which should be sourced by `leap`
            when applying the force field.

        Examples
        --------
        To create a source for the GAFF force field with tip3p water:

        >>> amber_gaff_source = TLeapForceFieldSource([
        >>>     'leaprc.gaff',
        >>>     'leaprc.water.tip3p'
        >>> ])

        To create a source for the GAFF 2 force field with tip3p water:

        >>> amber_gaff_2_source = TLeapForceFieldSource([
        >>>     'leaprc.gaff2',
        >>>     'leaprc.water.tip3p'
        >>> ])
        """

        self._leap_sources = leap_sources

    def __getstate__(self):
        return {'leap_sources': self._leap_sources}

    def __setstate__(self, state):
        self._leap_sources = state['leap_sources']


class LigParGenForceFieldSource(ForceFieldSource):
    """A wrapper and the OPLSAAM force field which can be applied
    via the `LigParGen server <http://zarbi.chem.yale.edu/ligpargen/>`_.

    References
    ----------
    [1] Potential energy functions for atomic-level simulations of water and organic and
        biomolecular systems. Jorgensen, W. L.; Tirado-Rives, J. Proc. Nat. Acad. Sci.
        USA 2005, 102, 6665-6670
    [2] 1.14*CM1A-LBCC: Localized Bond-Charge Corrected CM1A Charges for Condensed-Phase
        Simulations. Dodda, L. S.; Vilseck, J. Z.; Tirado-Rives, J.; Jorgensen, W. L.
        J. Phys. Chem. B, 2017, 121 (15), pp 3864-3870
    [3] LigParGen web server: An automatic OPLS-AA parameter generator for organic ligands.
        Dodda, L. S.;Cabeza de Vaca, I.; Tirado-Rives, J.; Jorgensen, W. L.
        Nucleic Acids Research, Volume 45, Issue W1, 3 July 2017, Pages W331-W336
    """

    class ChargeModel(Enum):

        CM1A_1_14_LBCC = '1.14*CM1A-LBCC'
        CM1A_1_14 = '1.14*CM1A'

    @property
    def preferred_charge_model(self):
        """ChargeModel: The preferred charge model to apply. In some cases
        the preferred charge model may not be applicable (e.g. 1.14*CM1A-LBCC
        may only be applied to neutral molecules) and so another model may be
        applied in its place.
        """
        return self._preferred_charge_model

    def __init__(self, preferred_charge_model=ChargeModel.CM1A_1_14_LBCC):
        """Constructs a new LigParGenForceFieldSource object

        Parameters
        ----------
        preferred_charge_model: ChargeModel
            The preferred charge model to apply. In some cases
            the preferred charge model may not be applicable
            (e.g. 1.14*CM1A-LBCC may only be applied to neutral
            molecules) and so another model may be applied in its
            place.
        """
        self._preferred_charge_model = preferred_charge_model

    def __getstate__(self):
        return {'preferred_charge_model': self._preferred_charge_model}

    def __setstate__(self, state):
        self._preferred_charge_model = state['preferred_charge_model']
