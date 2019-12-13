"""
A collection of classes used to query a storage backend for
data which matches a set of criteria.
"""
import abc

from propertyestimator.attributes import UNDEFINED, Attribute, AttributeClass
from propertyestimator.datasets import PropertyPhase
from propertyestimator.storage import StoredSimulationData
from propertyestimator.substances import Substance


class BaseDataQuery(AttributeClass, abc.ABC):
    """A base class for queries which can be made to
    a `StorageBackend`.
    """

    @classmethod
    @abc.abstractmethod
    def supported_data_class(cls):
        """Returns the type of data class that this
        query can be applied to.

        Returns
        -------
        type of BaseStoredData
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def apply(self, data_object):
        """Apply this query to a data object.

        Parameters
        ----------
        data_object: BaseStoredData
            The data object to apply the query to.

        Returns
        -------
        tuple of Any, optional
            The values of the matched parameters of the data
            object fully matched this query, otherwise `None`.
        """
        raise NotImplementedError()


class SubstanceQuery(AttributeClass, abc.ABC):
    """A query which focuses on finding data which was
    collected for substances with specific traits, e.g
    which contains both a solute and solvent, or only a
    solvent etc.
    """

    components_only = Attribute(
        docstring="Only match pure data which was collected for "
        "one of the components in the query substance.",
        type_hint=bool,
        default_value=False,
    )

    # component_roles = Attribute(
    #     docstring="Returns data for only the subset of a substance "
    #     "which has the requested roles.",
    #     type_hint=list,
    #     optional=True,
    # )

    def validate(self, attribute_type=None):

        super(SubstanceQuery, self).validate(attribute_type)

        # if (
        #     self.components_only
        #     and self.component_roles != UNDEFINED
        #     and len(self.components_only) > 0
        # ):
        #
        #     raise ValueError(
        #         "The `component_roles` attribute cannot be used when "
        #         "the `components_only` attribute is `True`."
        #     )


class SimulationDataQuery(BaseDataQuery):
    """A class used to query a `StorageBackend` for
    `StoredSimulationData` which meet the specified set
    of criteria.
    """

    @classmethod
    def supported_data_class(cls):
        return StoredSimulationData

    substance: Substance = Attribute(
        docstring="The substance which the data should have been collected "
        "for. Data for a subset of this substance can be queried for by "
        "using the `substance_query` attribute",
        type_hint=Substance,
        optional=True,
    )
    substance_query: SubstanceQuery = Attribute(
        docstring="The subset of the `substance` to query for. This option "
        "can only be used when the `substance` attribute is set.",
        type_hint=SubstanceQuery,
        optional=True,
    )

    property_phase = Attribute(
        docstring="The phase of the substance (e.g. liquid, gas).",
        type_hint=PropertyPhase,
        optional=True,
    )

    number_of_molecules = Attribute(
        docstring="The total number of molecules in the system.",
        type_hint=int,
        optional=True,
    )

    def _match_substance(self, data_object):
        """Attempt to match the substance (or a subset of it).

        Parameters
        ----------
        data_object: StoredSimulationData
            The data object to match against.

        Returns
        -------
        Substance, optional
            The matched substance if a match is made, otherwise
            `None`.
        """
        if self.substance == UNDEFINED:
            return None

        data_substance: Substance = data_object.substance

        if self.substance_query == UNDEFINED:
            return None if self.substance != data_substance else self.substance

        # Handle the sub-substance match.
        if self.substance_query.components_only:

            if data_substance.number_of_components != 1:
                # We are only interested in pure data.
                return None

            for component in self.substance.components:

                if component.smiles != data_substance.components[0].smiles:
                    continue

                # Make sure the amount type matches up i.e either both
                # are defined in mole fraction, or both as an exact amount.
                data_amount = next(iter(data_substance.get_amounts(component.smiles)))
                query_amount = next(iter(self.substance.get_amounts(component.smiles)))

                if type(data_amount) != type(query_amount):
                    continue

                if (
                    isinstance(data_amount, Substance.ExactAmount)
                    and data_amount != query_amount
                ):
                    # Make sure there is the same amount if we are
                    # dealing with exact amounts.
                    continue

                # A match was found.
                return data_substance

        return None

    def apply(self, data_object):

        if not isinstance(data_object, StoredSimulationData):
            return False

        matches = []

        # Check the substance
        if self.substance != UNDEFINED:
            matches.append(self._match_substance(data_object))

        # Check the phase.
        if self.property_phase != UNDEFINED:

            matches.append(
                None
                if data_object.property_phase != self.property_phase
                else self.property_phase
            )

        # Check the molecule count.
        if self.number_of_molecules != UNDEFINED:

            matches.append(
                None
                if data_object.number_of_molecules != self.number_of_molecules
                else self.number_of_molecules
            )

        if any(x is None for x in matches):
            return None

        return tuple(matches)

    def validate(self, attribute_type=None):
        super(SimulationDataQuery, self).validate(attribute_type)

        if self.substance_query != UNDEFINED and self.substance == UNDEFINED:

            raise ValueError(
                "The `substance_query` can only be used when the "
                "`substance` attribute is set."
            )
