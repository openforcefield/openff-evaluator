"""
A collection of classes used to query a storage backend for
data which matches a set of criteria.
"""

import abc

from openff.evaluator.attributes import UNDEFINED, Attribute, AttributeClass
from openff.evaluator.datasets import PropertyPhase
from openff.evaluator.forcefield import ForceFieldSource
from openff.evaluator.storage.attributes import QueryAttribute
from openff.evaluator.storage.data import (
    ForceFieldData,
    StoredFreeEnergyData,
    StoredEquilibrationData,
    StoredSimulationData,
)
from openff.evaluator.substances import ExactAmount, Substance
from openff.evaluator.thermodynamics import ThermodynamicState


class BaseDataQuery(AttributeClass, abc.ABC):
    """A base class for queries which can be made to
    a `StorageBackend`.
    """

    @classmethod
    @abc.abstractmethod
    def data_class(cls):
        """The type of data class that this
        query can be applied to.

        Returns
        -------
        type of BaseStoredData
        """
        raise NotImplementedError()

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

        if not isinstance(data_object, self.data_class()):
            return None

        matches = []

        for attribute_name in self.get_attributes(QueryAttribute):
            attribute = getattr(self.__class__, attribute_name)

            if not hasattr(data_object, attribute_name) or attribute.custom_match:
                continue

            query_value = getattr(self, attribute_name)

            if query_value == UNDEFINED:
                continue

            data_value = getattr(data_object, attribute_name)

            matches.append(None if data_value != query_value else data_value)

        if any(x is None for x in matches):
            return None

        return tuple(matches)

    @classmethod
    def from_data_object(cls, data_object):
        """Returns the query which would match this data
        object.

        Parameters
        ----------
        data_object: BaseStoredData
            The data object to construct the query for.

        Returns
        -------
        cls
            The query which would match this data object.
        """
        query = cls()

        for attribute_name in cls.get_attributes():
            if not hasattr(data_object, attribute_name):
                continue

            attribute_value = getattr(data_object, attribute_name)
            setattr(query, attribute_name, attribute_value)

        return query


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

    # component_roles = QueryAttribute(
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


class ForceFieldQuery(BaseDataQuery):
    """A class used to query a `StorageBackend` for
    `ForceFieldData` which meet the specified criteria.
    """

    @classmethod
    def data_class(cls):
        return ForceFieldData

    force_field_source = QueryAttribute(
        docstring="The force field source to query for.",
        type_hint=ForceFieldSource,
        optional=True,
    )


class BaseSimulationDataQuery(BaseDataQuery, abc.ABC):
    """The base class for queries which will retrieve ``BaseSimulationData`` derived
    data.
    """

    substance = QueryAttribute(
        docstring="The substance which the data should have been collected "
        "for. Data for a subset of this substance can be queried for by "
        "using the `substance_query` attribute",
        type_hint=Substance,
        optional=True,
        custom_match=True,
    )
    substance_query = QueryAttribute(
        docstring="The subset of the `substance` to query for. This option "
        "can only be used when the `substance` attribute is set.",
        type_hint=SubstanceQuery,
        optional=True,
        custom_match=True,
    )

    thermodynamic_state = QueryAttribute(
        docstring="The state at which the data should have been collected.",
        type_hint=ThermodynamicState,
        optional=True,
    )
    property_phase = QueryAttribute(
        docstring="The phase of the substance (e.g. liquid, gas).",
        type_hint=PropertyPhase,
        optional=True,
    )

    source_calculation_id = QueryAttribute(
        docstring="The server id which should have generated this data.",
        type_hint=str,
        optional=True,
    )
    force_field_id = QueryAttribute(
        docstring="The id of the force field parameters which used to "
        "generate the data.",
        type_hint=str,
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
                data_amount = next(
                    iter(data_substance.get_amounts(component.identifier))
                )
                query_amount = next(
                    iter(self.substance.get_amounts(component.identifier))
                )

                if type(data_amount) is not type(query_amount):
                    continue

                if isinstance(data_amount, ExactAmount) and data_amount != query_amount:
                    # Make sure there is the same amount if we are
                    # dealing with exact amounts.
                    continue

                # A match was found.
                return data_substance

        return None

    def apply(self, data_object, attributes_to_ignore=None):
        matches = []

        # Apply a custom match behaviour for the substance
        # attribute.
        if self.substance != UNDEFINED:
            matches.append(self._match_substance(data_object))

        base_matches = super(BaseSimulationDataQuery, self).apply(data_object)
        base_matches = [None] if base_matches is None else base_matches

        matches = [*matches, *base_matches]

        if len(matches) == 0 or any(x is None for x in matches):
            return None

        return tuple(matches)

    def validate(self, attribute_type=None):
        super(BaseSimulationDataQuery, self).validate(attribute_type)

        if self.substance_query != UNDEFINED and self.substance == UNDEFINED:
            raise ValueError(
                "The `substance_query` can only be used when the "
                "`substance` attribute is set."
            )


class EquilibrationDataQuery(BaseSimulationDataQuery):
    """A class used to query a ``StorageBackend`` for ``StoredEquilibrationData`` objects
    which meet the specified set of criteria.
    """

    @classmethod
    def data_class(cls):
        return StoredEquilibrationData

    number_of_molecules = QueryAttribute(
        docstring="The total number of molecules in the system.",
        type_hint=int,
        optional=True,
    )

    max_number_of_molecules = QueryAttribute(
        docstring="The max/input number of molecules in the system.",
        type_hint=int,
        optional=True,
    )

    calculation_layer = QueryAttribute(
        docstring="Calculation layer type (e.g. 'SimulationLayer', 'ReweightingLayer')",
        type_hint=str,
        optional=True,
    )


class SimulationDataQuery(BaseSimulationDataQuery):
    """A class used to query a ``StorageBackend`` for ``StoredSimulationData`` objects
    which meet the specified set of criteria.
    """

    @classmethod
    def data_class(cls):
        return StoredSimulationData

    number_of_molecules = QueryAttribute(
        docstring="The total number of molecules in the system.",
        type_hint=int,
        optional=True,
    )


class FreeEnergyDataQuery(BaseSimulationDataQuery):
    """A class used to query a ``StorageBackend`` for ``FreeEnergyData`` objects which
    meet the specified set of criteria.
    """

    @classmethod
    def data_class(cls):
        return StoredFreeEnergyData
