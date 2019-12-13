"""
A collection of classes used to query a storage backend for
data which matches a set of criteria.
"""
import abc

from propertyestimator.attributes import UNDEFINED, Attribute, AttributeClass
from propertyestimator.datasets import PropertyPhase
from propertyestimator.storage import StoredSimulationData


class BaseDataQuery(AttributeClass, abc.ABC):
    """A base class for queries which can be made to
    a `StorageBackend`.
    """

    @classmethod
    @abc.abstractmethod
    def supported_data_classes(cls):
        """Returns the types of data classes that this
        query can be applied to.

        Returns
        -------
        list of type of BaseStoredData
        """
        raise NotImplementedError()


class SubstanceQuery(AttributeClass, abc.ABC):
    """A query which focuses on finding data which was
    collected for substances with specific traits, e.g
    which contains both a solute and solvent, or only a
    solvent etc.
    """

    components_only = Attribute(
        docstring="Optionally apply this query to each component "
        "in the substance of interest.",
        type_hint=bool,
        default_value=False,
    )

    component_roles = Attribute(
        docstring="Returns data for only the subset of a substance "
        "which has the requested roles.",
        type_hint=list,
        optional=True,
    )

    def validate(self, attribute_type=None):

        super(SubstanceQuery, self).validate(attribute_type)

        if (
            self.components_only
            and self.component_roles != UNDEFINED
            and len(self.components_only) > 0
        ):

            raise ValueError(
                "The `component_roles` attribute cannot be used when "
                "the `components_only` attribute is `True`."
            )


class SimulationDataQuery(BaseDataQuery):
    """A class used to query a `StorageBackend` for
    `StoredSimulationData` which meet the specified set
    of criteria.
    """

    @classmethod
    def supported_data_classes(cls):
        return [StoredSimulationData]

    substance_query = Attribute(
        docstring="The a query for the substance that the data should have "
        "been measured for.",
        type_hint=SubstanceQuery,
        optional=True,
    )
    property_phase = Attribute(
        docstring="The phase of the substance (e.g. liquid, gas).",
        type_hint=PropertyPhase,
        optional=True,
    )

    total_number_of_molecules = Attribute(
        docstring="The total number of molecules in the system.",
        type_hint=int,
        optional=True,
    )
