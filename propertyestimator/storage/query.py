"""
A collection of classes used to query a storage backend for
data which matches a set of criteria.
"""
import abc

from propertyestimator.attributes import Attribute, AttributeClass
from propertyestimator.datasets import PropertyPhase
from propertyestimator.storage import StoredSimulationData
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState


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


class SimulationDataQuery(BaseDataQuery):
    """A class used to query a `StorageBackend` for
    `StoredSimulationData` which meet the specified set
    of criteria.
    """

    @classmethod
    def supported_data_classes(cls):
        return [StoredSimulationData]

    substance = Attribute(
        docstring="The substance that the data should have been measured for.",
        type_hint=Substance,
        optional=False,
    )
    property_phase = Attribute(
        docstring="The phase of the substance (e.g. liquid, gas).",
        type_hint=PropertyPhase,
        optional=False,
    )

    thermodynamic_state = Attribute(
        docstring="The state at which the data should have been collected.",
        type_hint=ThermodynamicState,
        optional=True,
    )

    source_calculation_id = Attribute(
        docstring="The server id of the calculation which yielded this data.",
        type_hint=str,
        optional=True,
    )
    force_field_id = Attribute(
        docstring="The id of the force field parameters used to generate the data.",
        type_hint=str,
        optional=True,
    )

    total_number_of_molecules = Attribute(
        docstring="The total number of molecules in the system.",
        type_hint=int,
        optional=True,
    )
