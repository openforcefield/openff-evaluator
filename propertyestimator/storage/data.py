"""
A collection of classes representing data stored by a storage backend.
"""
import abc
import hashlib

from propertyestimator.attributes import AttributeClass
from propertyestimator.datasets import PropertyPhase
from propertyestimator.forcefield import ForceFieldSource
from propertyestimator.storage.attributes import (
    ComparisonBehaviour,
    FilePath,
    StorageAttribute,
)
from propertyestimator.substances import Substance
from propertyestimator.thermodynamics import ThermodynamicState


class BaseStoredData(AttributeClass, abc.ABC):
    """A base representation of cached data to be stored by
    a storage backend.

    The expectation is that stored data may exist in storage
    as two parts:

        1) A JSON serialized representation of this class (or
           a subclass), which contains lightweight information
           such as the state and composition of the system. Any
           larger pieces of data, such as coordinates or
           trajectories, should be referenced as a file name.

        2) A directory like structure (either directly a directory,
           or some NetCDF like compressed archive) of ancillary
           files which do not easily lend themselves to be
           serialized within a JSON object, whose files are referenced
           by their file name by the data object.

    The ancillary directory-like structure is not required if the
    data may be suitably stored in the data object itself.
    """

    @classmethod
    @abc.abstractmethod
    def has_ancillary_data(cls):
        """Returns whether this data object requires an
        accompanying data directory-like structure.

        Returns
        -------
        bool
            True if this class requires an accompanying
            data directory-like structure.
        """
        raise NotImplementedError()


class HashableStoredData(BaseStoredData, abc.ABC):
    """Represents a class of data objects which can be
    compared directly solely by their hash values.
    """

    def __eq__(self, other):
        return type(self) == type(other) and hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError


class MergeableStoredData(BaseStoredData, abc.ABC):
    """Represents a class of data objects which may
    contain redundant data and can be merged into a
    single object in such cases.
    """

    @staticmethod
    def are_compatible(stored_data_1, stored_data_2):

        if type(stored_data_1) != type(stored_data_2):
            return False

        attribute_names = stored_data_1.__class__.get_attributes(StorageAttribute)

        for name in attribute_names:

            attribute = getattr(stored_data_1.__class__, name)

            if attribute.comparison_behavior == ComparisonBehaviour.Ignore:
                continue

            value_1 = getattr(stored_data_1, name)
            value_2 = getattr(stored_data_2, name)

            if value_1 == value_2:
                continue

            return False

        return True

    @staticmethod
    @abc.abstractmethod
    def most_information(stored_data_1, stored_data_2):
        """Returns the data object with the largest information
         content. The two data objects must be compatible (see
         `are_compatible`).

        Parameters
        ----------
        stored_data_1: BaseStoredData
            The first piece of data to compare.
        stored_data_2: BaseStoredData
            The second piece of data to compare.

        Returns
        -------
        BaseStoredData
            The data object with the highest information
            content.
        """
        raise NotImplementedError()


class StoredSimulationData(MergeableStoredData):
    """A representation of data which has been cached
    from a single previous simulation.

    Notes
    -----
    The ancillary directory which stores larger information such
    as trajectories should be of the form:

    .. code-block::

        |--- data_object.json
        |--- data_directory
             |--- coordinate_file_name.pdb
             |--- trajectory_file_name.dcd
             |--- statistics_file_name.csv
    """

    substance = StorageAttribute(
        docstring="A description of the composition of the stored system.",
        type_hint=Substance,
    )
    thermodynamic_state = StorageAttribute(
        docstring="The state at which the data was collected.",
        type_hint=ThermodynamicState,
    )
    property_phase = StorageAttribute(
        docstring="The phase of the system (e.g. liquid, gas).",
        type_hint=PropertyPhase,
    )

    source_calculation_id = StorageAttribute(
        docstring="The server id of the calculation which yielded this data.",
        type_hint=str,
        comparison_behavior=ComparisonBehaviour.Ignore,
        optional=True,
    )
    provenance = StorageAttribute(
        docstring="Provenance information about how this data was generated.",
        type_hint=dict,
        comparison_behavior=ComparisonBehaviour.Ignore,
        optional=True,
    )

    force_field_id = StorageAttribute(
        docstring="The id of the force field parameters used to generate the data.",
        type_hint=str,
    )

    coordinate_file_name = StorageAttribute(
        docstring="The name of a coordinate file which encodes the "
        "topology information of the system.",
        type_hint=FilePath,
        comparison_behavior=ComparisonBehaviour.Ignore,
    )
    trajectory_file_name = StorageAttribute(
        docstring="The name of a .dcd trajectory file containing "
        "configurations generated by the simulation.",
        type_hint=FilePath,
        comparison_behavior=ComparisonBehaviour.Ignore,
    )

    statistics_file_name = StorageAttribute(
        docstring="The name of a `StatisticsArray` csv file, containing "
        "statistics generated by the simulation.",
        type_hint=FilePath,
        comparison_behavior=ComparisonBehaviour.Ignore,
    )
    statistical_inefficiency = StorageAttribute(
        docstring="The statistical inefficiency of the collected data.",
        type_hint=float,
        comparison_behavior=ComparisonBehaviour.Ignore,
    )

    number_of_molecules = StorageAttribute(
        docstring="The total number of molecules in the system.", type_hint=int,
    )

    @classmethod
    def has_ancillary_data(cls):
        return True

    @staticmethod
    def most_information(stored_data_1, stored_data_2):
        """Returns the data object with the lowest
        `statistical_inefficiency`.
        """

        assert isinstance(stored_data_1, StoredSimulationData)
        assert isinstance(stored_data_2, StoredSimulationData)

        # Make sure the two objects can actually be merged.
        if not StoredSimulationData.are_compatible(stored_data_1, stored_data_2):

            raise ValueError(
                "The two pieces of data are incompatible and cannot "
                "be merged into one."
            )

        if (
            stored_data_1.statistical_inefficiency
            < stored_data_2.statistical_inefficiency
        ):
            return stored_data_1

        return stored_data_2


class ForceFieldData(HashableStoredData):
    """A data container for force field objects which
    will be saved to disk.
    """

    force_field_source = StorageAttribute(
        docstring="The force field source object.", type_hint=ForceFieldSource,
    )

    @classmethod
    def has_ancillary_data(cls):
        return False

    def __hash__(self):

        force_field_string = self.force_field_source.json()
        return hashlib.sha256(force_field_string.encode()).hexdigest()
