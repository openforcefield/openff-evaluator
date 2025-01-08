"""
A collection of classes representing data stored by a storage backend.
"""

import abc
from typing import Optional

from openff.evaluator.attributes import AttributeClass
from openff.evaluator.datasets import PropertyPhase
from openff.evaluator.forcefield import ForceFieldSource
from openff.evaluator.storage.attributes import FilePath, StorageAttribute
from openff.evaluator.substances import Substance
from openff.evaluator.thermodynamics import ThermodynamicState
from openff.evaluator.utils.observables import Observable, ObservableFrame


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

    def to_storage_query(self):
        """Returns the storage query which would match this
        data object.

        Returns
        -------
        BaseDataQuery
            The storage query which would match this
            data object.
        """
        raise NotImplementedError()


class HashableStoredData(BaseStoredData, abc.ABC):
    """Represents a class of data objects which can be
    rapidly compared / indexed by their hash values.
    """

    def __eq__(self, other):
        return type(self) is type(other) and hash(self) == hash(other)

    def __ne__(self, other):
        return not self.__eq__(other)

    @abc.abstractmethod
    def __hash__(self):
        raise NotImplementedError


class ForceFieldData(HashableStoredData):
    """A data container for force field objects which
    will be saved to disk.
    """

    force_field_source = StorageAttribute(
        docstring="The force field source object.",
        type_hint=ForceFieldSource,
    )

    @classmethod
    def has_ancillary_data(cls):
        return False

    def to_storage_query(self):
        """
        Returns
        -------
        SimulationDataQuery
            The storage query which would match this
            data object.
        """
        from .query import ForceFieldQuery

        return ForceFieldQuery.from_data_object(self)

    def __eq__(self, other):
        return super(ForceFieldData, self).__eq__(other)

    def __ne__(self, other):
        return super(ForceFieldData, self).__ne__(other)

    def __hash__(self):
        force_field_string = self.force_field_source.json()
        return hash(force_field_string.encode())


class ReplaceableData(BaseStoredData, abc.ABC):
    """Represents a piece of stored data which can be
    replaced in a `StorageBackend` by another piece of
    data of the same type.

    This may be the case for example when attempting to
    store a piece of `StoredSimulationData`, but another
    piece of data measured from the same calculation and
    for the same system already exists in the system, but
    stores less configurations.
    """

    @classmethod
    @abc.abstractmethod
    def most_information(cls, stored_data_1, stored_data_2):
        """Returns the data object with the highest information
        content.

        Parameters
        ----------
        stored_data_1: ReplaceableData
            The first piece of data to compare.
        stored_data_2: ReplaceableData
            The second piece of data to compare.

        Returns
        -------
        ReplaceableData, optional
            The data object with the highest information
            content, or `None` if the two pieces of information
            are incompatible with one another.
        """

        assert isinstance(stored_data_1, ReplaceableData)
        assert type(stored_data_1) is type(stored_data_2)

        # Make sure the two objects are compatible.
        data_query = stored_data_1.to_storage_query()

        if data_query.apply(stored_data_2) is None:
            return None

        return stored_data_1


class BaseSimulationData(ReplaceableData, abc.ABC):
    """A base class for classes which will store the outputs of a molecular simulation"""

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
    )

    force_field_id = StorageAttribute(
        docstring="The id of the force field parameters used to generate the data.",
        type_hint=str,
    )

    @classmethod
    def has_ancillary_data(cls):
        return True


class StoredSimulationData(BaseSimulationData):
    """A representation of data which has been cached from a single previous simulation.

    Notes
    -----
    The ancillary directory which stores larger information such as trajectories should
    be of the form:

    .. code-block::

        |--- data_object.json
        |--- data_directory
             |--- coordinate_file_name.pdb
             |--- trajectory_file_name.dcd
    """

    coordinate_file_name = StorageAttribute(
        docstring="The name of a coordinate file which encodes the "
        "topology information of the system.",
        type_hint=FilePath,
    )
    trajectory_file_name = StorageAttribute(
        docstring="The name of a .dcd trajectory file containing "
        "configurations generated by the simulation.",
        type_hint=FilePath,
    )

    observables = StorageAttribute(
        docstring="A frame of observables collected over the duration of the "
        "simulation.",
        type_hint=ObservableFrame,
    )
    statistical_inefficiency = StorageAttribute(
        docstring="The statistical inefficiency of the collected data.",
        type_hint=float,
    )

    number_of_molecules = StorageAttribute(
        docstring="The total number of molecules in the system.",
        type_hint=int,
    )

    max_number_of_molecules = StorageAttribute(
        docstring="The max number of molecules allowed in the system",
        type_hint=int,
    )

    calculation_layer = StorageAttribute(
        docstring="The CalculationLayer used to generate this data.",
        type_hint=str,
        optional=True,
    )

    @classmethod
    def most_information(cls, stored_data_1, stored_data_2):
        """Returns the data object with the lowest
        `statistical_inefficiency`.

        Parameters
        ----------
        stored_data_1: StoredSimulationData
            The first piece of data to compare.
        stored_data_2: StoredSimulationData
            The second piece of data to compare.

        Returns
        -------
        StoredSimulationData
        """
        if (
            super(StoredSimulationData, cls).most_information(
                stored_data_1, stored_data_2
            )
            is None
        ):
            return None
        if (
            stored_data_1.statistical_inefficiency
            < stored_data_2.statistical_inefficiency
        ):
            return stored_data_1

        return stored_data_2

    def to_storage_query(self):
        """
        Returns
        -------
        SimulationDataQuery
            The storage query which would match this
            data object.
        """
        from .query import SimulationDataQuery

        return SimulationDataQuery.from_data_object(self)


class StoredFreeEnergyData(BaseSimulationData):
    """A representation of data which has been cached from an free energy calculation
    which computed the free energy difference between a start and end state.

    Notes
    -----
    The ancillary directory which stores larger information such as trajectories should
    be of the form:

    .. code-block::

        |--- data_object.json
        |--- data_directory
             |--- topology_file_name.pdb
             |--- start_state_trajectory.dcd
             |--- end_state_trajectory.dcd
    """

    free_energy_difference = StorageAttribute(
        docstring="The free energy difference between the end state "
        "and the start state.",
        type_hint=Observable,
    )

    topology_file_name = StorageAttribute(
        docstring="The name of a coordinate file which encodes the topology of the "
        "system.",
        type_hint=FilePath,
    )

    start_state_trajectory = StorageAttribute(
        docstring="The name of a .dcd trajectory file containing configurations "
        "generated by the simulation of the start state of the system.",
        type_hint=FilePath,
    )
    end_state_trajectory = StorageAttribute(
        docstring="The name of a .dcd trajectory file containing configurations "
        "generated by the simulation of the end state of the system.",
        type_hint=FilePath,
    )

    @classmethod
    def most_information(
        cls,
        stored_data_1: "StoredFreeEnergyData",
        stored_data_2: "StoredFreeEnergyData",
    ) -> Optional["StoredFreeEnergyData"]:
        """A comparison function which will always retain both pieces of free energy
        data. At this time no situation can be envisaged that the same free energy data
        from exactly the same calculation will be store.

        Parameters
        ----------
        stored_data_1
            The first piece of data to compare.
        stored_data_2:
            The second piece of data to compare.
        """
        return None

    def to_storage_query(self):
        """
        Returns
        -------
        FreeEnergyDataQuery
            The storage query which would match this data object.
        """
        from .query import FreeEnergyDataQuery

        return FreeEnergyDataQuery.from_data_object(self)
