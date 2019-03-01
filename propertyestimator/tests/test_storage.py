"""
Units tests for propertyestimator.storage
"""
import json
import tempfile

from simtk import unit

from propertyestimator.storage import LocalFileStorage, StoredSimulationData
from propertyestimator.substances import Mixture
from propertyestimator.thermodynamics import ThermodynamicState
from propertyestimator.utils import get_data_filename
from propertyestimator.utils.serialization import serialize_force_field


def test_local_force_field_storage():
    """A simple test to that force fields can be stored and
    retrieved using the local storage backend."""

    from openforcefield.typing.engines import smirnoff
    force_field = smirnoff.ForceField(get_data_filename('forcefield/smirnoff99Frosst.offxml'))

    with tempfile.TemporaryDirectory() as temporary_directory:

        local_storage = LocalFileStorage(temporary_directory)
        local_storage.store_force_field('tmp_id', force_field)

        retrieved_force_field = local_storage.retrieve_force_field('tmp_id')

        serialized_force_field = serialize_force_field(force_field)
        serialized_retrieved_force_field = serialize_force_field(retrieved_force_field)

        assert json.dumps(serialized_force_field) == json.dumps(serialized_retrieved_force_field)

        local_storage_new = LocalFileStorage(temporary_directory)
        assert local_storage_new.has_force_field(force_field)


def test_local_simulation_storage():
    """A simple test to that force fields can be stored and
    retrieved using the local storage backend."""

    substance = Mixture()
    substance.add_component('C', 1.0, False)

    dummy_simulation_data = StoredSimulationData()

    dummy_simulation_data.thermodynamic_state = ThermodynamicState(298.0*unit.kelvin,
                                                                   1.0*unit.atmosphere)

    dummy_simulation_data.statistical_inefficiency = 1.0
    dummy_simulation_data.force_field_id = 'tmp_ff_id'

    dummy_simulation_data.substance = substance

    with tempfile.TemporaryDirectory() as temporary_directory:

        local_storage = LocalFileStorage(temporary_directory)
        local_storage.store_simulation_data(substance.identifier, dummy_simulation_data)

        retrieved_data_array = local_storage.retrieve_simulation_data(substance)
        assert len(retrieved_data_array) == 1

        retrieved_data = retrieved_data_array[substance.identifier][0]

        assert dummy_simulation_data.thermodynamic_state == retrieved_data.thermodynamic_state
        assert dummy_simulation_data.statistical_inefficiency == retrieved_data.statistical_inefficiency
        assert dummy_simulation_data.force_field_id == retrieved_data.force_field_id
        assert dummy_simulation_data.substance == retrieved_data.substance

        local_storage_new = LocalFileStorage(temporary_directory)
        assert local_storage_new.has_object(dummy_simulation_data.unique_id)
