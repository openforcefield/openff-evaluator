"""
Units tests for propertyestimator.protocols.storage
"""
import json
import os
import tempfile

from propertyestimator.protocols.storage import (
    UnpackStoredDataCollection,
    UnpackStoredSimulationData,
)
from propertyestimator.storage.dataclasses import StoredDataCollection
from propertyestimator.tests.utils import (
    build_tip3p_smirnoff_force_field,
    create_dummy_stored_simulation_data,
    create_dummy_substance,
)
from propertyestimator.utils.exceptions import PropertyEstimatorException
from propertyestimator.utils.serialization import TypedJSONEncoder


def test_unpack_stored_simulation_data():
    """A test that compatible simulation data gets merged
    together within the`LocalStorage` system."""

    with tempfile.TemporaryDirectory() as directory:

        force_field_path = os.path.join(directory, "ff.json")

        with open(force_field_path, "w") as file:
            file.write(build_tip3p_smirnoff_force_field().json())

        dummy_substance = create_dummy_substance(1)

        dummy_directory_path = os.path.join(directory, "data")
        dummy_data_path = os.path.join(directory, "data.json")

        data_coordinate_name = "data_1.pdb"

        data_object = create_dummy_stored_simulation_data(
            directory_path=dummy_directory_path,
            substance=dummy_substance,
            force_field_id="ff_id_1",
            coordinate_file_name=data_coordinate_name,
            statistical_inefficiency=1.0,
        )

        with open(dummy_data_path, "w") as file:
            json.dump(data_object, file, cls=TypedJSONEncoder)

        unpack_stored_data = UnpackStoredSimulationData("unpack_data")
        unpack_stored_data.simulation_data_path = (
            dummy_data_path,
            dummy_directory_path,
            force_field_path,
        )

        result = unpack_stored_data.execute(directory, None)
        assert not isinstance(result, PropertyEstimatorException)


def test_unpack_stored_data_collection():

    with tempfile.TemporaryDirectory() as directory:

        force_field_path = os.path.join(directory, "ff.json")

        with open(force_field_path, "w") as file:
            file.write(build_tip3p_smirnoff_force_field().json())

        dummy_substance = create_dummy_substance(1)

        data_collection_directory = os.path.join(directory, f"dummy_collection")

        os.makedirs(data_collection_directory, exist_ok=True)

        dummy_data_collection = StoredDataCollection()
        dummy_data_collection.substance = dummy_substance
        dummy_data_collection.force_field_id = "ff_id"

        for inner_data_index in range(2):

            data_key = f"data_{inner_data_index}"
            data_path = os.path.join(data_collection_directory, data_key)

            data_coordinate_name = f"data_{inner_data_index}.pdb"

            inner_data = create_dummy_stored_simulation_data(
                directory_path=data_path,
                substance=dummy_substance,
                force_field_id=f"ff_id",
                coordinate_file_name=data_coordinate_name,
                statistical_inefficiency=1.0,
            )

            dummy_data_collection.data[data_key] = inner_data

        dummy_data_path = os.path.join(directory, "data_collection.json")

        with open(dummy_data_path, "w") as file:
            json.dump(dummy_data_collection, file, cls=TypedJSONEncoder)

        unpack_data_collection = UnpackStoredDataCollection(f"unpack_data_collection")
        unpack_data_collection.input_data_path = (
            dummy_data_path,
            data_collection_directory,
            force_field_path,
        )

        result = unpack_data_collection.execute(directory, None)
        assert not isinstance(result, PropertyEstimatorException)
