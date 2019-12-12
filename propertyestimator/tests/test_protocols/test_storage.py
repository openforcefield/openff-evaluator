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
