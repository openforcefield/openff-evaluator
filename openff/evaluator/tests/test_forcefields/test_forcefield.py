import pytest

from openff.evaluator.forcefield import SmirnoffForceFieldSource
from openff.evaluator.utils import get_data_filename


def test_load_smirnoff_plugins():
    force_field_path = get_data_filename(
        "test/forcefields/buckingham-force-field.offxml"
    )

    obj = SmirnoffForceFieldSource.from_path(force_field_path, load_plugins=True)

    assert "DampedBuckingham68" in obj.to_force_field().registered_parameter_handlers

    SmirnoffForceFieldSource.from_path(force_field_path)

    with pytest.raises(
        KeyError,
        match="Cannot find a registered parameter handler class for tag 'DampedBuckingham68'",
    ):
        SmirnoffForceFieldSource.from_path(force_field_path)
