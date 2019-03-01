"""
Units tests for propertyestimator.layers.simulation
"""
import json

from propertyestimator.properties.density import Density
from propertyestimator.utils.serialization import TypedJSONEncoder

from propertyestimator.tests.utils import create_dummy_property


def test_physical_property_state_methods():

    dummy_property = create_dummy_property(Density)
    property_state = dummy_property.__getstate__()

    recreated_property = Density()
    recreated_property.__setstate__(property_state)

    recreated_state = recreated_property.__getstate__()

    original_json = json.dumps(property_state, cls=TypedJSONEncoder)
    recreated_json = json.dumps(recreated_state, cls=TypedJSONEncoder)

    assert original_json == recreated_json
