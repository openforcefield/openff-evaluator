"""
Units tests for propertyestimator.utils.tcp
"""
from propertyestimator.utils import tcp


def test_message_packing():
    """Test that packing / unpacking ints works as expected"""
    assert tcp.unpack_int(tcp.pack_int(20))[0] == 20


def test_message_type_enum():
    """Test the message type enum creation."""
    assert (
        tcp.PropertyEstimatorMessageTypes(0)
        == tcp.PropertyEstimatorMessageTypes.Undefined
    )

    assert (
        tcp.PropertyEstimatorMessageTypes(1)
        == tcp.PropertyEstimatorMessageTypes.Submission
    )

    assert (
        tcp.PropertyEstimatorMessageTypes(2) == tcp.PropertyEstimatorMessageTypes.Query
    )
