"""
Units tests for openff.evaluator.utils.tcp
"""
from openff.evaluator.utils import tcp


def test_message_packing():
    """Test that packing / unpacking ints works as expected"""
    assert tcp.unpack_int(tcp.pack_int(20))[0] == 20


def test_message_type_enum():
    """Test the message type enum creation."""
    assert tcp.EvaluatorMessageTypes(0) == tcp.EvaluatorMessageTypes.Undefined

    assert tcp.EvaluatorMessageTypes(1) == tcp.EvaluatorMessageTypes.Submission

    assert tcp.EvaluatorMessageTypes(2) == tcp.EvaluatorMessageTypes.Query
