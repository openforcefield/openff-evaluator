from openff.evaluator.workflow.utils import ProtocolPath


def test_protocol_path_id_replacement():
    """Tests that the protocol id function on the protocol path
    behaves as expected."""

    protocol_path = ProtocolPath("", "protocol_id_1", "protocol_id_11")
    assert protocol_path.full_path == "protocol_id_1/protocol_id_11."

    # Make sure only full matches lead to id replacement
    protocol_path.replace_protocol("protocol_id_", "new_id_1")
    assert protocol_path.full_path == "protocol_id_1/protocol_id_11."

    protocol_path.replace_protocol("rotocol_id_1", "new_id_1")
    assert protocol_path.full_path == "protocol_id_1/protocol_id_11."

    protocol_path.replace_protocol("protocol_id_1", "new_id_1")
    assert protocol_path.full_path == "new_id_1/protocol_id_11."
