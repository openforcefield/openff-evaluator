"""
A collection of utilities which aid in sending and receiving messages sent over tcp.
"""

import struct
from enum import IntEnum

int_struct = struct.Struct("<i")

unpack_int = int_struct.unpack
pack_int = int_struct.pack


class EvaluatorMessageTypes(IntEnum):

    Undefined = 0
    Submission = 1
    Query = 2


def recvall(sock, n):
    # Helper function to recv n bytes or return None if EOF is hit
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data
