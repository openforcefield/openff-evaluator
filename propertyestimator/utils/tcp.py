"""
A collection of utilities which aid in sending and receiving messages sent over tcp.
"""

import struct
from enum import IntEnum


int_struct = struct.Struct("<i")

unpack_int = int_struct.unpack
pack_int = int_struct.pack


class PropertyEstimatorMessageTypes(IntEnum):

    Undefined = 0
    Submission = 1
    Query = 2
