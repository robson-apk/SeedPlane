"""Versioned binary framing for persistent HIVE agent connections.

The envelope is binary and length-delimited; control/work payloads are compact
UTF-8 JSON in v1. Model tensors are intentionally not sent by this protocol.
"""
from dataclasses import dataclass
from enum import IntEnum
import json
import socket
import struct


MAGIC = b"SPH1"
VERSION = 1
HEADER = struct.Struct("!4sBBHQQI")
MAX_PAYLOAD = 64 * 1024 * 1024


class Message(IntEnum):
    AUTH_CHALLENGE = 1
    AUTH = 2
    HELLO = 3
    READY = 4
    PULL = 5
    WORK = 6
    RESULT = 7
    STOP = 8
    WAIT = 9
    ERROR = 10


@dataclass(frozen=True)
class Frame:
    kind: Message
    request_id: int = 0
    sequence: int = 0
    payload: bytes = b""
    flags: int = 0


def json_payload(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def parse_json(payload):
    return json.loads(payload.decode("utf-8"))


def recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ConnectionError("HIVE peer closed the connection")
        chunks.extend(chunk)
    return bytes(chunks)


def send_frame(sock: socket.socket, frame: Frame) -> None:
    if len(frame.payload) > MAX_PAYLOAD:
        raise ValueError(f"HIVE payload exceeds {MAX_PAYLOAD} bytes")
    header = HEADER.pack(MAGIC, VERSION, int(frame.kind), frame.flags,
                         frame.request_id, frame.sequence, len(frame.payload))
    sock.sendall(header + frame.payload)


def recv_frame(sock: socket.socket) -> Frame:
    raw = recv_exact(sock, HEADER.size)
    magic, version, kind, flags, request_id, sequence, length = HEADER.unpack(raw)
    if magic != MAGIC:
        raise ValueError("invalid HIVE frame magic")
    if version != VERSION:
        raise ValueError(f"unsupported HIVE protocol version {version}")
    if length > MAX_PAYLOAD:
        raise ValueError(f"HIVE payload length {length} exceeds limit")
    try:
        message = Message(kind)
    except ValueError as exc:
        raise ValueError(f"unknown HIVE message type {kind}") from exc
    return Frame(message, request_id, sequence, recv_exact(sock, length), flags)
