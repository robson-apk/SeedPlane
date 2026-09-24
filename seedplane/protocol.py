"""SeedPlane cluster protocol v1.

The legacy research worker uses ``multiprocessing.connection`` and therefore
pickle.  This module is the safe control-plane replacement: length-prefixed,
bounded canonical JSON authenticated with HMAC-SHA256.  It intentionally does
not deserialize Python objects or accept commands outside a small allowlist.

TLS/Noise and pairing are transport concerns planned for V25; v1 refuses a
non-loopback bind unless the operator provides a key explicitly.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import socket
import struct
import time
import uuid
from dataclasses import dataclass
from typing import Any, BinaryIO, Mapping


MAGIC = b"SPV1"
VERSION = 1
HEADER = struct.Struct("!4sI")
MAX_CONTROL_BYTES = 1 << 20  # 1 MiB, checked before allocation
MAX_CLOCK_SKEW_MS = 5 * 60 * 1000
HASH_HEX_LEN = 64
MESSAGE_TYPES = frozenset({
    "hello", "capabilities", "health", "heartbeat", "model_status",
    "load", "unload", "benchmark", "cancel", "error", "ok",
})
REQUIRED = frozenset({
    "version", "type", "message_id", "cluster_id", "worker_id",
    "request_id", "session_id", "generation", "deadline_ms", "payload",
})


class ProtocolError(ValueError):
    """Peer sent an invalid or unauthenticated protocol message."""


def _canonical(value: Mapping[str, Any]) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                          allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"message is not canonical JSON: {exc}") from exc


def _uuid(value: Any, field: str, allow_empty: bool = False) -> str:
    if allow_empty and value == "":
        return ""
    if not isinstance(value, str):
        raise ProtocolError(f"{field} must be a UUID string")
    try:
        return str(uuid.UUID(value))
    except (ValueError, AttributeError) as exc:
        raise ProtocolError(f"invalid {field}") from exc


def _hash(value: Any, field: str) -> str:
    if value in (None, ""):
        return ""
    if not isinstance(value, str) or len(value) != HASH_HEX_LEN:
        raise ProtocolError(f"{field} must be an empty value or SHA-256 hex")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ProtocolError(f"invalid {field}") from exc
    return value.lower()


def new_message(message_type: str, *, cluster_id: str, worker_id: str,
                request_id: str = "", session_id: str = "", generation: int = 0,
                model_hash: str = "", plan_hash: str = "", deadline_ms: int | None = None,
                payload: Mapping[str, Any] | None = None, message_id: str | None = None) -> dict[str, Any]:
    """Create and validate a v1 message with safe defaults."""
    now = int(time.time() * 1000)
    msg = {
        "version": VERSION,
        "type": message_type,
        "message_id": message_id or str(uuid.uuid4()),
        "cluster_id": cluster_id,
        "worker_id": worker_id,
        "request_id": request_id,
        "session_id": session_id,
        "generation": generation,
        "model_hash": model_hash,
        "plan_hash": plan_hash,
        "sent_ms": now,
        "deadline_ms": deadline_ms if deadline_ms is not None else now + 30_000,
        "payload": dict(payload or {}),
    }
    return validate_message(msg, now_ms=now, check_deadline=False)


def validate_message(message: Any, *, now_ms: int | None = None, check_deadline: bool = True) -> dict[str, Any]:
    if not isinstance(message, dict):
        raise ProtocolError("message must be an object")
    missing = REQUIRED - message.keys()
    if missing:
        raise ProtocolError("missing fields: " + ", ".join(sorted(missing)))
    unknown = set(message) - (REQUIRED | {"model_hash", "plan_hash", "sent_ms"})
    if unknown:
        raise ProtocolError("unknown fields: " + ", ".join(sorted(unknown)))
    if message["version"] != VERSION:
        raise ProtocolError(f"unsupported protocol version {message['version']!r}")
    if message["type"] not in MESSAGE_TYPES:
        raise ProtocolError(f"unsupported message type {message['type']!r}")
    out = dict(message)
    out["message_id"] = _uuid(message["message_id"], "message_id")
    out["cluster_id"] = _uuid(message["cluster_id"], "cluster_id")
    out["worker_id"] = _uuid(message["worker_id"], "worker_id")
    out["request_id"] = _uuid(message["request_id"], "request_id", allow_empty=True)
    out["session_id"] = _uuid(message["session_id"], "session_id", allow_empty=True)
    if isinstance(message["generation"], bool) or not isinstance(message["generation"], int) or message["generation"] < 0:
        raise ProtocolError("generation must be a non-negative integer")
    if not isinstance(message["deadline_ms"], int) or isinstance(message["deadline_ms"], bool):
        raise ProtocolError("deadline_ms must be an integer")
    if "sent_ms" in message and (not isinstance(message["sent_ms"], int) or isinstance(message["sent_ms"], bool)):
        raise ProtocolError("sent_ms must be an integer")
    out["model_hash"] = _hash(message.get("model_hash", ""), "model_hash")
    out["plan_hash"] = _hash(message.get("plan_hash", ""), "plan_hash")
    if not isinstance(message["payload"], dict):
        raise ProtocolError("payload must be an object")
    if len(_canonical(message["payload"])) > MAX_CONTROL_BYTES // 2:
        raise ProtocolError("payload exceeds control-plane limit")
    now = int(time.time() * 1000) if now_ms is None else now_ms
    if check_deadline and message["deadline_ms"] < now:
        raise ProtocolError("message deadline expired")
    if message.get("sent_ms", now) > now + MAX_CLOCK_SKEW_MS:
        raise ProtocolError("message timestamp too far in the future")
    return out


def encode_message(message: Mapping[str, Any], key: bytes) -> bytes:
    if not isinstance(key, bytes) or len(key) < 16:
        raise ProtocolError("authentication key must contain at least 16 bytes")
    msg = validate_message(dict(message), check_deadline=False)
    body = _canonical(msg)
    signed = {"message": msg, "mac": hmac.new(key, body, hashlib.sha256).hexdigest()}
    payload = _canonical(signed)
    if len(payload) > MAX_CONTROL_BYTES:
        raise ProtocolError("encoded frame exceeds control-plane limit")
    return HEADER.pack(MAGIC, len(payload)) + payload


def decode_payload(payload: bytes, key: bytes, *, now_ms: int | None = None) -> dict[str, Any]:
    if len(payload) > MAX_CONTROL_BYTES:
        raise ProtocolError("frame exceeds control-plane limit")
    try:
        envelope = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON frame") from exc
    if not isinstance(envelope, dict) or set(envelope) != {"message", "mac"}:
        raise ProtocolError("invalid signed envelope")
    if not isinstance(envelope["mac"], str) or len(envelope["mac"]) != HASH_HEX_LEN:
        raise ProtocolError("invalid message authentication code")
    body = _canonical(envelope["message"])
    expected = hmac.new(key, body, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(envelope["mac"], expected):
        raise ProtocolError("message authentication failed")
    return validate_message(envelope["message"], now_ms=now_ms)


def recv_exact(stream: BinaryIO | socket.socket, size: int) -> bytes:
    chunks: list[bytes] = []
    left = size
    while left:
        chunk = stream.recv(left) if hasattr(stream, "recv") else stream.read(left)
        if not chunk:
            raise EOFError("connection closed inside frame")
        chunks.append(chunk)
        left -= len(chunk)
    return b"".join(chunks)


def recv_message(stream: BinaryIO | socket.socket, key: bytes, *, now_ms: int | None = None) -> dict[str, Any]:
    header = recv_exact(stream, HEADER.size)
    magic, size = HEADER.unpack(header)
    if magic != MAGIC:
        raise ProtocolError("invalid frame magic")
    if size > MAX_CONTROL_BYTES:
        raise ProtocolError("declared frame exceeds control-plane limit")
    return decode_payload(recv_exact(stream, size), key, now_ms=now_ms)


def send_message(stream: BinaryIO | socket.socket, message: Mapping[str, Any], key: bytes) -> None:
    frame = encode_message(message, key)
    if hasattr(stream, "sendall"):
        stream.sendall(frame)
    else:
        stream.write(frame)
        if hasattr(stream, "flush"):
            stream.flush()


def require_context(message: Mapping[str, Any], *, model_hash: str, plan_hash: str) -> None:
    """Reject execution against a model or plan different from the loaded context."""
    expected_model = _hash(model_hash, "expected model_hash")
    expected_plan = _hash(plan_hash, "expected plan_hash")
    if message.get("model_hash", "") != expected_model:
        raise ProtocolError("model hash mismatch")
    if message.get("plan_hash", "") != expected_plan:
        raise ProtocolError("plan hash mismatch")


@dataclass
class ReplayGuard:
    """Reject duplicate messages and stale request generations with bounded state."""

    max_messages: int = 4096

    def __post_init__(self) -> None:
        self._seen: dict[str, None] = {}
        self._generations: dict[tuple[str, str], int] = {}

    def accept(self, message: Mapping[str, Any]) -> None:
        mid = str(message["message_id"])
        if mid in self._seen:
            raise ProtocolError("duplicate message")
        request = str(message.get("request_id", ""))
        session = str(message.get("session_id", ""))
        generation = int(message.get("generation", 0))
        if request:
            key = (session, request)
            latest = self._generations.get(key, generation)
            if generation < latest:
                raise ProtocolError("stale request generation")
            self._generations[key] = generation
        self._seen[mid] = None
        while len(self._seen) > self.max_messages:
            self._seen.pop(next(iter(self._seen)))
