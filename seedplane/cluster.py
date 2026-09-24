"""Small safe SeedPlane v1 control-plane server/client."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .protocol import ProtocolError, ReplayGuard, new_message, recv_message, send_message


DEFAULT_PORT = 52100
ENV_KEY = "SEEDPLANE_CLUSTER_KEY"


class DeviceRegistry:
    """Small non-secret device catalog; authentication keys never enter it."""

    def __init__(self, path: Path):
        self.path = path

    def load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ProtocolError(f"invalid device registry {self.path}: {exc}") from exc
        if not isinstance(raw, dict) or raw.get("version") != 1 or not isinstance(raw.get("devices"), dict):
            raise ProtocolError("invalid device registry schema")
        out: dict[str, dict[str, Any]] = {}
        for name, record in raw["devices"].items():
            if not isinstance(name, str) or not name or not isinstance(record, dict):
                raise ProtocolError("invalid device registry entry")
            host, port, worker_id = record.get("host"), record.get("port"), record.get("worker_id")
            if not isinstance(host, str) or not isinstance(port, int) or not 1 <= port <= 65535:
                raise ProtocolError(f"invalid endpoint for device {name!r}")
            if worker_id:
                worker_id = str(uuid.UUID(worker_id))
            out[name] = {"host": host, "port": port, "worker_id": worker_id or ""}
        return out

    def save(self, devices: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps({"version": 1, "devices": devices}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        try: tmp.chmod(0o600)
        except OSError: pass
        tmp.replace(self.path)

    def add(self, name: str, host: str, port: int, worker_id: str) -> None:
        if not name or any(ch not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for ch in name):
            raise ProtocolError("device name may contain only letters, digits, '-' and '_'")
        worker_id = str(uuid.UUID(worker_id))
        devices = self.load()
        if name in devices and devices[name] != {"host": host, "port": port, "worker_id": worker_id}:
            raise ProtocolError(f"device {name!r} already exists; forget it before replacing")
        for other, record in devices.items():
            if other != name and record["worker_id"] == worker_id:
                raise ProtocolError(f"worker identity already registered as {other!r}")
        devices[name] = {"host": host, "port": port, "worker_id": worker_id}
        self.save(devices)

    def forget(self, name: str) -> None:
        devices = self.load()
        if name not in devices:
            raise ProtocolError(f"unknown device {name!r}")
        del devices[name]
        self.save(devices)


def cluster_key(value: str | None = None) -> bytes:
    raw = value if value is not None else os.environ.get(ENV_KEY, "")
    if len(raw) < 16:
        raise ProtocolError(f"set {ENV_KEY} to at least 16 characters")
    return raw.encode("utf-8")


def stable_id(path: Path, name: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return str(uuid.UUID(path.read_text(encoding="ascii").strip()))
    value = str(uuid.uuid4())
    path.write_text(value + "\n", encoding="ascii")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return value


def default_capabilities() -> dict[str, Any]:
    caps: dict[str, Any] = {
        "hostname": socket.gethostname(),
        "system": platform.system(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "operations": ["hello", "capabilities", "health", "heartbeat", "model_status", "cancel"],
        "backends": [],
    }
    try:
        import torch
        caps["torch"] = torch.__version__
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            caps["backends"].append("mps")
        if torch.cuda.is_available():
            caps["backends"].append("cuda")
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            caps["backends"].append("xpu")
        caps["backends"].append("cpu")
    except Exception as exc:  # doctor reports absence without preventing control-plane startup
        caps["torch_error"] = type(exc).__name__
        caps["backends"].append("cpu")
    return caps


@dataclass
class WorkerState:
    worker_id: str
    cluster_id: str
    started: float = field(default_factory=time.monotonic)
    models: dict[str, str] = field(default_factory=dict)
    cancelled: set[tuple[str, int]] = field(default_factory=set)


class WorkerServer:
    def __init__(self, host: str, port: int, key: bytes, *, worker_id: str, cluster_id: str,
                 capabilities: dict[str, Any] | None = None):
        self.host, self.port, self.key = host, port, key
        self.state = WorkerState(worker_id=worker_id, cluster_id=cluster_id)
        self.capabilities = capabilities or default_capabilities()
        self.replay = ReplayGuard()
        self._stop = threading.Event()

    def response(self, request: dict[str, Any], kind: str, payload: dict[str, Any]) -> dict[str, Any]:
        return new_message(kind, cluster_id=self.state.cluster_id, worker_id=self.state.worker_id,
                           request_id=request.get("request_id", ""), session_id=request.get("session_id", ""),
                           generation=request.get("generation", 0), model_hash=request.get("model_hash", ""),
                           plan_hash=request.get("plan_hash", ""), payload=payload)

    def dispatch(self, message: dict[str, Any]) -> dict[str, Any]:
        self.replay.accept(message)
        if message["cluster_id"] != self.state.cluster_id:
            raise ProtocolError("cluster identity mismatch")
        kind = message["type"]
        if kind == "hello":
            return self.response(message, "ok", {"protocol": 1, "worker_id": self.state.worker_id})
        if kind == "capabilities":
            return self.response(message, "ok", self.capabilities)
        if kind in {"health", "heartbeat"}:
            return self.response(message, "ok", {"status": "ready", "uptime_s": round(time.monotonic() - self.state.started, 3)})
        if kind == "model_status":
            h = message.get("model_hash", "")
            return self.response(message, "ok", {"model_hash": h, "status": self.state.models.get(h, "missing")})
        if kind == "cancel":
            request = message.get("request_id", "")
            if not request:
                raise ProtocolError("cancel requires request_id")
            self.state.cancelled.add((request, int(message["generation"])))
            return self.response(message, "ok", {"cancelled": True})
        raise ProtocolError(f"operation {kind!r} is not enabled by this worker")

    def handle(self, conn: socket.socket) -> None:
        conn.settimeout(10)
        try:
            request = recv_message(conn, self.key)
            response = self.dispatch(request)
        except (ProtocolError, EOFError, OSError) as exc:
            # Authentication failures intentionally receive no detailed response.
            if isinstance(exc, ProtocolError) and "authentication" in str(exc):
                return
            try:
                response = new_message("error", cluster_id=self.state.cluster_id, worker_id=self.state.worker_id,
                                       payload={"error": str(exc)[:300]})
            except Exception:
                return
        try:
            send_message(conn, response, self.key)
        except OSError:
            pass

    def serve_forever(self) -> None:
        with socket.socket(socket.AF_INET6 if ":" in self.host else socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind((self.host, self.port)); sock.listen(32); sock.settimeout(0.5)
            actual = sock.getsockname()[1]
            print(json.dumps({"worker": self.state.worker_id, "cluster": self.state.cluster_id,
                              "listen": f"{self.host}:{actual}", "protocol": 1}), flush=True)
            while not self._stop.is_set():
                try:
                    conn, _ = sock.accept()
                except socket.timeout:
                    continue
                with conn:
                    self.handle(conn)

    def stop(self) -> None:
        self._stop.set()


def request(host: str, port: int, key: bytes, message: dict[str, Any], timeout: float = 5.0) -> dict[str, Any]:
    with socket.create_connection((host, port), timeout=timeout) as conn:
        conn.settimeout(timeout)
        send_message(conn, message, key)
        return recv_message(conn, key)


def probe(host: str, port: int, key: bytes, cluster_id: str, coordinator_id: str,
          timeout: float = 5.0) -> dict[str, Any]:
    hello = new_message("hello", cluster_id=cluster_id, worker_id=coordinator_id,
                        payload={"role": "coordinator"})
    h = request(host, port, key, hello, timeout)
    caps = new_message("capabilities", cluster_id=cluster_id, worker_id=coordinator_id)
    c = request(host, port, key, caps, timeout)
    return {"hello": h["payload"], "capabilities": c["payload"]}
