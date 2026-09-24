"""Persistent HIVE broker and pull-based agents (v1 throughput lane)."""
from concurrent.futures import Future
from dataclasses import dataclass, field
import hashlib
import hmac
import ipaddress
import secrets
import socket
import threading
import time
from typing import Callable, Optional

from .protocol import Frame, Message, json_payload, parse_json, recv_frame, send_frame
from .cost import MeasuredCostModel


class WorkFuture:
    def __init__(self, request_id: int, future: Future, work):
        self.request_id = request_id
        self._future = future
        self._work = work

    @property
    def worker_id(self):
        return self._work.worker

    def result(self, timeout=None):
        return self._future.result(timeout)

    def done(self):
        return self._future.done()

    def add_done_callback(self, callback):
        self._future.add_done_callback(callback)


@dataclass
class _Work:
    request_id: int
    tile: "WorkTile"
    deadline: float
    future: Future
    sequence: int
    lease: int = 0
    worker: Optional[str] = None


@dataclass(frozen=True)
class WorkTile:
    """Typed unit of ready work; ranges require an executor that advertises support.

    ``dependencies`` is lineage metadata in v1. The caller must wait for those
    prerequisites before submitting; the broker does not yet resolve a DAG.
    """
    kind: str
    model: str
    payload: dict
    required_caps: frozenset[str] = frozenset()
    dependencies: tuple = ()
    sequence_range: Optional[tuple] = None
    hidden_range: Optional[tuple] = None
    token_budget: int = 0
    branch_id: int = 0
    predicted_cost: dict = field(default_factory=dict)
    deadline_s: float = 300.0
    priority: int = 0
    movable: bool = False
    stealable: bool = False

    def __post_init__(self):
        if not self.kind or not self.model or not isinstance(self.payload, dict):
            raise ValueError("WorkTile needs kind, model, and an object payload")
        if self.token_budget < 0 or self.deadline_s <= 0:
            raise ValueError("WorkTile token budget must be non-negative and deadline positive")
        for name in ("sequence_range", "hidden_range"):
            span = getattr(self, name)
            if span is not None and (len(span) != 2 or span[0] < 0 or span[1] <= span[0]):
                raise ValueError(f"invalid WorkTile {name}")
        object.__setattr__(self, "required_caps", frozenset(self.required_caps))
        object.__setattr__(self, "dependencies", tuple(self.dependencies))


class HiveBroker:
    """Coordinator-side ready queue; compatible agents pull one leased task at a time."""

    def __init__(self, host="127.0.0.1", port=0, authkey=None, pull_poll_s=0.25):
        self.host, self.port = host, int(port)
        self.authkey = authkey.encode() if isinstance(authkey, str) else authkey
        if not self._is_loopback(host) and not self.authkey:
            raise ValueError("non-loopback HIVE broker requires an explicit authkey")
        if pull_poll_s <= 0:
            raise ValueError("pull_poll_s must be positive")
        self.pull_poll_s = float(pull_poll_s)
        self.cost_model = MeasuredCostModel()
        self._lock = threading.RLock()
        self._changed = threading.Condition(self._lock)
        self._queue: list[_Work] = []
        self._workers: dict[str, dict] = {}
        self._connections: set[socket.socket] = set()
        self._threads: set[threading.Thread] = set()
        self._next_request = 1
        self._next_sequence = 1
        self._stopping = False
        self._listener = None
        self._accept_thread = None

    @staticmethod
    def _is_loopback(host):
        try:
            return ipaddress.ip_address(host).is_loopback
        except ValueError:
            return host.lower() == "localhost"

    @property
    def address(self):
        if self._listener is None:
            raise RuntimeError("broker is not started")
        return self._listener.getsockname()[:2]

    def start(self):
        if self._listener is not None:
            raise RuntimeError("broker already started")
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((self.host, self.port))
        listener.listen(64)
        listener.settimeout(0.25)
        self._listener = listener
        self.host, self.port = listener.getsockname()[:2]
        self._accept_thread = threading.Thread(target=self._accept_loop, name="hive-broker-accept", daemon=True)
        self._accept_thread.start()
        return self

    def submit(self, payload, *, model, required_caps=("generate",), priority=0, timeout=300,
               kind="throughput.generate", dependencies=(), sequence_range=None,
               hidden_range=None, token_budget=0, branch_id=0, predicted_cost=None,
               movable=False, stealable=False):
        """Submit one dependency-ready tile for compatible agents to pull."""
        tile = WorkTile(kind=kind, model=model, payload=dict(payload),
                        required_caps=frozenset(required_caps), dependencies=tuple(dependencies),
                        sequence_range=sequence_range, hidden_range=hidden_range,
                        token_budget=token_budget, branch_id=branch_id,
                        predicted_cost=dict(predicted_cost or {}), deadline_s=timeout,
                        priority=priority, movable=movable, stealable=stealable)
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        with self._changed:
            if self._stopping or self._listener is None:
                raise RuntimeError("broker is not running")
            request_id = self._next_request
            self._next_request += 1
            item = _Work(request_id, tile, time.monotonic() + timeout, Future(), request_id)
            self._queue.append(item)
            self._queue.sort(key=lambda x: (-x.tile.priority, x.sequence))
            self._changed.notify_all()
            return WorkFuture(request_id, item.future, item)

    def workers(self):
        with self._lock:
            return {name: {"caps": sorted(info["caps"]), "models": sorted(info["models"]),
                           "connected_at": info["connected_at"]} for name, info in self._workers.items()}

    def _accept_loop(self):
        while not self._stopping:
            try:
                conn, _ = self._listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            conn.settimeout(30)
            with self._lock:
                self._connections.add(conn)
            thread = threading.Thread(target=self._serve_agent, args=(conn,), daemon=True)
            with self._lock:
                self._threads.add(thread)
            thread.start()

    def _authenticate(self, conn):
        if not self.authkey:
            return
        challenge = secrets.token_bytes(32)
        send_frame(conn, Frame(Message.AUTH_CHALLENGE, payload=challenge))
        frame = recv_frame(conn)
        expected = hmac.new(self.authkey, challenge, hashlib.sha256).digest()
        if frame.kind != Message.AUTH or not hmac.compare_digest(frame.payload, expected):
            raise PermissionError("HIVE agent authentication failed")

    def _take_work(self, worker):
        poll_deadline = time.monotonic() + self.pull_poll_s
        with self._changed:
            while not self._stopping and time.monotonic() < poll_deadline:
                now = time.monotonic()
                kept = []
                for item in self._queue:
                    if item.deadline <= now:
                        if not item.future.done():
                            item.future.set_exception(TimeoutError(f"HIVE work {item.request_id} expired in queue"))
                    else:
                        kept.append(item)
                self._queue = kept
                caps, models = worker["caps"], worker["models"]
                eligible = next((i for i, item in enumerate(self._queue)
                                 if item.tile.required_caps.issubset(caps) and item.tile.model in models), None)
                if eligible is not None:
                    item = self._queue.pop(eligible)
                    item.lease = self._next_sequence
                    self._next_sequence += 1
                    item.worker = worker["id"]
                    return item
                self._changed.wait(min(self.pull_poll_s, max(0.0, poll_deadline - time.monotonic())))
            return None

    def _serve_agent(self, conn):
        worker_id = None
        active = None
        try:
            self._authenticate(conn)
            hello = recv_frame(conn)
            if hello.kind != Message.HELLO:
                raise ValueError("first authenticated HIVE message must be HELLO")
            meta = parse_json(hello.payload)
            worker_id = str(meta.get("worker_id", "")).strip()
            if not worker_id:
                raise ValueError("agent HELLO missing worker_id")
            worker = {"id": worker_id,
                      "caps": frozenset(map(str, meta.get("caps", []))),
                      "models": frozenset(map(str, meta.get("models", []))),
                      "connected_at": time.time()}
            with self._changed:
                if worker_id in self._workers:
                    raise ValueError(f"duplicate connected worker_id: {worker_id}")
                self._workers[worker_id] = worker
                self._changed.notify_all()
            conn.settimeout(None)
            send_frame(conn, Frame(Message.READY, payload=json_payload({"protocol": 1})))
            while not self._stopping:
                pull = recv_frame(conn)
                if pull.kind == Message.STOP:
                    break
                if pull.kind != Message.PULL:
                    raise ValueError(f"expected PULL, received {pull.kind.name}")
                active = self._take_work(worker)
                if active is None:
                    if self._stopping:
                        send_frame(conn, Frame(Message.STOP))
                        break
                    send_frame(conn, Frame(Message.WAIT))
                    continue
                send_frame(conn, Frame(Message.WORK, active.request_id, active.lease,
                                       json_payload({"kind": active.tile.kind,
                                                     "model": active.tile.model,
                                                     "payload": active.tile.payload,
                                                     "dependencies": active.tile.dependencies,
                                                     "sequence_range": active.tile.sequence_range,
                                                     "hidden_range": active.tile.hidden_range,
                                                     "token_budget": active.tile.token_budget,
                                                     "branch_id": active.tile.branch_id,
                                                     "predicted_cost": active.tile.predicted_cost,
                                                     "movable": active.tile.movable,
                                                     "stealable": active.tile.stealable,
                                                     "lease_ttl_s": max(0.0, active.deadline - time.monotonic())})))
                result = recv_frame(conn)
                if (result.kind != Message.RESULT or result.request_id != active.request_id
                        or result.sequence != active.lease):
                    raise ValueError("HIVE result does not match active work lease")
                data = parse_json(result.payload)
                if active.deadline <= time.monotonic():
                    if not active.future.done():
                        active.future.set_exception(TimeoutError(f"HIVE work {active.request_id} lease expired"))
                elif data.get("ok"):
                    metrics = data.get("metrics", {})
                    duration_ms = metrics.get("duration_ms")
                    if duration_ms is not None:
                        self.cost_model.observe(
                            kind=active.tile.kind, model=active.tile.model,
                            worker=worker_id, token_budget=active.tile.token_budget,
                            duration_ms=duration_ms,
                            batch_size=int(metrics.get("batch_size", 1)),
                            load=float(metrics.get("load", 0.0)))
                    if not active.future.done():
                        active.future.set_result(data.get("result"))
                elif not active.future.done():
                    active.future.set_exception(RuntimeError(str(data.get("error", "agent execution failed"))))
                active = None
        except (OSError, ConnectionError, TimeoutError, ValueError, PermissionError) as exc:
            if active is not None and not active.future.done():
                active.future.set_exception(ConnectionError(f"worker {worker_id} lost lease: {exc}"))
        finally:
            if worker_id:
                with self._changed:
                    self._workers.pop(worker_id, None)
                    self._changed.notify_all()
            with self._lock:
                self._connections.discard(conn)
            try:
                conn.close()
            except OSError:
                pass

    def close(self):
        with self._changed:
            if self._stopping:
                return
            self._stopping = True
            for item in self._queue:
                if not item.future.done():
                    item.future.set_exception(RuntimeError("HIVE broker closed before work was leased"))
            self._queue.clear()
            self._changed.notify_all()
            connections = list(self._connections)
        if self._listener:
            self._listener.close()
        for conn in connections:
            try:
                send_frame(conn, Frame(Message.STOP))
            except OSError:
                pass
        if self._accept_thread:
            self._accept_thread.join(timeout=1)
        for thread in list(self._threads):
            thread.join(timeout=1)
        for conn in connections:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            conn.close()

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()


class HiveAgent:
    """A persistent pull agent. ``execute`` owns local model/runtime affinity."""

    def __init__(self, host, port, worker_id, *, caps, models, execute: Callable,
                 authkey=None, connect_timeout=10):
        self.host, self.port = host, int(port)
        self.worker_id = str(worker_id)
        self.caps = sorted(set(map(str, caps)))
        self.models = sorted(set(map(str, models)))
        self.execute = execute
        self.authkey = authkey.encode() if isinstance(authkey, str) else authkey
        self.connect_timeout = connect_timeout
        self._stop = threading.Event()
        self.sock = None
        self.completed = 0

    def stop(self):
        self._stop.set()

    def run(self):
        self.sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        self.sock.settimeout(None)
        try:
            if self.authkey:
                challenge = recv_frame(self.sock)
                if challenge.kind != Message.AUTH_CHALLENGE:
                    raise ValueError("expected HIVE authentication challenge")
                digest = hmac.new(self.authkey, challenge.payload, hashlib.sha256).digest()
                send_frame(self.sock, Frame(Message.AUTH, payload=digest))
            send_frame(self.sock, Frame(Message.HELLO, payload=json_payload({
                "worker_id": self.worker_id, "caps": self.caps, "models": self.models,
            })))
            ready = recv_frame(self.sock)
            if ready.kind != Message.READY:
                raise RuntimeError("HIVE broker did not accept agent HELLO")
            while not self._stop.is_set():
                send_frame(self.sock, Frame(Message.PULL))
                frame = recv_frame(self.sock)
                if frame.kind == Message.STOP:
                    break
                if frame.kind == Message.WAIT:
                    continue
                if frame.kind == Message.ERROR:
                    raise RuntimeError(parse_json(frame.payload).get("error", "broker error"))
                if frame.kind != Message.WORK:
                    raise ValueError(f"unexpected broker message {frame.kind.name}")
                envelope = parse_json(frame.payload)
                try:
                    started = time.perf_counter()
                    result = self.execute(envelope["payload"])
                    duration_ms = (time.perf_counter() - started) * 1000.0
                    response = {"ok": True, "result": result,
                                "metrics": {"duration_ms": duration_ms, "batch_size": 1}}
                except Exception as exc:  # return task-local failure; keep the persistent agent alive
                    response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
                send_frame(self.sock, Frame(Message.RESULT, frame.request_id, frame.sequence,
                                            json_payload(response)))
                self.completed += 1
        except (OSError, ConnectionError):
            if not self._stop.is_set():
                raise
        finally:
            if self.sock:
                self.sock.close()
                self.sock = None
