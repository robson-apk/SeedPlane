import io
import json
import socket
import struct
import threading
import time
import unittest
import uuid
from types import SimpleNamespace
from unittest.mock import patch

from seedplane import cli
from seedplane.cluster import WorkerServer, probe
from seedplane.protocol import (DATA_HEADER, HEADER, MAGIC, MAX_CONTROL_BYTES, DataFrame, ProtocolError, ReplayGuard,
                                decode_payload, encode_data_frame, encode_message, new_message, recv_data_frame,
                                recv_message, require_context)


KEY = b"test-key-at-least-sixteen-bytes"
C = str(uuid.uuid4())
W = str(uuid.uuid4())


class ProtocolCodecTests(unittest.TestCase):
    def msg(self, **kw):
        return new_message("health", cluster_id=C, worker_id=W, **kw)

    def test_round_trip_and_canonical_mac(self):
        m = self.msg(payload={"z": 1, "a": "á"})
        self.assertEqual(recv_message(io.BytesIO(encode_message(m, KEY)), KEY), m)

    def test_tamper_is_rejected(self):
        frame = bytearray(encode_message(self.msg(payload={"ok": True}), KEY))
        frame[-4] ^= 1
        with self.assertRaises(ProtocolError):
            recv_message(io.BytesIO(frame), KEY)

    def test_bad_magic_and_oversize_rejected_before_payload_read(self):
        with self.assertRaises(ProtocolError):
            recv_message(io.BytesIO(HEADER.pack(b"NOPE", 0)), KEY)
        with self.assertRaises(ProtocolError):
            recv_message(io.BytesIO(HEADER.pack(MAGIC, MAX_CONTROL_BYTES + 1)), KEY)

    def test_expired_unknown_and_invalid_hash_rejected(self):
        with self.assertRaises(ProtocolError):
            recv_message(io.BytesIO(encode_message(self.msg(deadline_ms=1), KEY)), KEY, now_ms=2)
        m = self.msg(); m["surprise"] = 1
        with self.assertRaises(ProtocolError):
            encode_message(m, KEY)
        with self.assertRaises(ProtocolError):
            new_message("health", cluster_id=C, worker_id=W, model_hash="bad")

    def test_key_has_minimum_strength(self):
        with self.assertRaises(ProtocolError):
            encode_message(self.msg(), b"short")

    def test_wrong_model_or_plan_context_is_rejected(self):
        m = self.msg(model_hash="a" * 64, plan_hash="b" * 64)
        require_context(m, model_hash="a" * 64, plan_hash="b" * 64)
        with self.assertRaisesRegex(ProtocolError, "model hash"):
            require_context(m, model_hash="c" * 64, plan_hash="b" * 64)
        with self.assertRaisesRegex(ProtocolError, "plan hash"):
            require_context(m, model_hash="a" * 64, plan_hash="c" * 64)


class ReplayGuardTests(unittest.TestCase):
    def test_duplicate_and_stale_generation(self):
        guard = ReplayGuard()
        request_id, session_id = str(uuid.uuid4()), str(uuid.uuid4())
        m = new_message("cancel", cluster_id=C, worker_id=W, request_id=request_id,
                        session_id=session_id, generation=2)
        guard.accept(m)
        with self.assertRaisesRegex(ProtocolError, "duplicate"):
            guard.accept(m)
        stale = new_message("cancel", cluster_id=C, worker_id=W, request_id=request_id,
                            session_id=session_id, generation=1)
        with self.assertRaisesRegex(ProtocolError, "stale"):
            guard.accept(stale)


class DataFrameTests(unittest.TestCase):
    def frame(self, payload=b"\x01\x02\x03"):
        return DataFrame("tokens", 0, C, W, str(uuid.uuid4()), str(uuid.uuid4()), 3,
                         int(time.time() * 1000) + 10_000, "a" * 64, "b" * 64, payload)

    def test_binary_round_trip(self):
        frame = self.frame(bytes(range(255)))
        self.assertEqual(recv_data_frame(io.BytesIO(encode_data_frame(frame, KEY)), KEY), frame)

    def test_payload_tamper_and_wrong_key_rejected(self):
        encoded = bytearray(encode_data_frame(self.frame(), KEY)); encoded[-1] ^= 1
        with self.assertRaisesRegex(ProtocolError, "checksum"):
            recv_data_frame(io.BytesIO(encoded), KEY)
        with self.assertRaisesRegex(ProtocolError, "authentication"):
            recv_data_frame(io.BytesIO(encode_data_frame(self.frame(), KEY)), b"other-key-long-enough")

    def test_declared_oversize_rejected_before_body(self):
        raw = bytearray(encode_data_frame(self.frame(b""), KEY)[:DATA_HEADER.size])
        values = list(DATA_HEADER.unpack(raw)); values[4] = 1000
        raw = DATA_HEADER.pack(*values)
        with self.assertRaisesRegex(ProtocolError, "exceeds"):
            recv_data_frame(io.BytesIO(raw), KEY, max_bytes=10)

    def test_expired_frame_rejected(self):
        frame = self.frame(); frame = DataFrame(**{**frame.__dict__, "deadline_ms": 1})
        with self.assertRaisesRegex(ProtocolError, "expired"):
            recv_data_frame(io.BytesIO(encode_data_frame(frame, KEY)), KEY, now_ms=2)


class WorkerTests(unittest.TestCase):
    def test_dispatch_allowlist_and_cluster_identity(self):
        s = WorkerServer("127.0.0.1", 0, KEY, worker_id=W, cluster_id=C,
                         capabilities={"backends": ["test"]})
        h = s.dispatch(new_message("health", cluster_id=C, worker_id=str(uuid.uuid4())))
        self.assertEqual(h["payload"]["status"], "ready")
        with self.assertRaisesRegex(ProtocolError, "cluster"):
            s.dispatch(new_message("health", cluster_id=str(uuid.uuid4()), worker_id=W))
        with self.assertRaisesRegex(ProtocolError, "not enabled"):
            s.dispatch(new_message("load", cluster_id=C, worker_id=W,
                                   message_id=str(uuid.uuid4()), model_hash="a" * 64))

    def test_loopback_server_probe(self):
        # Reserve a port first; acceptable in a local unit test and avoids exposing LAN sockets.
        with socket.socket() as tmp:
            tmp.bind(("127.0.0.1", 0)); port = tmp.getsockname()[1]
        server = WorkerServer("127.0.0.1", port, KEY, worker_id=W, cluster_id=C,
                              capabilities={"backends": ["test"], "operations": ["health"]})
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        deadline = time.time() + 3
        while True:
            try:
                got = probe("127.0.0.1", port, KEY, C, str(uuid.uuid4()), timeout=0.2); break
            except OSError:
                if time.time() >= deadline: raise
                time.sleep(0.02)
        server.stop(); thread.join(2)
        self.assertEqual(got["hello"]["protocol"], 1)
        self.assertEqual(got["capabilities"]["backends"], ["test"])


class CliTests(unittest.TestCase):
    def test_worker_defaults_to_loopback_and_protocol_port(self):
        with patch('sys.argv', ['seedplane', 'worker']), patch.object(cli, 'cmd_worker') as worker:
            cli.main()
        self.assertEqual((worker.call_args.args[0].host, worker.call_args.args[0].port), ('127.0.0.1', 52100))

    def test_worker_external_bind_requires_cluster_key(self):
        args = SimpleNamespace(host='0.0.0.0', port=52100)
        with patch.dict('os.environ', {}, clear=True), self.assertRaises(SystemExit):
            cli.cmd_worker(args)

    def test_devices_address_validation(self):
        self.assertEqual(cli._endpoint('worker.local:52100'), ('worker.local', 52100))
        with self.assertRaises(Exception): cli._endpoint('missing-port')

    def test_doctor_has_security_warning(self):
        checks = cli._doctor_checks()
        legacy = next(c for c in checks if c['name'] == 'legacy-serve')
        self.assertEqual(legacy['status'], 'warn')


if __name__ == "__main__":
    unittest.main()
