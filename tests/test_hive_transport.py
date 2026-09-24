import socket
import threading
import time
import unittest

from seedplane.hive import HiveAgent, HiveBroker
from seedplane.hive.protocol import Frame, Message, recv_frame, send_frame
from seedplane.hive.cost import MeasuredCostModel
from seedplane.hive.scheduler import plan_homogeneous_batch


class HiveFrameTests(unittest.TestCase):
    def test_binary_frame_survives_fragmented_transport(self):
        left, right = socket.socketpair()
        try:
            frame = Frame(Message.WORK, request_id=42, sequence=9, payload=b'{"x":1}')

            def fragmented_send():
                packet = bytearray()
                from seedplane.hive.protocol import HEADER
                packet.extend(HEADER.pack(b"SPH1", 1, int(frame.kind), 0,
                                          frame.request_id, frame.sequence, len(frame.payload)))
                packet.extend(frame.payload)
                for octet in packet:
                    left.sendall(bytes([octet]))
                left.close()

            sender = threading.Thread(target=fragmented_send)
            sender.start()
            received = recv_frame(right)
            sender.join()
            self.assertEqual(received, frame)
        finally:
            right.close()


class HivePullPoolTests(unittest.TestCase):
    def test_agents_pull_only_compatible_work_over_persistent_sessions(self):
        with HiveBroker(pull_poll_s=0.03) as broker:
            draft = HiveAgent(*broker.address, "draft-box", caps=("draft",), models=("qwen-small",),
                              execute=lambda payload: {"worker": "draft-box", "value": payload["value"]})
            gpu = HiveAgent(*broker.address, "gpu-box", caps=("generate",), models=("qwen-target",),
                            execute=lambda payload: {"worker": "gpu-box", "value": payload["value"]})
            threads = [threading.Thread(target=a.run, daemon=True) for a in (draft, gpu)]
            for thread in threads:
                thread.start()
            deadline = time.monotonic() + 2
            while len(broker.workers()) < 2 and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(set(broker.workers()), {"draft-box", "gpu-box"})

            first = broker.submit({"value": 7}, model="qwen-target", required_caps=("generate",))
            second = broker.submit({"value": 11}, model="qwen-small", required_caps=("draft",))
            self.assertEqual(first.result(timeout=2), {"worker": "gpu-box", "value": 7})
            self.assertEqual(second.result(timeout=2), {"worker": "draft-box", "value": 11})
            estimate = broker.cost_model.estimate(kind="throughput.generate", model="qwen-target",
                                                  worker="gpu-box", token_budget=0)
            self.assertIsNone(estimate)  # one sample is recorded but not trusted for prediction yet
            self.assertEqual(len(broker.cost_model.snapshot()), 2)
            deadline = time.monotonic() + 1
            while (gpu.completed != 1 or draft.completed != 1) and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertEqual(gpu.completed, 1)
            self.assertEqual(draft.completed, 1)

            draft.stop()
            gpu.stop()
            for thread in threads:
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())

    def test_hmac_auth_and_external_bind_guard(self):
        with self.assertRaises(ValueError):
            HiveBroker(host="0.0.0.0")
        with HiveBroker(authkey="local-test-secret", pull_poll_s=0.03) as broker:
            agent = HiveAgent(*broker.address, "auth-agent", caps=("run",), models=("m",),
                              execute=lambda payload: payload, authkey="local-test-secret")
            thread = threading.Thread(target=agent.run, daemon=True)
            thread.start()
            deadline = time.monotonic() + 2
            while not broker.workers() and time.monotonic() < deadline:
                time.sleep(0.01)
            result = broker.submit({"ok": True}, model="m", required_caps=("run",))
            self.assertEqual(result.result(timeout=2), {"ok": True})
            agent.stop()
            thread.join(timeout=2)
            self.assertFalse(thread.is_alive())


class HiveCostModelTests(unittest.TestCase):
    def test_observations_are_keyed_and_quantiles_are_robust(self):
        model = MeasuredCostModel(min_samples=3, history=4)
        for value in (10, 12, 11, 100):
            model.observe(kind="decode", model="qwen", worker="gpu-a",
                          token_budget=16, duration_ms=value)
        estimate = model.estimate(kind="decode", model="qwen", worker="gpu-a",
                                  token_budget=16)
        self.assertEqual(estimate.samples, 4)
        self.assertEqual(estimate.p50_ms, 11.5)
        self.assertEqual(estimate.p95_ms, 100)
        self.assertIsNone(model.estimate(kind="decode", model="qwen", worker="gpu-b",
                                         token_budget=16))

    def test_adaptive_lease_chooses_work_to_amortize_fixed_cost(self):
        self.assertEqual(MeasuredCostModel.choose_lease_tokens(
            throughput_tokens_s=100, fixed_overhead_ms=20, max_fixed_fraction=0.2), 8)
        self.assertEqual(MeasuredCostModel.choose_lease_tokens(
            throughput_tokens_s=100, fixed_overhead_ms=0, min_tokens=4), 4)

    def test_adding_a_slow_worker_never_worsens_optimal_predicted_makespan(self):
        pair = plan_homogeneous_batch(16, {"B580": 120, "RX570": 270})
        trio = plan_homogeneous_batch(16, {"B580": 120, "RX570": 270, "M4": 500})
        short = plan_homogeneous_batch(4, {"B580": 10, "RX570": 20, "M4": 200})
        self.assertLessEqual(trio.predicted_makespan_ms, pair.predicted_makespan_ms)
        self.assertEqual(trio.assignments, {"B580": 10, "RX570": 4, "M4": 2})
        self.assertEqual(short.assignments["M4"], 0)


class HiveAdmissionTests(unittest.TestCase):
    def test_batch_reserves_work_by_measured_service_cost(self):
        with HiveBroker(pull_poll_s=0.02) as broker:
            agents = [HiveAgent(*broker.address, name, caps=("generate",), models=("qwen",),
                               execute=lambda payload: payload)
                      for name in ("B580", "RX570", "M4")]
            threads = [threading.Thread(target=agent.run, daemon=True) for agent in agents]
            for thread in threads:
                thread.start()
            deadline = time.monotonic() + 2
            while len(broker.workers()) != 3 and time.monotonic() < deadline:
                time.sleep(0.01)
            for name, duration in (("B580", 10), ("RX570", 20), ("M4", 200)):
                for _ in range(3):
                    broker.cost_model.observe(kind="decode", model="qwen", worker=name,
                                              token_budget=4, duration_ms=duration)
            futures = broker.submit_batch([{"n": n} for n in range(4)], model="qwen",
                                          required_caps=("generate",), kind="decode", token_budget=4)
            planned = {name: sum(future.planned_worker_id == name for future in futures)
                       for name in ("B580", "RX570", "M4")}
            self.assertEqual(planned, {"B580": 3, "RX570": 1, "M4": 0})
            self.assertEqual([future.result(timeout=2) for future in futures],
                             [{"n": n} for n in range(4)])
            for agent in agents:
                agent.stop()
            for thread in threads:
                thread.join(timeout=2)
                self.assertFalse(thread.is_alive())


if __name__ == "__main__":
    unittest.main()
