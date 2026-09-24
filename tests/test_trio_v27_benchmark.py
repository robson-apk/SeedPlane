import shlex
import sys
import time
import unittest

from experiments.trio_v27.benchmark_trio import Worker, parse_identity, parse_worker, percentile, run_workload


class FakeWorker:
    def __init__(self, name):
        self.name = name
        self.bytes_sent = 0
        self.bytes_received = 0

    def generate(self, request_id, arrival_rel_s, origin):
        now = time.perf_counter()
        return {
            "request_id": request_id,
            "worker": self.name,
            "arrival_s": arrival_rel_s,
            "dispatch_s": now - origin,
            "first_token_s": now - origin,
            "completion_s": now - origin,
            "queue_wait_s": max(0.0, now - origin - arrival_rel_s),
            "ttft_from_arrival_s": max(0.0, now - origin - arrival_rel_s),
            "completion_from_arrival_s": max(0.0, now - origin - arrival_rel_s),
            "service_roundtrip_s": 0.001,
            "generated": 128,
            "runtime_decode_tok_s": 128000.0,
            "non_decode_remainder_s": 0.0,
            "token_sha256": "same-greedy-output",
        }


class TrioBenchmarkTests(unittest.TestCase):
    def test_nearest_rank_p99_is_max_for_24_requests(self):
        self.assertEqual(percentile(list(range(1, 25)), .99), 24)

    def test_worker_specs_parse(self):
        self.assertEqual(parse_worker("M4|local|/tmp/qwen_vk /tmp/model --serve"),
                         ("M4", "local", "", "/tmp/qwen_vk /tmp/model --serve"))
        self.assertEqual(parse_worker("RX570|ssh|robson@host|~/qwen_vk ~/model --serve"),
                         ("RX570", "ssh", "robson@host", "~/qwen_vk ~/model --serve"))

    def test_identity_captures_five_artifact_hashes(self):
        identity = parse_identity("M4|runtime|bundle|weights|tokenizer|plan")
        self.assertEqual(identity[0], "M4")
        self.assertEqual(identity[1]["tokenizer_sha256"], "tokenizer")

    def test_burst_scheduler_accounts_for_every_request_and_worker(self):
        workers = [FakeWorker("B580"), FakeWorker("RX570"), FakeWorker("M4")]
        result = run_workload(workers, arrival_interval=0.0, round_index=0, condition_index=0)
        self.assertEqual(result["requests"], 24)
        self.assertEqual(result["generated_tokens"], 24 * 128)
        self.assertEqual(sum(result["assignments"].values()), 24)
        self.assertEqual(set(result["assignments"]), {"B580", "RX570", "M4"})
        self.assertTrue(all(count > 0 for count in result["assignments"].values()))
        self.assertTrue(result["all_request_outputs_identical_within_condition"])

    def test_local_jsonl_worker_stream_smoke(self):
        mock_server = (
            "import json,sys\n"
            "print(json.dumps({'ready':True,'device':'mock'}),flush=True)\n"
            "for line in sys.stdin:\n"
            " request=json.loads(line)\n"
            " for token in range(request['max_new_tokens']):\n"
            "  print(json.dumps({'token':token}),flush=True)\n"
            " print(json.dumps({'done':True,'generated':request['max_new_tokens'],'decode_tok_s':128.0}),flush=True)\n"
        )
        command = f"{shlex.quote(sys.executable)} -u -c {shlex.quote(mock_server)}"
        worker = Worker("mock", "local", "", command)
        try:
            origin = time.perf_counter()
            result = worker.generate(0, 0.0, origin)
            self.assertEqual(result["generated"], 128)
            self.assertEqual(len(result["token_sha256"]), 64)
            self.assertGreater(worker.bytes_sent, 0)
            self.assertGreater(worker.bytes_received, 0)
        finally:
            worker.close()


if __name__ == "__main__":
    unittest.main()
