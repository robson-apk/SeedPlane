import json
import os
import sys
import tempfile
import unittest
import socket
from pathlib import Path

import numpy as np
import torch
from unittest.mock import patch

from seedplane.engine import ShardPlan
from seedplane import llama_backend
from seedplane.qwen_engine import Qwen2Engine, chat_prompt
from seedplane.planner import Device, plan_pipeline
from seedplane import native_bundle, native


class ShardPlanTests(unittest.TestCase):
    def test_windows_cover_each_core_once(self):
        plan = ShardPlan(shard=4, halo=2, sinks=1)
        windows = list(plan.windows(10))
        self.assertEqual([(a, b) for a, b, _ in windows], [(0, 4), (4, 8), (8, 10)])
        self.assertEqual(np.concatenate([np.arange(a, b) for a, b, _ in windows]).tolist(), list(range(10)))
        self.assertEqual(windows[1][2].tolist(), [0, 2, 3, 4, 5, 6, 7])

    def test_empty_sequence(self):
        self.assertEqual(list(ShardPlan().windows(0)), [])

    def test_rejects_invalid_values(self):
        for args in ((0, 1, 0), (1, -1, 0), (1, 0, -1)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ShardPlan(*args)
        with self.assertRaises(ValueError):
            list(ShardPlan().windows(-1))


class PlannerTests(unittest.TestCase):
    def test_single_device_uses_all_layers(self):
        plan = plan_pipeline([Device('cpu', 100)], n_layers=12)
        self.assertEqual(plan.layers, [12])
        self.assertEqual(plan.tensor_split, '12')

    def test_faster_device_receives_more_layers(self):
        plan = plan_pipeline([Device('slow', 100), Device('fast', 400)], n_layers=20)
        self.assertEqual(sum(plan.layers), 20)
        self.assertGreater(plan.layers[1], plan.layers[0])

    def test_rejects_invalid_inputs(self):
        with self.assertRaises(ValueError):
            plan_pipeline([], 12)
        with self.assertRaises(ValueError):
            plan_pipeline([Device('broken', 0)], 12)


class PiecePlannerTests(unittest.TestCase):
    def test_drops_marginal_remote_worker_on_short_span(self):
        pieces, _ = llama_backend.plan_pieces(4096, {'gpu': 17788, 'mac': 1237}, span=True)
        self.assertEqual(pieces['gpu'], [(0, 4096)])
        self.assertEqual(pieces['mac'], [])

    def test_keeps_remote_worker_when_gain_is_material(self):
        pieces, _ = llama_backend.plan_pieces(16384, {'gpu': 17788, 'mac': 1237}, span=True)
        self.assertTrue(pieces['mac'])

    def test_drops_three_percent_predicted_gain_inside_safety_margin(self):
        pieces, _ = llama_backend.plan_pieces(8192, {'gpu': 19606, 'mac': 1234}, span=True)
        self.assertEqual(pieces['mac'], [])

    def test_end_to_end_rate_uses_timeline_not_compute_ms(self):
        workers = [type('W', (), {'name': 'gpu'})(), type('W', (), {'name': 'mac'})()]
        win = (0, 100, np.arange(100))

        def fake_run(_workers, _ids, _work, _want, timeline=None):
            if timeline is not None:
                timeline.extend([
                    {'worker': 'gpu', 'start': 0.0, 'end': 0.01},
                    {'worker': 'gpu', 'start': 0.01, 'end': 0.02},
                    {'worker': 'mac', 'start': 0.0, 'end': 0.10},
                    {'worker': 'mac', 'start': 0.10, 'end': 0.20},
                ])
            return {}, 0.2

        with patch.object(llama_backend, 'run_pieces', side_effect=fake_run):
            rates = llama_backend.measure_rates(workers, np.arange(100), win)
        self.assertAlmostEqual(rates['gpu'], 10000)
        self.assertAlmostEqual(rates['mac'], 1000)

    def test_empty_batch_is_a_noop(self):
        worker = type('W', (), {'name': 'gpu'})()
        self.assertEqual(llama_backend.run_batch([worker], []), ([], 0.0, {'gpu': 0}))

    def test_batch_admits_slow_worker_only_above_break_even(self):
        class FakeWorker:
            def __init__(self, name):
                self.name = name
                self.s, self.peer = socket.socketpair()
            def send_window(self, ids, win, want):
                self.peer.sendall(llama_backend.RESP.pack(llama_backend.MAGIC, 0, 1, int(ids[0]), 1.0))
            def recv(self):
                return llama_backend.RESP.unpack(self.s.recv(llama_backend.RESP.size))[1:]
            def close(self):
                self.s.close(); self.peer.close()

        gpu, mac = FakeWorker('gpu'), FakeWorker('mac')
        try:
            ids = np.arange(4); job = (ids, (0, 4, np.arange(4)))
            _, _, below = llama_backend.run_batch([gpu, mac], [job] * 17, estimates={'gpu': .21, 'mac': 3.55})
            self.assertEqual(below, {'gpu': 17, 'mac': 0})
            out, _, above = llama_backend.run_batch([gpu, mac], [job] * 18, estimates={'gpu': .21, 'mac': 3.55})
            self.assertEqual(above, {'gpu': 17, 'mac': 1})
            self.assertEqual([row[2] for row in out], [0] * 18)
        finally:
            gpu.close(); mac.close()

    def test_batch_three_equal_workers_always_makes_tail_progress(self):
        class FakeWorker:
            def __init__(self, name):
                self.name = name; self.s, self.peer = socket.socketpair()
            def send_window(self, ids, win, want):
                self.peer.sendall(llama_backend.RESP.pack(llama_backend.MAGIC, 0, 1, 0, 1.0))
            def recv(self):
                return llama_backend.RESP.unpack(self.s.recv(llama_backend.RESP.size))[1:]
            def close(self):
                self.s.close(); self.peer.close()
        workers = [FakeWorker(str(i)) for i in range(3)]
        try:
            job = (np.arange(1), (0, 1, np.arange(1)))
            out, _, counts = llama_backend.run_batch(workers, [job], estimates={str(i): 1 for i in range(3)})
            self.assertIsNotNone(out[0]); self.assertEqual(sum(counts.values()), 1)
        finally:
            for worker in workers: worker.close()


class QwenEngineUtilityTests(unittest.TestCase):
    def test_greedy_sampling(self):
        self.assertEqual(Qwen2Engine.sample(torch.tensor([1.0, 3.0, 2.0])), 1)

    def test_chatml_rendering(self):
        text = chat_prompt([{'role': 'user', 'content': 'olá'}])
        self.assertEqual(text, '<|im_start|>user\nolá<|im_end|>\n<|im_start|>assistant\n')



def tiny_qwen2(directory, seed=0):
    """Random 2-layer Qwen2 with head_dim 64 in BF16 safetensors (the layout Hugging Face ships)."""
    from safetensors.torch import save_file
    L, H, NH, NKV, HD, I, V = 2, 128, 2, 1, 64, 256, 50
    (Path(directory) / 'config.json').write_text(json.dumps({
        'model_type': 'qwen2', 'num_hidden_layers': L, 'num_attention_heads': NH, 'num_key_value_heads': NKV,
        'hidden_size': H, 'intermediate_size': I, 'vocab_size': V, 'max_position_embeddings': 512,
        'rope_theta': 1e6, 'rms_norm_eps': 1e-6}))
    g = torch.Generator().manual_seed(seed); r = lambda *shape: torch.randn(*shape, generator=g) * 0.05
    w = {'model.embed_tokens.weight': r(V, H) * 20, 'model.norm.weight': torch.ones(H)}
    for l in range(L):
        p = f'model.layers.{l}.'
        w.update({p + 'input_layernorm.weight': torch.ones(H), p + 'post_attention_layernorm.weight': torch.ones(H),
                  p + 'self_attn.q_proj.weight': r(H, H), p + 'self_attn.q_proj.bias': r(H),
                  p + 'self_attn.k_proj.weight': r(NKV * HD, H), p + 'self_attn.k_proj.bias': r(NKV * HD),
                  p + 'self_attn.v_proj.weight': r(NKV * HD, H), p + 'self_attn.v_proj.bias': r(NKV * HD),
                  p + 'self_attn.o_proj.weight': r(H, H), p + 'mlp.gate_proj.weight': r(I, H),
                  p + 'mlp.up_proj.weight': r(I, H), p + 'mlp.down_proj.weight': r(H, I)})
    save_file({k: v.to(torch.bfloat16) for k, v in w.items()}, str(Path(directory) / 'model.safetensors'))
    return w


class QwenEnginePositionTests(unittest.TestCase):
    def test_explicit_positions_match_default_and_shift_changes_logits(self):
        with tempfile.TemporaryDirectory() as d:
            tiny_qwen2(d); e = Qwen2Engine(d, 'cpu'); ids = [1, 2, 3, 4]
            a, _ = e.forward(ids, e.new_cache(8), all_logits=True)
            b, _ = e.forward(ids, e.new_cache(8), all_logits=True, positions=[0, 1, 2, 3])
            c, _ = e.forward(ids, e.new_cache(8), all_logits=True, positions=[0, 40, 41, 42])
            self.assertTrue(torch.equal(a, b)); self.assertGreater(float((a - c).abs().max()), 1e-4)


class NativeBundleTests(unittest.TestCase):
    def test_convert_layout_roundtrip_and_plan(self):
        with tempfile.TemporaryDirectory() as d:
            w = tiny_qwen2(d); out = Path(d) / 'out.sp'
            m = native_bundle.convert(d, out, shard=8, halo=4, sinks=2)
            self.assertEqual(m['format'], 'seedplane-bundle/2'); self.assertEqual(m['plan'], {'shard': 8, 'halo': 4, 'sinks': 2})
            self.assertTrue(all(t['offset'] % native_bundle.ALIGN == 0 for t in m['tensors'].values()))
            blob = np.fromfile(out / 'weights.spw', dtype=np.uint8); t = m['tensors']['1.qkv_w']
            got = blob[t['offset']:t['offset'] + t['bytes']].view('<f2').reshape(t['shape']).astype(np.float32)
            p = 'model.layers.1.self_attn.'
            ref = torch.cat([w[p + 'q_proj.weight'], w[p + 'k_proj.weight'], w[p + 'v_proj.weight']]).to(torch.bfloat16).float()
            self.assertLess(float(np.abs(got - ref.numpy().astype(np.float16).astype(np.float32)).max()), 1e-12)
            self.assertEqual(json.loads((out / 'seedplane.json').read_text())['weights']['sha256'], m['weights']['sha256'])

    def test_keeps_v1_plan_and_rejects_oversized_window(self):
        with tempfile.TemporaryDirectory() as d:
            tiny_qwen2(d)
            (Path(d) / 'seedplane.json').write_text(json.dumps({'format': 'seedplane-bundle/1', 'source': 'tiny',
                                                                 'plan': {'shard': 16, 'halo': 8, 'sinks': 1}}))
            m = native_bundle.convert(d, Path(d) / 'a.sp')
            self.assertEqual((m['plan'], m['source']), ({'shard': 16, 'halo': 8, 'sinks': 1}, 'tiny'))
            with self.assertRaises(ValueError):
                native_bundle.convert(d, Path(d) / 'b.sp', shard=4000, halo=200)



FAKE_ENGINE = '''#!{python}
import json, sys
print(json.dumps({{"ready": True, "device": "fake", "mode": "shadow-batch", "plan": {{"shard": 8, "halo": 4, "sinks": 2}},
                  "positions": 64, "window_slots": 14}}), flush=True)
seq = []
for line in sys.stdin:
    q = json.loads(line)
    if q["op"] == "chat":
        if q.get("reset"): seq.clear()
        for t in ("Ol", "á", "!"):
            seq.append(t); print(json.dumps({{"token": len(seq), "text": t}}), flush=True)
        print(json.dumps({{"done": True, "reason": "stop", "generated": 3, "prompt_tokens": 5, "prefill_s": 0.0,
                          "decode_s": 0.0, "decode_tok_s": 1.0, "position": len(seq)}}), flush=True)
    elif q["op"] == "tokenize": print(json.dumps({{"ids": [ord(c) for c in q["text"]]}}), flush=True)
    else: print(json.dumps({{"error": "unknown op " + q["op"]}}), flush=True)
'''


@unittest.skipIf(os.name == 'nt', 'fake engine script needs a POSIX shebang')
class NativeFrontEndTests(unittest.TestCase):
    def test_protocol_streaming_state_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            exe = Path(d) / 'qwen_vk'; exe.write_text(FAKE_ENGINE.format(python=sys.executable)); exe.chmod(0o755)
            with native.NativeEngine(d, engine=exe) as eng:
                self.assertEqual(eng.info['plan']['shard'], 8)
                self.assertEqual(''.join(eng.chat('oi', reset=True)), 'Olá!')
                self.assertEqual((eng.last['reason'], eng.last['position']), ('stop', 3))
                self.assertEqual(''.join(eng.chat('de novo')), 'Olá!'); self.assertEqual(eng.last['position'], 6)
                self.assertEqual(eng.tokenize('ab'), [97, 98])
                with self.assertRaises(RuntimeError): eng.state()

    def test_missing_engine_is_reported(self):
        with patch.dict(os.environ, {'SEEDPLANE_NATIVE_ENGINE': ''}), patch.object(native, 'REPO_ENGINE', Path('/nonexistent/qwen_vk')), \
             patch('shutil.which', return_value=None):
            with self.assertRaises(FileNotFoundError): native.find_engine()


if __name__ == '__main__':
    unittest.main()
