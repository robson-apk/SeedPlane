import unittest

import numpy as np
from unittest.mock import patch

from seedplane.engine import ShardPlan
from seedplane import llama_backend
from seedplane.planner import Device, plan_pipeline


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


if __name__ == '__main__':
    unittest.main()
