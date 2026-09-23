import unittest

import numpy as np

from seedplane.engine import ShardPlan
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


if __name__ == '__main__':
    unittest.main()
