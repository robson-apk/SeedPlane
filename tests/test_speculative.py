import unittest

from seedplane.speculative import CommitLedger, SpeculativeForest, verify_greedy


class SpeculativeForestTests(unittest.TestCase):
    def target(self, prefix):
        # Deterministic target sequence: token 1, then 2, then 3, ...
        expected = min(len(prefix), 5)
        logits = [0.0] * 8
        logits[expected] = 1.0
        return logits

    def test_forest_deduplicates_shared_prefix(self):
        forest = SpeculativeForest.build([[1, 2, 3], [1, 2, 4], [1, 5]])
        self.assertEqual(forest.node_count, 5)
        self.assertEqual(len(forest.paths), 3)

    def test_verifier_accepts_longest_exact_branch_and_corrects_rejection(self):
        forest = SpeculativeForest.build([[1, 2, 7], [1, 2, 3, 4]])
        result = verify_greedy([0], forest, self.target)
        self.assertEqual(result.branch, (1, 2, 3, 4))
        self.assertEqual(result.emitted, (1, 2, 3, 4))
        self.assertEqual(result.accepted_draft_tokens, 4)
        self.assertFalse(result.rejected)

        rejected = verify_greedy([0], SpeculativeForest.build([[1, 2, 7]]), self.target)
        self.assertEqual(rejected.emitted, (1, 2, 3))
        self.assertEqual(rejected.accepted_draft_tokens, 2)
        self.assertTrue(rejected.rejected)

    def test_commit_ledger_only_advances_on_commit(self):
        ledger = CommitLedger([0])
        result = verify_greedy(ledger.tokens, SpeculativeForest.build([[1, 2]]), self.target)
        self.assertEqual(ledger.frontier, 1)
        ledger.commit(result)
        self.assertEqual(ledger.tokens, [0, 1, 2])
        self.assertEqual(ledger.frontier, 3)
        self.assertEqual(ledger.commits, 1)

    def test_empty_forest_falls_back_to_target(self):
        result = verify_greedy([0], SpeculativeForest.build([]), self.target)
        self.assertEqual(result.emitted, (1,))
        self.assertEqual(result.target_steps, 1)


if __name__ == "__main__":
    unittest.main()
