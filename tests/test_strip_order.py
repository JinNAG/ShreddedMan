from itertools import permutations
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from strip_order import optimize_orders


class GlobalOrderTests(unittest.TestCase):
    def test_matches_exhaustive_search_including_runner_up(self):
        scores = np.random.default_rng(3).uniform(0, 1, (7, 7))
        np.fill_diagonal(scores, -np.inf)
        expected = sorted(
            ((sum(scores[a, b] for a, b in zip(p, p[1:])), p) for p in permutations(range(7))),
            reverse=True,
        )[:3]
        result = optimize_orders(scores, count=3)
        for actual, reference in zip(result, expected):
            self.assertAlmostEqual(actual[0], reference[0])
            self.assertEqual(actual[1], reference[1])

    def test_high_scoring_disconnected_cycles_are_rejected(self):
        scores = np.zeros((9, 9))
        np.fill_diagonal(scores, -np.inf)
        for i in range(8):
            scores[i, i + 1] = 0.8
        # Assignment alone would prefer three disjoint cycles.
        for a, b in ((2, 0), (5, 3), (8, 6)):
            scores[a, b] = 0.9
        ranked = optimize_orders(scores, count=2)
        for score, order in ranked:
            self.assertEqual(set(order), set(range(9)))
            self.assertEqual(len(order), 9)
            self.assertAlmostEqual(score, sum(scores[a, b] for a, b in zip(order, order[1:])))
        self.assertNotEqual(ranked[0][1], ranked[1][1])


if __name__ == "__main__":
    unittest.main()
