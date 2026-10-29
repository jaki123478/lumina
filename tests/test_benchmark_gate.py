import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark_gate import accept_candidate


class BenchmarkGateTests(unittest.TestCase):
    def test_rejects_regression(self):
        self.assertFalse(accept_candidate(14, 13))

    def test_allows_non_regression_by_default(self):
        self.assertTrue(accept_candidate(14, 14))
        self.assertTrue(accept_candidate(14, 15))

    def test_strict_mode_requires_real_gain(self):
        self.assertFalse(accept_candidate(14, 14, True))
        self.assertTrue(accept_candidate(14, 15, True))


if __name__ == '__main__':
    unittest.main()
