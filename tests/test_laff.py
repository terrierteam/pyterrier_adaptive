import unittest
import pandas as pd
from pyterrier_adaptive import Laff


class TestLaff(unittest.TestCase):
    def test_laff(self):
        laff = Laff()
        scores = laff.compute_affinity_scores("hello", "world")
        self.assertEqual(len(scores), 1)

        scores = laff.compute_affinity_scores(["hello", "world"], ["world", "hello"])
        self.assertEqual(len(scores), 2)

        inp = pd.DataFrame({
            "text": ["hello", "world"],
            "other_text": ["world", "hello"]
        })
        out = laff(inp)
        self.assertEqual(len(out), 2)
        self.assertTrue('affinity_score' in out.columns)


if __name__ == '__main__':
    unittest.main()
