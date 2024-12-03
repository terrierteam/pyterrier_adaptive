import unittest
import pandas as pd
from pyterrier_adaptive import Laff


class TestLaff(unittest.TestCase):
    def test_laff(self):
        laff = Laff()
        scores = laff.compute_affinity(["hello", "world"], ["world", "hello"])
        self.assertEqual(len(scores), 2)
        self.assertTrue(isinstance(scores[0], float))
        self.assertTrue(isinstance(scores[1], float))

        inp = pd.DataFrame({
            "text": ["hello", "world"],
            "other_text": ["world", "hello"]
        })
        out = laff(inp)
        self.assertEqual(len(out), 2)
        self.assertTrue('affinity' in out.columns)


if __name__ == '__main__':
    unittest.main()
