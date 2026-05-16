import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from karush.convex.relaxations import randomized_rounding, max_cut_sdp_relaxation

class TestConvex(unittest.TestCase):
    def test_randomized_rounding_list_input(self):
        X = [[1.0, 0.0], [0.0, 1.0]]
        candidates = randomized_rounding(X, num_trials=10)
        self.assertEqual(len(candidates), 10)
        self.assertEqual(len(candidates[0]), 2)

    def test_randomized_rounding_dos_prevention(self):
        X = np.eye(10)
        with self.assertRaises(ValueError):
            randomized_rounding(X, num_trials=100001)

    def test_max_cut_sdp_relaxation_shape_validation(self):
        W = np.zeros((10, 1))
        with self.assertRaises(ValueError):
            max_cut_sdp_relaxation(W)

if __name__ == '__main__':
    unittest.main()
