import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from karush.convex.relaxations import max_cut_sdp_relaxation

class TestMaxCutOOMDos(unittest.TestCase):
    def test_max_cut_oom(self):
        # Create a large matrix to test memory exhaustion in list construction
        W = np.eye(10001)

        with self.assertRaisesRegex(ValueError, "System dimensions exceed safe limit for memory allocation."):
            max_cut_sdp_relaxation(W)

if __name__ == '__main__':
    unittest.main()
