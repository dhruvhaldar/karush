import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from karush.semidefinite.interior_point import svec, smat
from karush.convex.relaxations import randomized_rounding

class TestShapeDoS(unittest.TestCase):
    def test_svec_geometric_shape(self):
        M = np.ones((5, 3))
        with self.assertRaises(ValueError):
            svec(M)

    def test_smat_geometric_shape(self):
        v = np.ones(3) # Not a valid length for any n*(n+1)/2 (n=1->1, n=2->3, n=3->6)
        with self.assertRaises(ValueError):
            smat(v, 5) # For n=5, we need length 15

    def test_randomized_rounding_geometric_shape(self):
        X = np.ones((5, 3))
        with self.assertRaises(ValueError):
            randomized_rounding(X)

if __name__ == '__main__':
    unittest.main()
