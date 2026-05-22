import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from karush.constrained.qp import solve_eq_qp

class TestQPSingular(unittest.TestCase):
    def test_qp_singular_matrix(self):
        # Create a singular block matrix
        G = np.array([[1.0, 0.0], [0.0, 1.0]])
        c = np.array([-1.0, -1.0])
        A = np.array([[1.0, 1.0], [1.0, 1.0]]) # Singular constraints
        b = np.array([1.0, 1.0])

        # This should raise LinAlgError, not return zeros
        with self.assertRaises(np.linalg.LinAlgError):
            solve_eq_qp(G, c, A, b)

if __name__ == '__main__':
    unittest.main()
