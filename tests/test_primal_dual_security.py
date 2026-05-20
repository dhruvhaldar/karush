import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from karush.constrained.primal_dual import primal_dual_qp

class TestPrimalDualSecurity(unittest.TestCase):
    def setUp(self):
        self.G = np.eye(2)
        self.c = np.array([-1, -1])
        self.A = np.array([[1, 1]])
        self.b = np.array([1])

    def test_primal_dual_x0_zero(self):
        # x0 contains a zero value, should raise ValueError to prevent divide by zero
        x0_bad = np.array([0.0, 1.0])
        z0 = np.array([1.0, 1.0])

        with self.assertRaisesRegex(ValueError, "strictly positive"):
            primal_dual_qp(self.G, self.c, self.A, self.b, x0_bad, z0)

    def test_primal_dual_z0_zero(self):
        # z0 contains a zero value, should raise ValueError to prevent divide by zero
        x0 = np.array([1.0, 1.0])
        z0_bad = np.array([1.0, 0.0])

        with self.assertRaisesRegex(ValueError, "strictly positive"):
            primal_dual_qp(self.G, self.c, self.A, self.b, x0, z0_bad)

    def test_primal_dual_x0_negative(self):
        x0_bad = np.array([-1.0, 1.0])
        z0 = np.array([1.0, 1.0])

        with self.assertRaisesRegex(ValueError, "strictly positive"):
            primal_dual_qp(self.G, self.c, self.A, self.b, x0_bad, z0)


if __name__ == '__main__':
    unittest.main()
