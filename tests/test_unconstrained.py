import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from karush.unconstrained.newton import newton_method
from karush.unconstrained.quasi_newton import bfgs_method
from karush.unconstrained.conjugate_gradient import conjugate_gradient

class TestUnconstrained(unittest.TestCase):
    def setUp(self):
        # Quadratic function: f(x) = (x[0]-1)^2 + (x[1]-2)^2
        self.f = lambda x: (x[0] - 1)**2 + (x[1] - 2)**2
        self.grad_f = lambda x: np.array([2 * (x[0] - 1), 2 * (x[1] - 2)])
        self.hess_f = lambda x: np.array([[2, 0], [0, 2]])
        self.x0 = [0.0, 0.0]
        self.solution = np.array([1.0, 2.0])

    def test_newton(self):
        x, _ = newton_method(self.f, self.grad_f, self.hess_f, self.x0)
        np.testing.assert_allclose(x, self.solution, atol=1e-5)

    def test_bfgs(self):
        x, _ = bfgs_method(self.f, self.grad_f, self.x0)
        np.testing.assert_allclose(x, self.solution, atol=1e-5)

    def test_conjugate_gradient(self):
        x, _ = conjugate_gradient(self.f, self.grad_f, self.x0)
        np.testing.assert_allclose(x, self.solution, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
