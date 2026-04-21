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
        self.grad_f_list = lambda x: [2 * (x[0] - 1), 2 * (x[1] - 2)]
        self.hess_f_list = lambda x: [[2, 0], [0, 2]]

    def test_newton(self):
        x, _ = newton_method(self.f, self.grad_f, self.hess_f, self.x0)
        np.testing.assert_allclose(x, self.solution, atol=1e-5)

    def test_bfgs(self):
        x, _ = bfgs_method(self.f, self.grad_f, self.x0)
        np.testing.assert_allclose(x, self.solution, atol=1e-5)

    def test_conjugate_gradient(self):
        x, _ = conjugate_gradient(self.f, self.grad_f, self.x0)
        np.testing.assert_allclose(x, self.solution, atol=1e-5)

    def test_bfgs_validation(self):
        with self.assertRaises(ValueError):
            bfgs_method(self.f, self.grad_f, [np.nan, 0.0])

        with self.assertRaises(ValueError):
            bfgs_method(self.f, self.grad_f, [0.0, np.inf])

    def test_max_iter_validation(self):
        with self.assertRaises(ValueError):
            bfgs_method(self.f, self.grad_f, self.x0, max_iter=10001)

    def test_newton_list_input(self):
        x, _ = newton_method(self.f, self.grad_f_list, self.hess_f_list, self.x0)
        np.testing.assert_allclose(x, self.solution, atol=1e-5)

    def test_bfgs_list_input(self):
        x, _ = bfgs_method(self.f, self.grad_f_list, self.x0)
        np.testing.assert_allclose(x, self.solution, atol=1e-5)

    def test_conjugate_gradient_list_input(self):
        x, _ = conjugate_gradient(self.f, self.grad_f_list, self.x0)
        np.testing.assert_allclose(x, self.solution, atol=1e-5)

    def test_multiple_element_list_input(self):
        def f_multi(x):
            return [(x[0] - 1)**2, (x[1] - 2)**2]

        def grad_f_multi(x):
            return [2 * (x[0] - 1), 2 * (x[1] - 2)]

        def hess_f_multi(x):
            return [[2, 0], [0, 2]]

        x_newton, _ = newton_method(f_multi, grad_f_multi, hess_f_multi, self.x0)
        np.testing.assert_allclose(x_newton, self.solution, atol=1e-5)

        x_bfgs, _ = bfgs_method(f_multi, grad_f_multi, self.x0)
        np.testing.assert_allclose(x_bfgs, self.solution, atol=1e-5)

        x_cg, _ = conjugate_gradient(f_multi, grad_f_multi, self.x0)
        np.testing.assert_allclose(x_cg, self.solution, atol=1e-5)

if __name__ == '__main__':
    unittest.main()
