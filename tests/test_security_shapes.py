import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from karush.unconstrained.quasi_newton import bfgs_method
from karush.unconstrained.conjugate_gradient import conjugate_gradient
from karush.constrained.sqp import sqp_equality_constrained

class TestSecurityShapes(unittest.TestCase):
    def test_bfgs_shape_validation(self):
        def f(x): return x[0]**2 + x[1]**2
        def grad_f_bad(x): return np.array([2*x[0], 2*x[1], 0])

        with self.assertRaises(ValueError):
            bfgs_method(f, grad_f_bad, [1.0, 1.0])

    def test_cg_shape_validation(self):
        def f(x): return x[0]**2 + x[1]**2
        def grad_f_bad(x): return np.array([2*x[0], 2*x[1], 0])

        with self.assertRaises(ValueError):
            conjugate_gradient(f, grad_f_bad, [1.0, 1.0])

    def test_sqp_shape_validation(self):
        def f(x): return x[0]**2 + x[1]**2
        def grad_f(x): return np.array([2*x[0], 2*x[1]])
        def grad_f_bad(x): return np.array([2*x[0], 2*x[1], 0])
        def hess_f(x): return np.array([[2, 0], [0, 2]])
        def hess_f_bad(x): return np.array([[2, 0, 0], [0, 2, 0]])
        def h(x): return np.array([x[0] + x[1] - 1])
        def grad_h(x): return np.array([1, 1])

        # Test gradient shape mismatch
        with self.assertRaises(ValueError):
            sqp_equality_constrained(f, grad_f_bad, hess_f, h, grad_h, [1.0, 1.0])

        # Test Hessian shape mismatch
        with self.assertRaises(ValueError):
            sqp_equality_constrained(f, grad_f, hess_f_bad, h, grad_h, [1.0, 1.0])

    def test_sqp_shape_validation_new(self):
        def f(x): return x[0]**2 + x[1]**2
        def grad_f(x): return np.array([2*x[0], 2*x[1]])
        def hess_f(x): return np.array([[2, 0], [0, 2]])

        def h(x): return np.array([x[0] + x[1] - 1])
        def grad_h(x): return np.array([1, 1])

        # Bad constraints: return shapes that don't match or are 3D
        def h_bad(x): return np.array([[x[0] + x[1] - 1]])
        def grad_h_bad(x): return np.array([[1, 1, 0]])
        def h_bad_len(x): return np.array([x[0], x[1], 1])

        # Test bad grad_h shape (wrong number of columns)
        with self.assertRaises(ValueError):
            sqp_equality_constrained(f, grad_f, hess_f, h, grad_h_bad, [1.0, 1.0])

        # Test bad h shape (2D instead of 1D)
        with self.assertRaises(ValueError):
            sqp_equality_constrained(f, grad_f, hess_f, h_bad, grad_h, [1.0, 1.0])

        # Test bad h length (doesn't match grad_h rows)
        with self.assertRaises(ValueError):
            sqp_equality_constrained(f, grad_f, hess_f, h_bad_len, grad_h, [1.0, 1.0])


if __name__ == '__main__':
    unittest.main()
