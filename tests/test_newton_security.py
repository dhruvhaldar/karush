import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from karush.unconstrained.newton import newton_method

class TestSecurity(unittest.TestCase):
    def test_newton_strict_shape_validation(self):
        def f(x): return x[0]**2 + x[1]**2
        def grad_f(x): return np.array([2*x[0], 2*x[1]])
        def grad_f_bad(x): return np.array([2*x[0], 2*x[1], 0])
        def hess_f(x): return np.array([[2, 0], [0, 2]])
        def hess_f_bad_shape(x): return np.array([[2, 0, 0], [0, 2, 0]])

        # Test non-square Hessian
        with self.assertRaises(ValueError):
            newton_method(f, grad_f, hess_f_bad_shape, [1.0, 1.0])

        # Test dimension mismatch between Hessian and gradient
        with self.assertRaises(ValueError):
            newton_method(f, grad_f_bad, hess_f, [1.0, 1.0])

    def test_newton_gradient_shape_mismatch(self):
        def f(x): return 0.0
        def grad_f(x): return np.array([1.0, 2.0]) # Shape (2,) while x is (1,)
        def hess_f(x): return np.array([[1.0, 0.0], [0.0, 1.0]])

        with self.assertRaises(ValueError):
            newton_method(f, grad_f, hess_f, [0.0])

if __name__ == '__main__':
    unittest.main()
