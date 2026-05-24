import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from karush.constrained.sqp import sqp_equality_constrained
from karush.constrained.primal_dual import primal_dual_qp
from karush.constrained.barrier import barrier_method
from karush.semidefinite.interior_point import solve_sdp_barrier
from karush.unconstrained.newton import newton_method

class TestSolveSingular(unittest.TestCase):
    def test_sqp_singular(self):
        def f(x): return x[0]**2 + x[1]**2
        def grad_f(x): return np.array([2*x[0], 2*x[1]])
        def hess_f(x): return np.array([[0, 0], [0, 0]]) # Singular Hessian
        def h(x): return np.array([x[0] + x[1] - 1])
        def grad_h(x): return np.array([[0, 0]]) # Singular constraint gradient

        with self.assertRaises(np.linalg.LinAlgError):
            sqp_equality_constrained(f, grad_f, hess_f, h, grad_h, [1.0, 1.0])

    def test_primal_dual_singular(self):
        G = np.array([[0, 0], [0, 0]]) # Singular G
        c = np.array([1, 1])
        A = np.array([[0, 0]]) # Singular A
        b = np.array([1])
        x0 = np.array([1.0, 1.0])
        z0 = np.array([1.0, 1.0])

        with self.assertRaises(np.linalg.LinAlgError):
            primal_dual_qp(G, c, A, b, x0, z0)

    def test_barrier_singular(self):
        def f(x): return x[0]**2 + x[1]**2
        def grad_f(x): return np.array([2*x[0], 2*x[1]])
        def hess_f(x): return np.array([[0, 0], [0, 0]]) # Singular Hessian
        def g_ineq(x): return np.array([x[0] + x[1] - 1])
        def grad_g_ineq(x): return np.array([[0, 0]]) # Singular constraint gradient

        with self.assertRaises(np.linalg.LinAlgError):
            barrier_method(f, grad_f, hess_f, g_ineq, grad_g_ineq, [-1.0, -1.0])

    def test_interior_point_singular(self):
        C = np.array([[1, 0], [0, 1]])
        A_list = [np.array([[0, 0], [0, 0]])] # Singular constraint matrix
        b = np.array([1])
        X0 = np.array([[1, 0], [0, 1]]) # PD initial point

        # Should raise LinAlgError due to singular KKT matrix constructed from A_list
        with self.assertRaises(np.linalg.LinAlgError):
            solve_sdp_barrier(C, A_list, b, X0)

    def test_newton_singular(self):
        def f(x): return x[0]**2 + x[1]**2
        def grad_f(x): return np.array([2*x[0], 2*x[1]])
        def hess_f(x): return np.array([[0, 0], [0, 0]]) # Singular Hessian

        with self.assertRaises(np.linalg.LinAlgError):
            newton_method(f, grad_f, hess_f, [1.0, 1.0])

if __name__ == '__main__':
    unittest.main()
