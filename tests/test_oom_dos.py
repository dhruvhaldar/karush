import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from karush.constrained.qp import solve_eq_qp
from karush.constrained.sqp import sqp_equality_constrained
from karush.constrained.primal_dual import primal_dual_qp
from karush.semidefinite.interior_point import solve_sdp_barrier
from karush.unconstrained.quasi_newton import bfgs_method

class TestOOMDos(unittest.TestCase):
    def test_qp_oom(self):
        G = np.eye(1)
        c = np.ones(1)
        A = np.ones((10001, 1))
        b = np.ones(10001)
        with self.assertRaisesRegex(ValueError, "System dimensions exceed safe limit for memory allocation."):
            solve_eq_qp(G, c, A, b)

    def test_sqp_oom(self):
        def f(x): return x[0]**2
        def grad_f(x): return np.array([2*x[0]])
        def hess_f(x): return np.array([[2]])
        def h(x): return np.ones(10001)
        def grad_h(x): return np.ones((10001, 1))

        with self.assertRaisesRegex(ValueError, "System dimensions exceed safe limit for memory allocation."):
            sqp_equality_constrained(f, grad_f, hess_f, h, grad_h, [1.0])

    def test_primal_dual_oom(self):
        G = np.eye(1)
        c = np.ones(1)
        A = np.ones((10001, 1))
        b = np.ones(10001)
        x0 = np.array([1.0])
        z0 = np.array([1.0])
        with self.assertRaisesRegex(ValueError, "System dimensions exceed safe limit for memory allocation."):
            primal_dual_qp(G, c, A, b, x0, z0)

    def test_interior_point_oom(self):
        n = 2 # dim_vec = 3
        C = np.eye(n)
        m = 10000
        A_list = [np.eye(n) for _ in range(m)]
        b = np.ones(m)
        X0 = np.eye(n)
        with self.assertRaisesRegex(ValueError, "System dimensions exceed safe limit for memory allocation."):
            solve_sdp_barrier(C, A_list, b, X0)

    def test_bfgs_oom(self):
        def f(x): return np.sum(x**2)
        def grad_f(x): return 2*x
        x0 = np.ones(10001)
        with self.assertRaisesRegex(ValueError, "System dimensions exceed safe limit for memory allocation."):
            bfgs_method(f, grad_f, x0)

if __name__ == '__main__':
    unittest.main()
