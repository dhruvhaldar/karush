import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from karush.constrained.qp import solve_eq_qp
from karush.constrained.sqp import sqp_equality_constrained
from karush.constrained.barrier import barrier_method

class TestConstrained(unittest.TestCase):
    def test_qp(self):
        # min 0.5 (x-1)^2 + 0.5 (y-1)^2
        # = 0.5 x^2 + 0.5 y^2 - x - y + constant
        # G = I, c = [-1, -1]
        # s.t. x + y = 2
        # Optimal solution (1, 1)
        
        G = np.eye(2)
        c = np.array([-1.0, -1.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([2.0])
        
        x, lam = solve_eq_qp(G, c, A, b)
        np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-5)
        
    def test_sqp(self):
        # min x^2 + y^2 s.t. x + y = 2
        # Global min at (1, 1)
        f = lambda x: x[0]**2 + x[1]**2
        grad_f = lambda x: np.array([2*x[0], 2*x[1]])
        hess_f = lambda x: np.array([[2, 0], [0, 2]])
        h = lambda x: np.array([x[0] + x[1] - 2])
        grad_h = lambda x: np.array([1, 1])
        x0 = [0.0, 0.0]
        
        x_opt, _ = sqp_equality_constrained(f, grad_f, hess_f, h, grad_h, x0, tol=1e-4)
        np.testing.assert_allclose(x_opt, [1.0, 1.0], atol=1e-3)
        
    def test_barrier(self):
        # min x^2 s.t. x >= 1 => -x + 1 <= 0
        f = lambda x: x[0]**2
        grad_f = lambda x: np.array([2*x[0]])
        hess_f = lambda x: np.array([[2]])
        
        g_ineq = lambda x: np.array([-x[0] + 1])
        grad_g_ineq = lambda x: np.array([[-1.0]]) 
        
        x0 = [2.0]
        # Barrier needs feasible start. 2 >= 1.
        x_opt, _ = barrier_method(f, grad_f, hess_f, g_ineq, grad_g_ineq, x0, tol=1e-4)
        np.testing.assert_allclose(x_opt, [1.0], atol=1e-3)

    def test_primal_dual(self):
        from karush.constrained.primal_dual import primal_dual_qp
        # min 0.5 (x1^2 + x2^2) s.t. x1+x2=1, x>=0
        G = np.eye(2)
        c = np.zeros(2)
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        x0 = [0.1, 0.9] # Interior point x > 0
        z0 = [1.0, 1.0] # Interior point z > 0
        
        x_opt, _ = primal_dual_qp(G, c, A, b, x0, z0, tol=1e-4)
        np.testing.assert_allclose(x_opt, [0.5, 0.5], atol=1e-3)

if __name__ == '__main__':
    unittest.main()
