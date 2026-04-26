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

    def test_qp_list_input(self):
        G = [[1.0, 0.0], [0.0, 1.0]]
        c = [-1.0, -1.0]
        A = [[1.0, 1.0]]
        b = [2.0]

        x, lam = solve_eq_qp(G, c, A, b)
        np.testing.assert_allclose(x, [1.0, 1.0], atol=1e-5)
        
    def test_sqp(self):
        # min x^2 + y^2 s.t. x + y = 2
        # Global min at (1, 1)
        def f(x):
            return x[0]**2 + x[1]**2
        def grad_f(x):
            return np.array([2*x[0], 2*x[1]])
        def hess_f(x):
            return np.array([[2, 0], [0, 2]])
        def h(x):
            return np.array([x[0] + x[1] - 2])
        def grad_h(x):
            return np.array([1, 1])
        x0 = [0.0, 0.0]
        
        x_opt, _ = sqp_equality_constrained(f, grad_f, hess_f, h, grad_h, x0, tol=1e-4)
        np.testing.assert_allclose(x_opt, [1.0, 1.0], atol=1e-3)
        
    def test_barrier(self):
        # min x^2 s.t. x >= 1 => -x + 1 <= 0
        def f(x):
            return x[0]**2
        def grad_f(x):
            return np.array([2*x[0]])
        def hess_f(x):
            return np.array([[2]])
        
        def g_ineq(x):
            return np.array([-x[0] + 1])
        def grad_g_ineq(x):
            return np.array([[-1.0]])
        
        x0 = [2.0]
        # Barrier needs feasible start. 2 >= 1.
        x_opt, _ = barrier_method(f, grad_f, hess_f, g_ineq, grad_g_ineq, x0, tol=1e-4)
        np.testing.assert_allclose(x_opt, [1.0], atol=1e-3)

    def test_barrier_list_input(self):
        def f(x): return x[0]**2
        def grad_f(x): return [2*x[0]]
        def hess_f(x): return [[2]]
        def g_ineq(x): return [-x[0] + 1]
        def grad_g_ineq(x): return [[-1.0]]
        x0 = [2.0]
        x_opt, _ = barrier_method(f, grad_f, hess_f, g_ineq, grad_g_ineq, x0, tol=1e-4)
        np.testing.assert_allclose(x_opt, [1.0], atol=1e-3)

    def test_sqp_list_input(self):
        def f(x): return x[0]**2 + x[1]**2
        def grad_f(x): return [2*x[0], 2*x[1]]
        def hess_f(x): return [[2, 0], [0, 2]]
        def h(x): return [x[0] + x[1] - 2]
        def grad_h(x): return [1, 1]
        x0 = [0.0, 0.0]
        x_opt, _ = sqp_equality_constrained(f, grad_f, hess_f, h, grad_h, x0, tol=1e-4)
        np.testing.assert_allclose(x_opt, [1.0, 1.0], atol=1e-3)

    def test_barrier_validation(self):
        def f(x): return x[0]**2
        def grad_f(x): return np.array([2*x[0]])
        def hess_f(x): return np.array([[2]])
        def g_ineq(x): return np.array([-x[0] + 1])
        def grad_g_ineq(x): return np.array([[-1.0]])
        x0 = [2.0]

        with self.assertRaises(ValueError):
            barrier_method(f, grad_f, hess_f, g_ineq, grad_g_ineq, x0, mu0=0.0)

        with self.assertRaises(ValueError):
            barrier_method(f, grad_f, hess_f, g_ineq, grad_g_ineq, x0, mu0=-1.0)

    def test_qp_dimension_validation(self):
        from karush.constrained.qp import solve_eq_qp
        G_1d = np.array([1.0, 1.0])
        c = np.array([1.0, 1.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])

        with self.assertRaises(ValueError):
            solve_eq_qp(G_1d, c, A, b)

        G_2d = np.eye(2)
        A_1d = np.array([1.0, 1.0])
        with self.assertRaises(ValueError):
            solve_eq_qp(G_2d, c, A_1d, b)

    def test_qp_dimension_validation_c_b(self):
        from karush.constrained.qp import solve_eq_qp
        G = np.eye(2)
        c_2d = np.array([[1.0, 1.0]])
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])

        with self.assertRaises(ValueError):
            solve_eq_qp(G, c_2d, A, b)

        c_1d = np.array([1.0, 1.0])
        b_2d = np.array([[1.0]])
        with self.assertRaises(ValueError):
            solve_eq_qp(G, c_1d, A, b_2d)

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

    def test_primal_dual_list_input(self):
        from karush.constrained.primal_dual import primal_dual_qp
        G = [[1.0, 0.0], [0.0, 1.0]]
        c = [0.0, 0.0]
        A = [[1.0, 1.0]]
        b = [1.0]
        x0 = [0.1, 0.9]
        z0 = [1.0, 1.0]

        x_opt, _ = primal_dual_qp(G, c, A, b, x0, z0, tol=1e-4)
        np.testing.assert_allclose(x_opt, [0.5, 0.5], atol=1e-3)

    def test_primal_dual_validation(self):
        from karush.constrained.primal_dual import primal_dual_qp
        G = np.eye(2)
        c = np.array([np.nan, 0.0])
        A = np.array([[1.0, 1.0]])
        b = np.array([1.0])
        x0 = [0.1, 0.9]
        z0 = [1.0, 1.0]

        with self.assertRaises(ValueError):
            primal_dual_qp(G, c, A, b, x0, z0)

        c = np.zeros(2)
        A = np.array([[1.0, np.inf]])
        with self.assertRaises(ValueError):
            primal_dual_qp(G, c, A, b, x0, z0)

    def test_sdp_barrier_validation(self):
        from karush.semidefinite.interior_point import solve_sdp_barrier
        C = np.eye(2)
        A_list = [np.eye(2)]
        b = np.array([1.0])
        X0 = np.eye(2)

        C_nan = np.array([[np.nan, 0.0], [0.0, 1.0]])
        with self.assertRaises(ValueError):
            solve_sdp_barrier(C_nan, A_list, b, X0)

        b_inf = np.array([np.inf])
        with self.assertRaises(ValueError):
            solve_sdp_barrier(C, A_list, b_inf, X0)

        A_list_inf = [np.array([[np.inf, 0.0], [0.0, 1.0]])]
        with self.assertRaises(ValueError):
            solve_sdp_barrier(C, A_list_inf, b, X0)

        X0_nan = np.array([[1.0, 0.0], [0.0, np.nan]])
        with self.assertRaises(ValueError):
            solve_sdp_barrier(C, A_list, b, X0_nan)

    def test_interior_point_initial_mu_validation(self):
        from karush.semidefinite.interior_point import solve_sdp_barrier
        C = np.eye(2)
        A_list = [np.eye(2)]
        b = np.array([1.0])
        X0 = np.eye(2)

        with self.assertRaises(ValueError):
            solve_sdp_barrier(C, A_list, b, X0, initial_mu=0.0)

        with self.assertRaises(ValueError):
            solve_sdp_barrier(C, A_list, b, X0, initial_mu=-1.0)

if __name__ == '__main__':
    unittest.main()
