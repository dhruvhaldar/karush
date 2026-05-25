import numpy as np
import pytest
from karush.constrained.barrier import barrier_method

def test_barrier_strictly_feasible_dos():
    def f(x): return x[0]**2 + x[1]**2
    def grad_f(x): return np.array([2*x[0], 2*x[1]])
    def hess_f(x): return np.array([[2, 0], [0, 2]])
    def g_ineq(x): return np.array([x[0] + x[1] - 1])
    def grad_g_ineq(x): return np.array([[1, 1]])

    # Infeasible initial point: x0=[1, 1] => g = 1 >= 0
    with pytest.raises(ValueError, match="Initial guess x0 must be strictly feasible"):
        barrier_method(f, grad_f, hess_f, g_ineq, grad_g_ineq, [1.0, 1.0])
