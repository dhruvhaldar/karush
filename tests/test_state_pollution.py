import numpy as np
import pytest
from karush.constrained.barrier import barrier_method

def test_barrier_state_pollution():
    cached_hess = np.eye(2)
    def f(x):
        return x[0]**2 + x[1]**2

    def grad_f(x):
        return np.array([2*x[0], 2*x[1]])

    def hess_f(x):
        return cached_hess

    def g_ineq(x):
        return np.array([x[0] - 1])

    def grad_g_ineq(x):
        return np.array([[1.0, 0.0]])

    x0 = np.array([0.0, 0.0])

    # Run a few iterations
    barrier_method(f, grad_f, hess_f, g_ineq, grad_g_ineq, x0, mu0=1.0, max_iter=2)

    # Assert that cached_hess was not mutated
    assert np.all(cached_hess == np.eye(2)), "State pollution vulnerability: cached_hess was mutated!"
