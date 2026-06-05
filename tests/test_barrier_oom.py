import numpy as np
import pytest
from karush.constrained.barrier import barrier_method

def test_barrier_oom():
    def f(x): return np.sum(x**2)
    def grad_f(x): return 2*x
    def hess_f(x): return 2*np.eye(len(x))

    # 5000 constraints, 6000 variables -> n+m = 11000
    # grad_g is 5000x6000 -> 30M elements -> 240MB
    # This will easily allocate 240MB, but we want to prevent it if n+m > 10000
    m = 5000
    n = 6000

    def g_ineq(x): return -np.ones(m)
    def grad_g_ineq(x): return np.zeros((m, n))

    x0 = np.ones(n)
    with pytest.raises(ValueError, match="System dimensions exceed safe limit"):
        barrier_method(f, grad_f, hess_f, g_ineq, grad_g_ineq, x0)

if __name__ == "__main__":
    test_barrier_oom()
