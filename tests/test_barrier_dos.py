import numpy as np
import pytest
from karush.constrained.barrier import barrier_method

def test_barrier_grad_g_shape():
    def f(x): return x[0]**2 + x[1]**2
    def grad_f(x): return np.array([2*x[0], 2*x[1]])
    def hess_f(x): return np.array([[2, 0], [0, 2]])
    def g_ineq(x): return np.array([x[0] + x[1] - 1])
    def grad_g_ineq(x): return np.array([[1, 1, 1]]) # 1x3 matrix instead of 1x2

    with pytest.raises(ValueError, match="Constraint gradient dimensions must match m x n."):
        barrier_method(f, grad_f, hess_f, g_ineq, grad_g_ineq, [0.0, 0.0])

def test_barrier_hess_shape():
    def f(x): return x[0]**2 + x[1]**2
    def grad_f(x): return np.array([2*x[0], 2*x[1]])
    def hess_f(x): return np.array([[2, 0, 0], [0, 2, 0]]) # 2x3 matrix instead of 2x2
    def g_ineq(x): return np.array([x[0] + x[1] - 1])
    def grad_g_ineq(x): return np.array([[1, 1]])

    with pytest.raises(ValueError, match="Hessian must be a square matrix matching x dimensions."):
        barrier_method(f, grad_f, hess_f, g_ineq, grad_g_ineq, [0.0, 0.0])

def test_barrier_grad_f_shape():
    def f(x): return x[0]**2 + x[1]**2
    def grad_f(x): return np.array([2*x[0], 2*x[1], 0]) # 1x3 instead of 1x2
    def hess_f(x): return np.array([[2, 0], [0, 2]])
    def g_ineq(x): return np.array([x[0] + x[1] - 1])
    def grad_g_ineq(x): return np.array([[1, 1]])

    with pytest.raises(ValueError, match="Gradient dimension must match x."):
        barrier_method(f, grad_f, hess_f, g_ineq, grad_g_ineq, [0.0, 0.0])
