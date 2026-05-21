import numpy as np
import pytest
from karush.constrained.primal_dual import primal_dual_qp

def test_primal_dual_strictly_positive_dos():
    G = np.eye(2)
    c = np.zeros(2)
    A = np.ones((1, 2))
    b = np.ones(1)

    # Test x0 containing zero
    x0_zero = np.array([0.0, 1.0])
    z0_valid = np.array([1.0, 1.0])
    with pytest.raises(ValueError, match="Initial points x0 and z0 must be strictly positive."):
        primal_dual_qp(G, c, A, b, x0_zero, z0_valid)

    # Test x0 containing negative
    x0_neg = np.array([-1.0, 1.0])
    with pytest.raises(ValueError, match="Initial points x0 and z0 must be strictly positive."):
        primal_dual_qp(G, c, A, b, x0_neg, z0_valid)

    # Test z0 containing zero
    x0_valid = np.array([1.0, 1.0])
    z0_zero = np.array([1.0, 0.0])
    with pytest.raises(ValueError, match="Initial points x0 and z0 must be strictly positive."):
        primal_dual_qp(G, c, A, b, x0_valid, z0_zero)

    # Test z0 containing negative
    z0_neg = np.array([1.0, -1.0])
    with pytest.raises(ValueError, match="Initial points x0 and z0 must be strictly positive."):
        primal_dual_qp(G, c, A, b, x0_valid, z0_neg)
