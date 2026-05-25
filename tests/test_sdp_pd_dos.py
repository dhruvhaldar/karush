import numpy as np
import pytest
from karush.semidefinite.interior_point import solve_sdp_barrier

def test_sdp_strictly_pd_dos():
    C = np.array([[1.0, 0.0], [0.0, 1.0]])
    A1 = np.array([[1.0, 0.0], [0.0, 0.0]])
    A_list = [A1]
    b = np.array([1.0])

    # Singular initial point
    X0_singular = np.array([[1.0, 0.0], [0.0, 0.0]])
    with pytest.raises(ValueError, match="Initial point X0 must be strictly positive definite"):
        solve_sdp_barrier(C, A_list, b, X0_singular)

    # Indefinite initial point
    X0_indefinite = np.array([[1.0, 0.0], [0.0, -1.0]])
    with pytest.raises(ValueError, match="Initial point X0 must be strictly positive definite"):
        solve_sdp_barrier(C, A_list, b, X0_indefinite)
