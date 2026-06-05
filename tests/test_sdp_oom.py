import numpy as np
import pytest
from karush.semidefinite.interior_point import solve_sdp_barrier

def test_sdp_oom():
    n = 200 # dim_vec = 20100
    m = 1000
    C = np.eye(n)
    b = np.ones(m)
    # Use identical reference to avoid memory usage in Python list
    A_mat = np.eye(n)
    A_list = [A_mat] * m
    X0 = np.eye(n)

    # Current behavior: this will try to allocate `np.array(A_list)` which is 1000 x 200 x 200 floats = 40M elements (320MB).
    # Then it will raise ValueError "System dimensions exceed safe limit".
    # Wait, if we use n=1000, m=10000, it would allocate 10000 x 1000 x 1000 = 80GB, causing real OOM.
    # We expect the ValueError to be raised *before* allocation.
    with pytest.raises(ValueError, match="System dimensions exceed safe limit"):
        solve_sdp_barrier(C, A_list, b, X0)

if __name__ == "__main__":
    test_sdp_oom()
