import time
import numpy as np
from karush.semidefinite.interior_point import _get_svec_indices, svec

n = 50

idx_i, idx_j, off_diag = _get_svec_indices(n)
M = np.random.randn(n, n)
M = M + M.T

start = time.time()
for _ in range(10000):
    v = M[idx_i, idx_j]
    v[off_diag] *= np.sqrt(2)
print("direct:", time.time() - start)

start = time.time()
for _ in range(10000):
    svec(M)
print("svec function:", time.time() - start)
