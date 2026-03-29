import time
import numpy as np
from karush.semidefinite.interior_point import svec, smat

n = 200
M = np.random.randn(n, n)
M = M + M.T # make it symmetric

start = time.time()
for _ in range(10):
    v = svec(M)
print("svec loop time:", time.time() - start)

start = time.time()
for _ in range(10):
    M2 = smat(v, n)
print("smat loop time:", time.time() - start)

def svec_vec(M):
    n = M.shape[0]
    # Indices for upper triangle
    idx_i, idx_j = np.triu_indices(n)
    v = M[idx_i, idx_j]
    # Multiply off-diagonal by sqrt(2)
    off_diag = idx_i != idx_j
    v[off_diag] *= np.sqrt(2)
    return v

def smat_vec(v, n):
    M = np.zeros((n, n))
    idx_i, idx_j = np.triu_indices(n)
    M[idx_i, idx_j] = v
    off_diag = idx_i != idx_j
    M[idx_i[off_diag], idx_j[off_diag]] /= np.sqrt(2)

    # Fill lower triangle
    M[idx_j[off_diag], idx_i[off_diag]] = M[idx_i[off_diag], idx_j[off_diag]]
    return M

start = time.time()
for _ in range(10):
    v2 = svec_vec(M)
print("svec_vec time:", time.time() - start)

start = time.time()
for _ in range(10):
    M3 = smat_vec(v2, n)
print("smat_vec time:", time.time() - start)

print("Match svec:", np.allclose(v, v2))
print("Match smat:", np.allclose(M2, M3))
