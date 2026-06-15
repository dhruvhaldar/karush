import time
import numpy as np

g = np.random.randn(100)
tol = 1e-6
tol_sq = tol**2

start = time.time()
for _ in range(1000000):
    np.linalg.norm(g) < tol
print("linalg.norm:", time.time() - start)

start = time.time()
for _ in range(1000000):
    np.dot(g, g) < tol_sq
print("dot:", time.time() - start)
