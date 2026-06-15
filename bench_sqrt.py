import time
import numpy as np

tol = 1e-6
tol_sq = tol**2

g_norm_sq = 1e-13

start = time.time()
for _ in range(10000000):
    a = np.sqrt(g_norm_sq) < tol
print("With sqrt:", time.time() - start)

start = time.time()
for _ in range(10000000):
    a = g_norm_sq < tol_sq
print("Without sqrt:", time.time() - start)
