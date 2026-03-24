import numpy as np
import time

n = 2000

G = np.eye(n)
x = np.random.rand(n)
z = np.random.rand(n)
r_L = np.random.rand(n)
r_C = np.random.rand(n)
sigma = 0.5
mu = 0.1

start = time.time()
X_inv = np.diag(1.0 / x)
Z = np.diag(z)
M1 = G + X_inv @ Z
rhs_1_1 = -r_L + X_inv @ ( -r_C + sigma * mu * np.ones(n) )
print("Original time:", time.time() - start)

start = time.time()
M2 = G + np.diag(z / x)
rhs_1_2 = -r_L + ( -r_C + sigma * mu * np.ones(n) ) / x
print("Optimized time:", time.time() - start)

print("M match:", np.allclose(M1, M2))
print("rhs match:", np.allclose(rhs_1_1, rhs_1_2))
