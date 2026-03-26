import time
import numpy as np

n = 500
num_trials = 1000
L = np.random.randn(n, n)

rng = np.random.default_rng(123)
start = time.time()
r_list = []
candidates = []
for _ in range(num_trials):
    r = rng.standard_normal(n)
    r_list.append(r)
    x = np.sign(L @ r)
    x[x == 0] = 1
    candidates.append(x)
print("Loop time:", time.time() - start)

rng = np.random.default_rng(123)
start = time.time()
R = np.column_stack([rng.standard_normal(n) for _ in range(num_trials)])
X = np.sign(L @ R)
X[X == 0] = 1
candidates_vec = list(X.T)
print("Vectorized time:", time.time() - start)

print("Match:", np.all([np.allclose(candidates[i], candidates_vec[i]) for i in range(num_trials)]))
