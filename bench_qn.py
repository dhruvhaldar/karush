import time
import numpy as np
from karush.unconstrained.quasi_newton import bfgs_method

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    g = np.zeros_like(x)
    g[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    g[1] = 200 * (x[1] - x[0]**2)
    return g

x0 = [-1.2, 1.0]

start = time.time()
for _ in range(5000):
    bfgs_method(rosenbrock, rosenbrock_grad, x0, tol=1e-8, max_iter=200)
print("Time BFGS:", time.time() - start)
