import time
import numpy as np
from karush.unconstrained.conjugate_gradient import conjugate_gradient

def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    g = np.zeros_like(x)
    g[0] = -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2)
    g[1] = 200 * (x[1] - x[0]**2)
    return g

x0 = [-1.2, 1.0]

start = time.time()
for _ in range(1000):
    conjugate_gradient(rosenbrock, rosenbrock_grad, x0)
print("Time:", time.time() - start)
