import numpy as np
import time

def f1():
    n = 100000
    p = np.random.rand(n)
    g_new = np.random.rand(n)
    beta = 0.5
    for _ in range(1000):
        # Current in-place modification
        p *= beta
        p -= g_new

def f2():
    n = 100000
    p = np.random.rand(n)
    g_new = np.random.rand(n)
    beta = 0.5
    for _ in range(1000):
        # We can use np.multiply with out parameter, but p *= beta is already in-place.
        p *= beta
        p -= g_new

t0 = time.time()
f1()
print(f"f1: {time.time() - t0}")
