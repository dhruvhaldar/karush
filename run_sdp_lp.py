from line_profiler import LineProfiler
import numpy as np
from karush.semidefinite.interior_point import solve_sdp_barrier

def run_test():
    n = 20
    np.random.seed(0)
    C = np.random.randn(n, n)
    C = C + C.T

    A_list = []
    b_list = []
    for _ in range(5):
        A = np.random.randn(n, n)
        A = A + A.T
        A_list.append(A)
        b_list.append(np.trace(A @ np.eye(n)))

    X0 = np.eye(n)

    for _ in range(5):
        solve_sdp_barrier(C, A_list, b_list, X0, initial_mu=1.0, tol=1e-4, max_iter=10)

lp = LineProfiler()
lp.add_function(solve_sdp_barrier)
lp_wrapper = lp(run_test)
lp_wrapper()
lp.print_stats()
