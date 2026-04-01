import numpy as np

def solve_eq_qp(G, c, A, b):
    """
    Solves the equality constrained quadratic program:
    min 1/2 x^T G x + c^T x
    s.t. A x = b
    
    Using the KKT system:
    [ G  A^T ] [ x ] = [ -c ]
    [ A   0  ] [ lambda ]   [ b  ]
    """
    # Security Enhancement: Add input sanitization to reject non-finite values (NaN/Inf)
    # which can lead to silent data corruption, infinite loops in solvers, or unhandled exceptions.
    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(c)):
        raise ValueError("Input arrays G and c must contain only finite numbers.")
    if A is not None and A.size > 0 and (not np.all(np.isfinite(A)) or not np.all(np.isfinite(b))):
        raise ValueError("Constraint arrays A and b must contain only finite numbers.")

    n = G.shape[0]
    # Handle cases where A is empty or None
    if A is None or A.size == 0:
        # Unconstrained QP: G x = -c
        return np.linalg.solve(G, -c), np.array([])
        
    m = A.shape[0]
    
    # Performance optimization: Replace np.block and np.concatenate with pre-allocation
    # and direct assignment. np.block creates unnecessary memory allocations and copies.
    KKT_mat = np.zeros((n + m, n + m))
    KKT_mat[:n, :n] = G
    KKT_mat[:n, n:] = A.T
    KKT_mat[n:, :n] = A
    
    rhs = np.empty(n + m)
    rhs[:n] = -c
    rhs[n:] = b
    
    try:
        sol = np.linalg.solve(KKT_mat, rhs)
    except np.linalg.LinAlgError:
        # Matrix might be singular
        return np.zeros(n), np.zeros(m)
    
    x = sol[:n]
    lam = sol[n:]
    
    return x, lam
