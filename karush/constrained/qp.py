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
    # Security Enhancement: Prevent memory exhaustion (OOM DoS) before allocating massive arrays
    n_check = len(G)
    m_check = len(A) if A is not None and len(A) > 0 else 0
    if n_check + m_check > 10000:
        raise ValueError("System dimensions exceed safe limit for memory allocation.")

    G = np.asarray(G, dtype=float)
    c = np.asarray(c, dtype=float)
    if A is not None:
        A = np.asarray(A, dtype=float)
        b = np.asarray(b, dtype=float)

    # Security Enhancement: Add input sanitization to reject non-finite values (NaN/Inf)
    # which can lead to silent data corruption, infinite loops in solvers, or unhandled exceptions.
    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(c)):
        raise ValueError("Input arrays G and c must contain only finite numbers.")
    if A is not None and A.size > 0 and (not np.all(np.isfinite(A)) or not np.all(np.isfinite(b))):
        raise ValueError("Constraint arrays A and b must contain only finite numbers.")

    if G.ndim != 2:
        raise ValueError("Input array G must be a 2D matrix.")
    if c.ndim != 1:
        raise ValueError("Input array c must be a 1D vector.")
    if A is not None and A.size > 0 and A.ndim != 2:
        raise ValueError("Constraint array A must be a 2D matrix.")
    if A is not None and A.size > 0 and b.ndim != 1:
        raise ValueError("Constraint array b must be a 1D vector.")

    n = G.shape[0]

    if G.shape[1] != n:
        raise ValueError("Input matrix G must be square.")
    if c.shape[0] != n:
        raise ValueError("Input vector c must have the same dimension as G.")
    if A is not None and A.size > 0:
        if A.shape[1] != n:
            raise ValueError("Constraint matrix A must have the same number of columns as G.")
        if b.shape[0] != A.shape[0]:
            raise ValueError("Constraint vector b must have the same number of rows as A.")
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
    
    sol = np.linalg.solve(KKT_mat, rhs)
    
    x = sol[:n]
    lam = sol[n:]
    
    return x, lam
