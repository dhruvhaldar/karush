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
    n = G.shape[0]
    # Handle cases where A is empty or None
    if A is None or A.size == 0:
        # Unconstrained QP: G x = -c
        return np.linalg.solve(G, -c), np.array([])
        
    m = A.shape[0]
    
    KKT_mat = np.block([
        [G, A.T],
        [A, np.zeros((m, m))]
    ])
    
    rhs = np.concatenate([-c, b])
    
    try:
        sol = np.linalg.solve(KKT_mat, rhs)
    except np.linalg.LinAlgError:
        # Matrix might be singular
        return np.zeros(n), np.zeros(m)
    
    x = sol[:n]
    lam = sol[n:]
    
    return x, lam
