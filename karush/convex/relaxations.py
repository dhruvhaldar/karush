import numpy as np
from ..semidefinite.interior_point import solve_sdp_barrier

def max_cut_sdp_relaxation(W, tol=1e-4, max_iter=20):
    """
    Solves the Semidefinite Relaxation (SDR) for the Max-Cut problem.
    Minimize Tr(W X) subject to X_ii = 1, X >= 0.
    
    For the Max-Cut problem on a graph with weight matrix W,
    we want to maximize sum_{i<j} W_ij (1 - x_i x_j)/2.
    This is equivalent to minimizing x^T W x.
    """
    n = W.shape[0]
    
    A_list = []
    b_list = []
    
    for i in range(n):
        Ai = np.zeros((n, n))
        Ai[i, i] = 1.0
        A_list.append(Ai)
        b_list.append(1.0)
        
    X0 = np.eye(n)
    
    X_opt = solve_sdp_barrier(W, A_list, b_list, X0, tol=tol, max_iter=max_iter)
    
    return X_opt

def randomized_rounding(X, num_trials=100):
    """
    Applies randomized rounding to the SDP solution X to get binary variables {-1, 1}.
    Returns a list of candidate vectors.
    """
    n = X.shape[0]
    try:
        L = np.linalg.cholesky(X + 1e-6 * np.eye(n))
    except np.linalg.LinAlgError:
        w, v = np.linalg.eigh(X)
        w[w < 0] = 0
        # Reconstruct L such that L @ L.T approx X
        # X = V D V.T = (V D^0.5) (V D^0.5).T
        L = v @ np.diag(np.sqrt(w))
        
    candidates = []
    for _ in range(num_trials):
        r = np.random.randn(n)
        # x = sign(L @ r)
        x = np.sign(L @ r)
        x[x == 0] = 1
        candidates.append(x)
        
    return candidates
