import numpy as np

def sqp_equality_constrained(f, grad_f, hess_f, h, grad_h, x0, tol=1e-6, max_iter=20):
    """
    Simple SQP for equality constrained optimization.
    min f(x)
    s.t. h(x) = 0
    
    This is a local SQP method without line search or merit function, 
    intended for demonstration purposes.
    """
    # Security Enhancement: Prevent memory exhaustion (OOM DoS) before allocating massive arrays
    n_check = len(x0)
    if n_check > 10000:
        raise ValueError("System dimensions exceed safe limit for memory allocation.")

    x = np.array(x0, dtype=float)
    if x.ndim != 1:
        raise ValueError("Initial guess x0 must be a 1D vector.")

    # Security Enhancement: Add input sanitization to reject non-finite values (NaN/Inf)
    # which can lead to silent data corruption, infinite loops in solvers, or unhandled exceptions.
    if not np.all(np.isfinite(x)):
        raise ValueError("Initial guess x0 must contain only finite numbers.")
    if isinstance(tol, bool) or not isinstance(tol, (int, float, np.number)) or np.isnan(tol) or tol <= 0:
        raise ValueError("Tolerance tol must be strictly positive.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("Maximum iterations max_iter must be a positive integer.")
    if max_iter > 10000:
        raise ValueError("Maximum iterations max_iter exceeds safe limit.")

    history = [x.copy()]
    
    # Optimization: pre-allocate KKT_mat and rhs lazily on the first iteration
    # to avoid double evaluating grad_h(x).
    KKT_mat = None
    rhs = None
    n = x.shape[0]

    # Performance optimization: Precompute tol**2 to avoid np.linalg.norm in inner loop.
    tol_sq = tol**2

    for k in range(max_iter):
        # DoS Prevention: Convert function outputs to numpy arrays to prevent unhandled
        # AttributeError/TypeError exceptions if user functions return standard Python lists.
        g = np.asarray(grad_f(x), dtype=float)
        if g.ndim != 1:
            raise ValueError("Gradient must be a 1D vector.")
        if g.shape[0] != x.shape[0]:
            raise ValueError("Gradient dimension must match x.")
        # Approximate Hessian of Lagrangian. For simplicity, use hess_f(x).
        # A full implementation would use the Hessian of the Lagrangian:
        # W = hess_f(x) + sum(lam_i * hess_h_i(x))
        W = np.asarray(hess_f(x), dtype=float)
        if W.ndim != 2:
            raise ValueError("Hessian must be a 2D matrix.")
        if W.shape[0] != W.shape[1] or W.shape[0] != x.shape[0]:
            raise ValueError("Hessian must be a square matrix matching x dimensions.")
        
        c_val = np.asarray(h(x), dtype=float)
        A = np.asarray(grad_h(x), dtype=float)
        
        # Ensure correct shapes for A and c_val
        if A.ndim == 1: 
            A = A.reshape(1, -1)
        
        c_val = np.atleast_1d(c_val)
        
        if A.ndim != 2:
            raise ValueError("Constraint gradient must be a 2D matrix.")
        if A.shape[1] != x.shape[0]:
            raise ValueError("Constraint gradient columns must match x dimensions.")
        if c_val.ndim != 1:
            raise ValueError("Constraint values must be a 1D vector.")
        if c_val.shape[0] != A.shape[0]:
            raise ValueError("Constraint values length must match number of constraint gradient rows.")

        if KKT_mat is None:
            m = A.shape[0]
            if n + m > 10000:
                raise ValueError("System dimensions exceed safe limit for memory allocation.")
            KKT_mat = np.zeros((n + m, n + m))
            rhs = np.empty(n + m)

        # Performance optimization: Replace `solve_eq_qp` which allocates a new block matrix
        # and rhs array every iteration with in-place updates to the pre-allocated ones.
        KKT_mat[:n, :n] = W
        KKT_mat[:n, n:] = A.T
        KKT_mat[n:, :n] = A

        rhs[:n] = -g
        rhs[n:] = -c_val
        
        sol = np.linalg.solve(KKT_mat, rhs)
        
        p = sol[:n]
        x_new = x + p
        
        if np.dot(p, p) < tol_sq and np.dot(c_val, c_val) < tol_sq:
            x = x_new
            history.append(x)
            break
            
        x = x_new
        history.append(x)
        
    return x, np.array(history)
