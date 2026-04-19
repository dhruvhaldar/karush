import numpy as np
from .qp import solve_eq_qp

def sqp_equality_constrained(f, grad_f, hess_f, h, grad_h, x0, tol=1e-6, max_iter=20):
    """
    Simple SQP for equality constrained optimization.
    min f(x)
    s.t. h(x) = 0
    
    This is a local SQP method without line search or merit function, 
    intended for demonstration purposes.
    """
    x = np.array(x0, dtype=float)

    # Security Enhancement: Add input sanitization to reject non-finite values (NaN/Inf)
    # which can lead to silent data corruption, infinite loops in solvers, or unhandled exceptions.
    if not np.all(np.isfinite(x)):
        raise ValueError("Initial guess x0 must contain only finite numbers.")
    if not isinstance(tol, (int, float, np.number)) or isinstance(tol, bool) or np.isnan(tol) or tol <= 0:
        raise ValueError("Tolerance tol must be strictly positive.")
    if not isinstance(max_iter, int) or isinstance(max_iter, bool) or max_iter <= 0:
        raise ValueError("Maximum iterations max_iter must be a positive integer.")
    if max_iter > 10000:
        raise ValueError("Maximum iterations max_iter exceeds safe limit.")

    history = [x.copy()]
    
    for k in range(max_iter):
        # DoS Prevention: Convert function outputs to numpy arrays to prevent unhandled
        # AttributeError/TypeError exceptions if user functions return standard Python lists.
        g = np.asarray(grad_f(x), dtype=float)
        # Approximate Hessian of Lagrangian. For simplicity, use hess_f(x).
        # A full implementation would use the Hessian of the Lagrangian:
        # W = hess_f(x) + sum(lam_i * hess_h_i(x))
        W = np.asarray(hess_f(x), dtype=float)
        
        c_val = np.asarray(h(x), dtype=float)
        A = np.asarray(grad_h(x), dtype=float)
        
        # Ensure correct shapes for A and c_val
        if A.ndim == 1: 
            A = A.reshape(1, -1)
        
        c_val = np.atleast_1d(c_val)
        
        # Check convergence
        # KKT conditions: norm(g + A.T @ lam) < tol and norm(c) < tol
        # Here we just check the step size and constraint violation
        if np.linalg.norm(c_val) < tol and k > 0:
             # Also check gradient of lagrangian if we tracked lambda
             pass

        # Solve QP subproblem:
        # min 0.5 p' W p + g' p
        # s.t. A p + c = 0 => A p = -c
        
        try:
            p, lam_qp = solve_eq_qp(W, g, A, -c_val)
        except np.linalg.LinAlgError:
            break
        
        x_new = x + p
        
        if np.linalg.norm(p) < tol and np.linalg.norm(c_val) < tol:
            x = x_new
            history.append(x.copy())
            break
            
        x = x_new
        history.append(x.copy())
        
    return x, np.array(history)
