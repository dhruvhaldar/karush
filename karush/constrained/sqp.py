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
    n = len(x)
    history = [x.copy()]
    
    for k in range(max_iter):
        g = grad_f(x)
        # Approximate Hessian of Lagrangian. For simplicity, use hess_f(x).
        # A full implementation would use the Hessian of the Lagrangian:
        # W = hess_f(x) + sum(lam_i * hess_h_i(x))
        W = hess_f(x) 
        
        c_val = h(x)
        A = grad_h(x)
        
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
