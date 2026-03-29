import numpy as np

def newton_method(f, grad_f, hess_f, x0, tol=1e-6, max_iter=100):
    """
    Newton's method for unconstrained optimization.
    
    Args:
        f: Objective function.
        grad_f: Gradient of the objective function.
        hess_f: Hessian of the objective function.
        x0: Initial guess.
        tol: Tolerance for stopping criterion.
        max_iter: Maximum number of iterations.
        
    Returns:
        x_opt: Optimal solution.
        history: List of iterates.
    """
    # Security Enhancement: Add input sanitization to reject non-finite values (NaN/Inf)
    # which can lead to silent data corruption, infinite loops in solvers, or unhandled exceptions.
    if not np.all(np.isfinite(x0)):
        raise ValueError("Initial guess x0 must contain only finite numbers.")
    if tol <= 0:
        raise ValueError("Tolerance tol must be strictly positive.")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("Maximum iterations max_iter must be a positive integer.")

    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    for k in range(max_iter):
        g = grad_f(x)
        if np.linalg.norm(g) < tol:
            break
            
        H = hess_f(x)
        # Solve H * p = -g
        try:
            p = np.linalg.solve(H, -g)
        except np.linalg.LinAlgError:
            # Fallback for singular Hessian or if simple Newton fails locally
            # In a robust implementation, we might use trust region or modify H
            p = -g 

        # Line search (backtracking)
        alpha = 1.0
        rho = 0.5
        c = 1e-4

        # Pre-compute values to avoid re-evaluating f(x) and the dot product in the loop
        fx = f(x)
        expected_decrease = c * np.dot(g, p)

        while f(x + alpha * p) > fx + alpha * expected_decrease:
            alpha *= rho
            if alpha < 1e-10: # Safety break
                break
            
        x += alpha * p
        history.append(x.copy())
        
    return x, np.array(history)
