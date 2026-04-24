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
    if isinstance(tol, bool) or not isinstance(tol, (int, float, np.number)) or np.isnan(tol) or tol <= 0:
        raise ValueError("Tolerance tol must be strictly positive.")
    if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("Maximum iterations max_iter must be a positive integer.")
    if max_iter > 10000:
        raise ValueError("Maximum iterations max_iter exceeds safe limit.")

    x = np.array(x0, dtype=float)
    history = [x.copy()]
    
    # Performance optimization: Evaluate objective function once outside the loop
    # and cache the accepted line search value to avoid redundant f(x) calls per iteration.
    fx = np.asarray(f(x), dtype=float)

    for k in range(max_iter):
        # DoS Prevention: Convert function outputs to numpy arrays to prevent unhandled
        # AttributeError/TypeError exceptions if user functions return standard Python lists.
        g = np.asarray(grad_f(x), dtype=float)
        if np.linalg.norm(g) < tol:
            break
            
        H = np.asarray(hess_f(x), dtype=float)
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

        expected_decrease = c * np.dot(g, p)

        while True:
            f_new = np.asarray(f(x + alpha * p), dtype=float)
            if np.all(f_new <= fx + alpha * expected_decrease):
                break
            alpha *= rho
            if alpha < 1e-10: # Safety break
                f_new = np.asarray(f(x + alpha * p), dtype=float)
                break
            
        x += alpha * p
        fx = f_new  # Cache the accepted function value for the next iteration
        history.append(x.copy())
        
    return x, np.array(history)
