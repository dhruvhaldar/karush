import numpy as np

def conjugate_gradient(f, grad_f, x0, tol=1e-6, max_iter=100):
    """
    Nonlinear Conjugate Gradient method (Fletcher-Reeves).
    """
    x = np.array(x0, dtype=float)

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
    # DoS Prevention: Convert function outputs to numpy arrays to prevent unhandled
    # AttributeError/TypeError exceptions if user functions return standard Python lists.
    g = np.asarray(grad_f(x), dtype=float)
    g_norm_sq = np.dot(g, g)
    p = -g
    
    for k in range(max_iter):
        if np.sqrt(g_norm_sq) < tol:
            break
            
        # Line search
        alpha = 1.0
        rho = 0.5
        c = 1e-4
        
        # Backtracking line search ensuring sufficient decrease
        # Pre-compute values to avoid re-evaluating f(x) and the dot product in the loop
        fx = np.asarray(f(x), dtype=float)
        expected_decrease = c * np.dot(g, p)

        while np.all(np.asarray(f(x + alpha * p), dtype=float) > fx + alpha * expected_decrease):
            alpha *= rho
            if alpha < 1e-10:
                break # Avoid infinite loop
            
        x_new = x + alpha * p
        g_new = np.asarray(grad_f(x_new), dtype=float)
        
        # Performance optimization: Cache expensive vector dot products.
        # Computing the norm squared once and reusing it avoids redundant O(n) operations
        # per iteration for the stopping criteria and Fletcher-Reeves update.
        g_new_norm_sq = np.dot(g_new, g_new)

        # Fletcher-Reeves update
        beta = g_new_norm_sq / (g_norm_sq + 1e-10)
        p_new = -g_new + beta * p
        
        # FIX: Reset to steepest descent if not a descent direction
        if np.dot(p_new, g_new) >= 0:
            p_new = -g_new
            
        p = p_new
        
        g = g_new
        g_norm_sq = g_new_norm_sq
        x = x_new
        history.append(x.copy())
        
    return x, np.array(history)
