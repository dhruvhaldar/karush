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
    if tol <= 0:
        raise ValueError("Tolerance tol must be strictly positive.")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("Maximum iterations max_iter must be a positive integer.")
    history = [x.copy()]
    g = grad_f(x)
    p = -g
    
    # Performance optimization: Cache the squared norm of the gradient
    # to avoid recomputing it in the next iteration's beta calculation.
    g_dot_g = np.dot(g, g)

    for k in range(max_iter):
        # We can also use the cached squared norm for the stopping criterion
        # if sqrt(g_dot_g) < tol, but we'll leave it as np.linalg.norm for exactness
        # or just use g_dot_g < tol**2.
        if np.sqrt(g_dot_g) < tol:
            break
            
        # Line search
        alpha = 1.0
        rho = 0.5
        c = 1e-4
        
        # Backtracking line search ensuring sufficient decrease
        # Pre-compute values to avoid re-evaluating f(x) and the dot product in the loop
        fx = f(x)
        expected_decrease = c * np.dot(g, p)

        while f(x + alpha * p) > fx + alpha * expected_decrease:
            alpha *= rho
            if alpha < 1e-10:
                break # Avoid infinite loop
            
        x_new = x + alpha * p
        g_new = grad_f(x_new)
        
        g_new_dot_g_new = np.dot(g_new, g_new)

        # Fletcher-Reeves update
        beta = g_new_dot_g_new / (g_dot_g + 1e-10)
        p_new = -g_new + beta * p
        
        # FIX: Reset to steepest descent if not a descent direction
        if np.dot(p_new, g_new) >= 0:
            p_new = -g_new
            
        p = p_new
        
        g = g_new
        g_dot_g = g_new_dot_g_new
        x = x_new
        history.append(x.copy())
        
    return x, np.array(history)
