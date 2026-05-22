import numpy as np

def conjugate_gradient(f, grad_f, x0, tol=1e-6, max_iter=100):
    """
    Nonlinear Conjugate Gradient method (Fletcher-Reeves).
    """
    x = np.array(x0, dtype=float)

    # Security Enhancement: Validate input dimensions to prevent unhandled TypeError/ValueError DoS.
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
    # DoS Prevention: Convert function outputs to numpy arrays to prevent unhandled
    # AttributeError/TypeError exceptions if user functions return standard Python lists.
    g = np.asarray(grad_f(x), dtype=float)
    # Security Enhancement: Validate dimensions of arrays returned by user-provided functions
    # before matrix operations or line searches to prevent unhandled ValueError DoS crashes.
    if g.ndim != 1:
        raise ValueError("Gradient must be a 1D vector.")
    if g.shape[0] != x.shape[0]:
        raise ValueError("Gradient dimension must match x.")
    g_norm_sq = np.dot(g, g)
    p = -g
    
    # Performance optimization: Evaluate objective function once outside the loop
    # and cache the accepted line search value to avoid redundant f(x) calls per iteration.
    fx_raw = f(x)
    try:
        fx_val = float(fx_raw)
    except (TypeError, ValueError):
        fx = np.asarray(fx_raw, dtype=float)
        fx_val = fx.item() if fx.size == 1 else fx

    for k in range(max_iter):
        if np.sqrt(g_norm_sq) < tol:
            break
            
        # Line search
        alpha = 1.0
        rho = 0.5
        c = 1e-4
        
        expected_decrease = c * np.dot(g, p)

        while True:
            f_new_raw = f(x + alpha * p)
            try:
                f_new_val = float(f_new_raw)
                is_scalar = True
            except (TypeError, ValueError):
                f_new = np.asarray(f_new_raw, dtype=float)
                f_new_val = f_new.item() if f_new.size == 1 else f_new
                is_scalar = f_new.size == 1

            # Use scalar comparison or fallback to np.all for vector functions
            if is_scalar:
                if f_new_val <= fx_val + alpha * expected_decrease:
                    break
            else:
                if np.all(f_new_val <= fx_val + alpha * expected_decrease):
                    break

            alpha *= rho
            if alpha < 1e-10:
                f_new_raw = f(x + alpha * p)
                try:
                    f_new_val = float(f_new_raw)
                except (TypeError, ValueError):
                    f_new = np.asarray(f_new_raw, dtype=float)
                    f_new_val = f_new.item() if f_new.size == 1 else f_new
                break # Avoid infinite loop
            
        x_new = x + alpha * p
        fx_val = f_new_val
        g_new = np.asarray(grad_f(x_new), dtype=float)
        if g_new.ndim != 1:
            raise ValueError("Gradient must be a 1D vector.")
        if g_new.shape[0] != x.shape[0]:
            raise ValueError("Gradient dimension must match x.")
        
        # Performance optimization: Cache expensive vector dot products.
        # Computing the norm squared once and reusing it avoids redundant O(n) operations
        # per iteration for the stopping criteria and Fletcher-Reeves update.
        g_new_norm_sq = np.dot(g_new, g_new)

        # Fletcher-Reeves update
        beta = g_new_norm_sq / (g_norm_sq + 1e-10)
        p_new = -g_new + beta * p
        
        # FIX: Reset to steepest descent if not a descent direction
        # Security Enhancement: Prevent division-by-zero or unhandled math exceptions if dot product is evaluated on incorrectly shaped arrays.
        if np.dot(p_new, g_new) >= 0:
            p_new = -g_new
            
        p = p_new
        
        g = g_new
        g_norm_sq = g_new_norm_sq
        x = x_new
        history.append(x.copy())
        
    return x, np.array(history)
