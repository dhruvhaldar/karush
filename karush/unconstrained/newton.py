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
    if x.ndim != 1:
        raise ValueError("Initial guess x0 must be a 1D vector.")
    history = [x.copy()]
    
    # Performance optimization: Evaluate objective function once outside the loop
    # and cache the accepted line search value to avoid redundant f(x) calls per iteration.
    fx_raw = f(x)
    try:
        fx_val = float(fx_raw)
    except (TypeError, ValueError):
        fx = np.asarray(fx_raw, dtype=float)
        fx_val = fx.item() if fx.size == 1 else fx

    for k in range(max_iter):
        # DoS Prevention: Convert function outputs to numpy arrays to prevent unhandled
        # AttributeError/TypeError exceptions if user functions return standard Python lists.
        g = np.asarray(grad_f(x), dtype=float)
        # Security Enhancement: Add input sanitization to validate array dimensions before passing
        # to np.linalg.solve or other matrix operations. Validating only for finite values is insufficient
        # and can lead to unhandled exception DoS crashes if user functions return incorrectly dimensioned arrays.
        if g.ndim != 1:
            raise ValueError("Gradient must be a 1D vector.")
        if g.shape[0] != x.shape[0]:
            raise ValueError("Gradient dimension must match x.")
        if np.linalg.norm(g) < tol:
            break
            
        H = np.asarray(hess_f(x), dtype=float)
        if H.ndim != 2:
            raise ValueError("Hessian must be a 2D matrix.")
        if H.shape[0] != H.shape[1]:
            raise ValueError("Hessian must be a square matrix.")
        if H.shape[0] != g.shape[0]:
            raise ValueError("Hessian dimensions must match gradient dimensions.")
        # Solve H * p = -g
        p = np.linalg.solve(H, -g)

        # Line search (backtracking)
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
            if alpha < 1e-10: # Safety break
                f_new_raw = f(x + alpha * p)
                try:
                    f_new_val = float(f_new_raw)
                except (TypeError, ValueError):
                    f_new = np.asarray(f_new_raw, dtype=float)
                    f_new_val = f_new.item() if f_new.size == 1 else f_new
                break
            
        x += alpha * p
        fx_val = f_new_val  # Cache the accepted function value for the next iteration
        history.append(x.copy())
        
    return x, np.array(history)
