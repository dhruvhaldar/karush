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
    # Security Enhancement: Bound dimensions before allocating arrays to prevent OOM DoS
    if len(x0) > 10000:
        raise ValueError("System dimensions exceed safe limit for memory allocation.")

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
    if len(x) > 10000:
        raise ValueError("System dimensions exceed safe limit for memory allocation.")
    history = [x.copy()]
    
    # Performance optimization: Evaluate objective function once outside the loop
    # and cache the accepted line search value to avoid redundant f(x) calls per iteration.
    fx_raw = f(x)
    try:
        fx_val = float(fx_raw)
    except (TypeError, ValueError):
        fx = np.asarray(fx_raw, dtype=float)
        fx_val = fx.item() if fx.size == 1 else fx

    # Performance optimization: Precompute tol**2 to avoid np.linalg.norm in inner loop.
    tol_sq = tol**2

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
        if np.dot(g, g) < tol_sq:
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

        # Performance optimization: Compute step = alpha * p once and scale
        # it in place (step *= rho) during backtracking instead of re-evaluating
        # alpha * p. This avoids redundant O(n) array allocations per iteration.
        step = alpha * p

        # Performance optimization: Avoid redundant array evaluation in line search acceptance
        # By assigning `x_new = x + step` first and passing it to `f`, we avoid evaluating `x + step`
        # again after the loop, saving an O(n) array allocation per line search acceptance.
        while True:
            x_new = x + step
            f_new_raw = f(x_new)
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
            step *= rho
            if alpha < 1e-10: # Safety break
                x_new = x + step
                f_new_raw = f(x_new)
                try:
                    f_new_val = float(f_new_raw)
                except (TypeError, ValueError):
                    f_new = np.asarray(f_new_raw, dtype=float)
                    f_new_val = f_new.item() if f_new.size == 1 else f_new
                break
            
        # Performance optimization: Replace in-place update with reassignment
        # so we can append `x` directly to history without a redundant `.copy()` allocation.
        x = x_new
        fx_val = f_new_val  # Cache the accepted function value for the next iteration
        history.append(x)
        
    return x, np.array(history)
