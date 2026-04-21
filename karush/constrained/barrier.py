import numpy as np

def barrier_method(f, grad_f, hess_f, g_ineq, grad_g_ineq, x0, mu0=1.0, tol=1e-6, max_iter=20):
    """
    Log-barrier method for inequality constrained optimization.
    min f(x)
    s.t. g_ineq(x) <= 0
    
    This function uses a simple Newton's method for inner minimization.
    It does not explicitly use the line search for the inner minimization, 
    but assumes the function behaves well locally.
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
    if isinstance(mu0, bool) or not isinstance(mu0, (int, float, np.number)) or np.isnan(mu0) or mu0 <= 0:
        raise ValueError("Barrier parameter mu0 must be strictly positive.")
    mu = mu0
    history = [x.copy()]
    
    for k in range(max_iter):
        
        # Inner loop: Minimize barrier function
        # phi(x) = f(x) - mu * sum(log(-g_ineq(x)))
        # Solve grad phi(x) = 0 using Newton
        
        for inner_iter in range(10): # Fixed inner iterations
            # DoS Prevention: Convert function outputs to numpy arrays to prevent unhandled
            # AttributeError/TypeError exceptions if user functions return standard Python lists.
            g_val = np.asarray(g_ineq(x), dtype=float)
            
            # Check feasibility: if any constraint is violated, barrier is undefined.
            # In a robust implementation, we'd use line search to ensure feasibility.
            if np.any(g_val >= 0):
                # Backtrack or reduce step if we stepped out
                # For simplicity, break here if initial x0 was feasible but step went out
                break
            
            grad_g_val = np.asarray(grad_g_ineq(x), dtype=float)
            
            # Gradient of barrier
            # grad ( - mu * sum log(-g_i) ) = sum ( -mu/(-g_i) * (-grad_g_i) ) = sum ( -mu/g_i * grad_g_i )
            # Performance optimization: Replaced np.sum with a direct matrix-vector dot product
            # (BLAS Level 2 optimization) for faster computation and lower memory overhead.
            grad_phi = np.asarray(grad_f(x), dtype=float) + (-mu/g_val) @ grad_g_val
            
            if np.linalg.norm(grad_phi) < tol:
                break
                
            # Hessian of barrier
            # hess phi = hess f + sum( 1/g_i^2 * grad g_i * grad g_i^T ) + sum( -1/g_i * hess g_i )
            # Simplified: ignore hess g_i term (Gauss-Newton like approx)
            hess_phi = np.asarray(hess_f(x), dtype=float)

            # Performance optimization: Replace O(m) loop of O(n^2) np.outer calls
            # with a single vectorized matrix-matrix multiplication (BLAS Level 3).
            # We avoid creating a dense O(m^2) diagonal matrix by broadcasting weights.
            weights = mu / (g_val**2)
            hess_phi += (grad_g_val.T * weights) @ grad_g_val
            
            try:
                p = np.linalg.solve(hess_phi, -grad_phi)
            except np.linalg.LinAlgError:
                break
                
            # Line search to ensure x + alpha*p is feasible
            alpha = 1.0
            while True:
                x_new = x + alpha * p
                g_new_val = np.asarray(g_ineq(x_new), dtype=float)
                if np.all(g_new_val < 0):
                    # Ideally check Wolfe conditions on phi, but feasibility is key for barrier
                    x = x_new
                    break
                alpha *= 0.5
                if alpha < 1e-8:
                    break
        
        history.append(x.copy())
        
        if mu < tol:
            break
            
        mu *= 0.1
            
    return x, np.array(history)
