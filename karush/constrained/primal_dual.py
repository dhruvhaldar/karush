import numpy as np

def primal_dual_qp(G, c, A, b, x0, z0, tol=1e-6, max_iter=20):
    """
    Primal-Dual Interior Point Method for Convex QP in standard form:
    min 1/2 x^T G x + c^T x
    s.t. A x = b
         x >= 0
         
    This is a simplified implementation.
    G must be positive semidefinite.
    """
    # Security Enhancement: Add input sanitization to reject non-finite values (NaN/Inf)
    # which can lead to silent data corruption, infinite loops in solvers, or unhandled exceptions.
    if not np.all(np.isfinite(G)) or not np.all(np.isfinite(c)):
        raise ValueError("Input arrays G and c must contain only finite numbers.")
    if not np.all(np.isfinite(A)) or not np.all(np.isfinite(b)):
        raise ValueError("Constraint arrays A and b must contain only finite numbers.")
    if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(z0)):
        raise ValueError("Initial points x0 and z0 must contain only finite numbers.")
    if tol <= 0:
        raise ValueError("Tolerance tol must be strictly positive.")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("Maximum iterations max_iter must be a positive integer.")
    if max_iter > 10000:
        raise ValueError("Maximum iterations max_iter exceeds safe limit.")

    x = np.array(x0, dtype=float)
    # Actually for standard form, x itself is constrained >= 0.
    # Standard form usually: min c'x, Ax=b, x>=0. For QP: min 0.5 x'Gx + c'x, Ax=b, x>=0.
    # KKT:
    # G x + c - A^T y - z = 0
    # A x - b = 0
    # X Z e = mu e  (perturbed complementarity)
    # x, z >= 0
    
    n = len(x)
    m = len(b)
    
    y = np.zeros(m) # Dual for equality
    z = np.array(z0, dtype=float) # Dual for inequality x >= 0
    
    history = [x.copy()]
    
    # Performance optimization: Replace np.block and np.concatenate with pre-allocation
    # outside the loop. In the loop, only update the blocks that change.
    KKT = np.zeros((n + m, n + m))
    KKT[:n, n:] = -A.T
    KKT[n:, :n] = A
    rhs = np.empty(n + m)

    for k in range(max_iter):
        # Residuals
        r_L = G @ x + c - A.T @ y - z
        r_A = A @ x - b
        r_C = x * z # complementarity
        
        mu = np.dot(x, z) / n
        sigma = 0.5 # Centering parameter
        
        # Solve Newton system
        # [ G   -A^T  -I ] [ dx ]   [ -r_L ]
        # [ A    0     0 ] [ dy ] = [ -r_A ]
        # [ Z    0     X ] [ dz ]   [ -r_C + sigma*mu*e ]
        
        # Eliminate dz: dz = X^-1 ( -r_C + sigma*mu*e - Z dx )
        # Substitute into first equation:
        # G dx - A^T dy - X^-1 ( ... - Z dx ) = -r_L
        # ( G + X^-1 Z ) dx - A^T dy = -r_L + X^-1 ( -r_C + sigma*mu*e )
        
        # Performance optimization: Avoid creating dense O(n^2) diagonal matrices
        # and performing O(n^3) matrix multiplication. Instead, compute the
        # diagonal elements directly in O(n) and use vectorized operations.
        # This replaces `X_inv = np.diag(1/x)`, `Z = np.diag(z)`, `M = G + X_inv @ Z`
        M = G + np.diag(z / x)
        
        # Avoid `X_inv @ vector` which is O(n^2) by using element-wise division O(n)
        rhs_1 = -r_L + ( -r_C + sigma * mu * np.ones(n) ) / x
        rhs_2 = -r_A
        
        # System:
        # [ M  -A^T ] [ dx ] = [ rhs_1 ]
        # [ A   0   ] [ dy ]   [ rhs_2 ]
        
        KKT[:n, :n] = M
        rhs[:n] = rhs_1
        rhs[n:] = rhs_2
        
        try:
            sol = np.linalg.solve(KKT, rhs)
        except np.linalg.LinAlgError:
            break
            
        dx = sol[:n]
        dy = sol[n:]
        dz = ( -r_C + sigma * mu * np.ones(n) - z * dx ) / x
        
        # Line search to keep x, z > 0
        alpha_p = 1.0
        alpha_d = 1.0
        
        idx_x = dx < 0
        if np.any(idx_x):
            alpha_p = min(1.0, 0.99 * np.min(-x[idx_x] / dx[idx_x]))
            
        idx_z = dz < 0
        if np.any(idx_z):
            alpha_d = min(1.0, 0.99 * np.min(-z[idx_z] / dz[idx_z]))
            
        x += alpha_p * dx
        y += alpha_d * dy
        z += alpha_d * dz
        
        history.append(x.copy())
        
        if np.linalg.norm(r_L) < tol and np.linalg.norm(r_A) < tol and np.dot(x, z) < tol:
            break
            
    return x, np.array(history)
