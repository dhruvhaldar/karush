import numpy as np

def bfgs_method(f, grad_f, x0, tol=1e-6, max_iter=100):
    """
    BFGS Quasi-Newton method.
    """
    x = np.array(x0, dtype=float)
    n = len(x)
    I = np.eye(n)
    H = I  # Inverse Hessian approximation
    history = [x.copy()]
    
    g = grad_f(x)
    
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
            
        p = -np.dot(H, g)
        
        # Line search
        alpha = 1.0
        rho = 0.5
        c = 1e-4
        
        # Simple backtracking line search
        while f(x + alpha * p) > f(x) + c * alpha * np.dot(g, p):
            alpha *= rho
            if alpha < 1e-10: 
                break
            
        x_new = x + alpha * p
        g_new = grad_f(x_new)
        
        s = x_new - x
        y = g_new - g
        
        # BFGS update
        ys = np.dot(y, s)
        if ys > 1e-10:
            rho_inv = 1.0 / ys
            V = I - rho_inv * np.outer(s, y)
            H = np.dot(V, np.dot(H, V.T)) + rho_inv * np.outer(s, s)
        else:
            # If ys is small, update might be unstable, so skip or restart H
            pass
            
        x = x_new
        g = g_new
        history.append(x.copy())
        
    return x, np.array(history)
