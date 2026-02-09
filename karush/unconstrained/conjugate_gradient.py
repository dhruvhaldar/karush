import numpy as np

def conjugate_gradient(f, grad_f, x0, tol=1e-6, max_iter=100):
    """
    Nonlinear Conjugate Gradient method (Fletcher-Reeves).
    """
    x = np.array(x0, dtype=float)
    history = [x.copy()]
    g = grad_f(x)
    p = -g
    
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            break
            
        # Line search
        alpha = 1.0
        rho = 0.5
        c = 1e-4
        
        # Backtracking line search ensuring sufficient decrease
        while f(x + alpha * p) > f(x) + c * alpha * np.dot(g, p):
            alpha *= rho
            if alpha < 1e-10: break # Avoid infinite loop
            
        x_new = x + alpha * p
        g_new = grad_f(x_new)
        
        # Fletcher-Reeves update
        beta = np.dot(g_new, g_new) / (np.dot(g, g) + 1e-10)
        p = -g_new + beta * p
        
        g = g_new
        x = x_new
        history.append(x.copy())
        
    return x, np.array(history)
