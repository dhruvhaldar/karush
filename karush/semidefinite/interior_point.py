import numpy as np

def svec(M):
    """
    Symmetric vectorization operator.
    Multiplies off-diagonal elements by sqrt(2) to preserve inner product.
    Tr(A @ B) = svec(A).T @ svec(B)
    """
    n = M.shape[0]
    v = []
    for i in range(n):
        for j in range(i, n):
            if i == j:
                v.append(M[i, j])
            else:
                v.append(M[i, j] * np.sqrt(2))
    return np.array(v)

def smat(v, n):
    """
    Inverse of svec.
    """
    M = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i, n):
            val = v[idx]
            if i == j:
                M[i, j] = val
            else:
                val /= np.sqrt(2)
                M[i, j] = val
                M[j, i] = val # Symmetric
            idx += 1
    return M

def solve_sdp_barrier(C, A_list, b, X0, initial_mu=1.0, tol=1e-6, max_iter=20):
    """
    Barrier method for Semidefinite Programming.
    min Tr(C @ X)
    s.t. Tr(A_i @ X) = b_i, i=0..m-1
         X > 0 (Positive Definite)
         
    A_list: list of constraint matrices A_i.
    b: list or array of scalars b_i.
    X0: Initial feasible point (X0 > 0 and satisfies constraints).
    
    This is a basic implementation for small-scale SDPs.
    """
    X = np.array(X0, dtype=float)
    mu = initial_mu
    m = len(b)
    n = X.shape[0]
    dim_vec = n * (n + 1) // 2
    
    # Precompute vectorized constraints
    # trace(A_i @ X) = svec(A_i).T @ svec(X)
    A_mat = np.array([svec(Ai) for Ai in A_list]) # m x dim_vec
    
    for k in range(max_iter):
        
        if mu < tol:
            break
            
        # Inner Newton loop for centering
        for inner in range(5):
            try:
                X_inv = np.linalg.inv(X)
            except np.linalg.LinAlgError:
                return X
            
            # Gradient of barrier objective: C - mu * X^-1
            Grad = C - mu * X_inv
            grad_vec = svec(Grad)
            
            # Hessian of barrier objective: H(D) = mu * X^-1 @ D @ X^-1
            H_mat = np.zeros((dim_vec, dim_vec))
            
            # Construct Hessian matrix
            for col in range(dim_vec):
                e_col = np.zeros(dim_vec)
                e_col[col] = 1.0
                D = smat(e_col, n)
                
                res = mu * X_inv @ D @ X_inv
                res_vec = svec(res)
                H_mat[:, col] = res_vec
            
            # KKT System
            residuals = A_mat @ svec(X) - np.array(b)
            
            KKT_lhs = np.block([
                [H_mat, A_mat.T],
                [A_mat, np.zeros((m, m))]
            ])
            
            rhs = np.concatenate([-grad_vec, -residuals])
            
            try:
                sol = np.linalg.solve(KKT_lhs, rhs)
            except np.linalg.LinAlgError:
                break
                
            dx_vec = sol[:dim_vec]
            dX = smat(dx_vec, n)
            
            # Line search
            alpha = 1.0
            step_accepted = False
            for ls in range(10):
                X_new = X + alpha * dX
                # Check PD
                try:
                    np.linalg.cholesky(X_new)
                    step_accepted = True
                    X = X_new
                    break
                except np.linalg.LinAlgError:
                    alpha *= 0.5
            
            if not step_accepted:
                break
                
            if np.linalg.norm(dx_vec) < tol:
                break
        
        mu *= 0.5
        
    return X
