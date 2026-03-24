## 2026-03-23 - BFGS Inverse Hessian Update Complexity
**Learning:** The BFGS inverse Hessian update formula `H = V @ H @ V.T + rho_inv * np.outer(s, s)` involves matrix-matrix multiplications which scale at $O(n^3)$. In the standard implementation, this is a significant bottleneck.
**Action:** By expanding the formula using the Sherman-Morrison-Woodbury identity and leveraging matrix-vector multiplications, the update can be performed in $O(n^2)$ time: `H = H - rho_inv * (np.outer(s, Hy) + np.outer(Hy, s)) + rho_inv * (rho_inv * yHy + 1.0) * np.outer(s, s)` where `Hy = np.dot(H, y)` and `yHy = np.dot(y, Hy)`. This provides a massive speedup for large optimization problems without altering behavior.

## 2024-05-24 - Avoid Dense Matrix Products for Diagonal Matrices
**Learning:** In NumPy, constructing diagonal matrices using `np.diag()` and then multiplying them with other matrices (e.g., `np.diag(a) @ np.diag(b)`) creates full dense matrices and performs an O(n^3) multiplication. This is a severe performance anti-pattern in iterative algorithms like Interior Point Methods.
**Action:** Always compute the element-wise product of the diagonals directly (e.g., `np.diag(a * b)`) or use broadcasting `(a * b)[:, None]` instead of explicit matrix multiplications when dealing with diagonal matrices.
