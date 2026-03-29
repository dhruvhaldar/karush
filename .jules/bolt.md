## 2026-03-23 - BFGS Inverse Hessian Update Complexity
**Learning:** The BFGS inverse Hessian update formula `H = V @ H @ V.T + rho_inv * np.outer(s, s)` involves matrix-matrix multiplications which scale at $O(n^3)$. In the standard implementation, this is a significant bottleneck.
**Action:** By expanding the formula using the Sherman-Morrison-Woodbury identity and leveraging matrix-vector multiplications, the update can be performed in $O(n^2)$ time: `H = H - rho_inv * (np.outer(s, Hy) + np.outer(Hy, s)) + rho_inv * (rho_inv * yHy + 1.0) * np.outer(s, s)` where `Hy = np.dot(H, y)` and `yHy = np.dot(y, Hy)`. This provides a massive speedup for large optimization problems without altering behavior.

## 2024-05-24 - Avoid Dense Matrix Products for Diagonal Matrices
**Learning:** In NumPy, constructing diagonal matrices using `np.diag()` and then multiplying them with other matrices (e.g., `np.diag(a) @ np.diag(b)`) creates full dense matrices and performs an O(n^3) multiplication. This is a severe performance anti-pattern in iterative algorithms like Interior Point Methods.
**Action:** Always compute the element-wise product of the diagonals directly (e.g., `np.diag(a * b)`) or use broadcasting `(a * b)[:, None]` instead of explicit matrix multiplications when dealing with diagonal matrices.

## 2025-05-19 - Vectorized Randomized Rounding Matrix Multiplications
**Learning:** In randomized rounding algorithms for Max-Cut and similar SDP relaxations, performing multiple matrix-vector multiplications in a loop (e.g., `L @ r` where `r` is a vector) is inefficient due to Python loop overhead and lack of BLAS Level 3 optimization.
**Action:** Replace the loop with a single matrix-matrix multiplication (e.g., `L @ R` where `R` is a matrix of standard normal samples). This yields a significant performance improvement by fully utilizing vectorized NumPy operations and optimized linear algebra libraries.

## 2026-06-15 - Vectorizing SDP Hessian Construction
**Learning:** In interior point methods for semidefinite programming, constructing the barrier Hessian matrix `H` by iterating over symmetric basis matrices `D` and performing `mu * svec(X_inv @ D @ X_inv)` creates an enormous performance bottleneck because it requires an $O(n^5)$ loop due to $O(n^3)$ nested operations inside the `dim_vec` loop (where `dim_vec = O(n^2)`). Attempting to vectorize this using a full Kronecker product `K = np.kron(X_inv, X_inv)` and `H = Q.T @ K @ Q` is a massive anti-pattern that creates an $O(n^6)$ algorithm and worse performance.
**Action:** Replace the explicit loop with a true $O(n^4)$ direct algebraic computation. The individual elements of `H_mat` map exactly to the combinations of elements of `X_inv`. By precomputing the index mappings for the upper triangular parts, `H_mat` can be computed directly using Numpy advanced indexing and broadcasting: `M = Vac * Vbd + Vad * Vbc`, followed by proper scaling. This provides huge speedups without $O(n^6)$ overhead.

## 2025-06-25 - Vectorizing Symmetric Matrix Representation (svec/smat)
**Learning:** In interior point methods for Semidefinite Programming, mapping symmetric matrices to vectors (`svec`) and back (`smat`) using nested Python loops (`for i in range(n): for j in range(i, n):`) is a significant bottleneck. Python loop overhead for element-wise operations on arrays creates terrible performance.
**Action:** Replace nested loops with vectorized operations using NumPy advanced indexing, specifically `np.triu_indices(n)` to extract and assign values in bulk, combined with boolean masks for scaling off-diagonal elements. This provides a dramatic speedup (e.g., ~80x for 200x200 matrices).
