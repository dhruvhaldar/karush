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

## 2024-03-31 - Grouping Outer Products for Rank-k Updates
**Learning:** Performing multiple `np.outer` calls and adding them together in a loop or equation is significantly slower than combining them into a single rank-k matrix multiplication using `np.column_stack` or similar constructions.
**Action:** When computing sums of outer products (e.g., `A @ B.T + C @ D.T`), combine them into a single matrix multiplication `np.column_stack([A, C]) @ np.column_stack([B, D]).T` to fully leverage BLAS Level 3 optimization.

## 2026-08-11 - Pre-allocating the KKT matrix inside iterative optimization algorithms
**Learning:** Using `np.block` and `np.concatenate` inside an iterative optimization loop (e.g., in Primal-Dual methods, SQP, or SDP Interior Point solvers) is a massive performance anti-pattern. These functions allocate new arrays and perform multiple data copies in every single iteration, even for the structural constraints blocks `A` and `A.T` that never change.
**Action:** Always pre-allocate the full block matrix `KKT_mat = np.zeros(...)` and right-hand side vector `rhs = np.empty(...)` *before* the loop. Assign static blocks like `A` and `A.T` once. Inside the loop, directly assign only the sub-matrices that change (e.g., `KKT_mat[:n, :n] = H`). This prevents memory allocation overhead and provides >8x speedup for the KKT matrix construction.

## 2026-10-24 - Caching Vector Dot Products in Iterative Algorithms
**Learning:** In iterative optimization algorithms like Conjugate Gradient, vector dot products (e.g., squared gradient norms) are often computed multiple times per iteration for both stopping criteria and update steps. This results in redundant O(n) computations.
**Action:** Calculate the squared norm or dot product once per iteration, cache the result, and reuse it across the iteration (e.g., using `np.sqrt(norm_sq) < tol` for stopping criteria). This simple caching prevents redundant linear time operations and improves overall algorithm speed.
## 2024-05-18 - Caching np.triu_indices in svec and smat
**Learning:** In frequently called numerical routines like `svec` and `smat` in `karush`, caching the output of `np.triu_indices(n)` and its associated boolean masks in module-level dictionaries based on matrix size `n` provides significant speedups by eliminating redundant array allocations inside iterative solver loops.
**Action:** Cache these indices in a module-level dictionary to avoid allocating arrays repeatedly.

## 2026-04-08 - In-place Diagonal Updates vs Full Matrix Addition
**Learning:** In iterative algorithms where the only changing part of a large dense matrix `M` is its diagonal (e.g. `M = G + np.diag(z / x)` where `G` is static), creating an explicit O(n^2) diagonal matrix using `np.diag()` and adding it to the full base matrix, and then assigning the full block into a larger KKT matrix is an unnecessary O(n^2) bottleneck.
**Action:** Pre-assign the static part (`KKT[:n, :n] = G`) outside the loop and precompute `diag_indices = np.diag_indices(n)` and `diag_G = np.diag(G)`. Inside the loop, update only the diagonal elements directly using advanced indexing (`KKT[diag_indices] = diag_G + z / x`). This reduces the update complexity from O(n^2) to O(n) and prevents memory allocations.
## 2024-05-24 - Optimize Barrier Gradient Computation
**Learning:** In NumPy, broadcasting followed by a sum over an axis (`np.sum(A[:, None] * B, axis=0)`) is significantly slower and uses more memory than a direct matrix-vector multiplication (`A @ B`), especially when dealing with constraints in optimization algorithms like the log-barrier method. The former relies on creating temporary arrays and Python-level looping constructs beneath the surface, while the latter hooks directly into highly optimized BLAS Level 2 operations.
**Action:** Always prefer `@` (or `np.dot`) over `np.sum` with broadcasting when accumulating gradients or performing weighted sums of vectors.
## 2024-05-15 - Vectorize repeated function calls over a list of matrices
**Learning:** Preprocessing logic like mapping a function (such as `svec` to vectorize matrices) over a list of numpy matrices using a Python loop or list comprehension introduces significant Python loop overhead. For $m$ constraint matrices, `[svec(Ai) for Ai in A_list]` requires repeatedly calling advanced indexing and NumPy operations in Python.
**Action:** Instead of iterating through a list, convert the list of 2D matrices into a 3D NumPy array using `np.array(A_list)` (which is very fast) and apply advanced indexing simultaneously on the stacked array (`A_stack[:, idx_i, idx_j]`). This leverages C-level vectorization for preprocessing multiple constraints simultaneously, providing a roughly ~10x speedup for this specific data transformation step in SDP solvers.

## 2024-05-19 - NumPy Scalar Broadcasting in Tight Loops
**Learning:** In iterative loops, using `scalar * np.ones(n)` creates unnecessary array allocations (O(n) memory) and initialization overhead every iteration. In operations like `(vector + scalar * np.ones(n)) / vector2`, NumPy can directly broadcast the scalar, effectively evaluating as `(vector + scalar) / vector2`.
**Action:** When updating vectorized right-hand sides or performing mathematical operations with scalars in optimization loops (like Primal-Dual methods), replace `np.ones(n)` multipliers with direct scalar broadcasting to eliminate memory allocation overhead.

## 2026-11-20 - Optimizing Symmetric Matrix Reconstruction (smat)
**Learning:** When reconstructing a symmetric matrix from its vectorized form using advanced indexing, allocating a dense matrix with `np.zeros`, assigning the upper triangle, dividing off-diagonal elements in the 2D representation, and finally mirroring them is suboptimal.
**Action:** Use `np.empty` instead of `np.zeros` to save allocation overhead. Scale the 1D vector first (`v_scaled = v.copy()`; `v_scaled[off_diag] /= np.sqrt(2)`), and then use bidirectional assignment (`M[idx_i, idx_j] = v_scaled` and `M[idx_j, idx_i] = v_scaled`) to construct the symmetric matrix directly. This reduces memory passes and yields a >2x speedup.

## 2026-04-18 - Optimize 2D Broadcasting Advanced Indexing
**Learning:** In NumPy, using 2D broadcasting advanced indexing (e.g., `X_inv[idx_a[:, None], idx_a[None, :]]`) to extract submatrices is memory intensive and slow compared to doing it sequentially in 1D.
**Action:** Replace 2D broadcasting with sequential 1D extractions (e.g., first extracting rows `X_inv_a = X_inv[idx_a]`, then columns `X_inv_a[:, idx_a]`) to reduce memory footprint and improve performance significantly.
