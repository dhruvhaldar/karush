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

## 2024-05-30 - Sequential 1D Extractions vs 2D Broadcasting
**Learning:** In NumPy operations within `karush`, combining 2D broadcasting advanced indexing (e.g., `X_inv[idx_a[:, None], idx_a[None, :]]`) to extract submatrices creates large intermediate arrays and represents a significant performance bottleneck in memory and computation, as well as a more implicit O(n^4) computation instead of a true memory-efficient sequential one.
**Action:** Replace 2D broadcasting advanced indexing with sequential 1D extractions (e.g., first extracting rows `X_inv_a = X_inv[idx_a]`, then extracting columns `Vac = X_inv_a[:, idx_a]`). This heavily reduces memory footprint and yields >50% improvement in execution time for large arrays while obtaining the identical result.

## 2024-05-10 - Avoid redundant array copying with Advanced Indexing
**Learning:** In NumPy, advanced indexing (e.g., `M[idx_i, idx_j]`) always returns a new copy of the data, not a view. Calling `.copy()` on the result of advanced indexing (e.g., `M[idx_i, idx_j].copy()`) creates a second, entirely redundant memory allocation which wastes both time and memory, especially for large arrays in hot loops. Additionally, converting an existing NumPy array using `np.array(b)` inside a loop when `b` was already safely converted to an array earlier creates an unnecessary copy each iteration.
**Action:** Remove explicit `.copy()` calls when they directly follow advanced indexing operations. Similarly, hoist array conversions (e.g. `np.asarray`) outside of iterative loops and avoid `np.array()` wrappers around known arrays within loops.

## 2024-06-20 - Cache Function Evaluations in Line Search Loops
**Learning:** During iterative optimization algorithms (like Newton, BFGS, or Conjugate Gradient), the backtracking line search evaluates the objective function `f(x)` inside a `while` loop condition (e.g., `f(x + alpha*p) > f(x) + ...`). Re-evaluating the current state `f(x)` on every iteration of the line search or on every outer step is entirely redundant because `x` does not change during the line search, and the successful `f_new` from the previous step becomes the new `f(x)`. This redundancy wastes nearly 50% of function evaluations, creating a large bottleneck when `f(x)` is computationally expensive.
**Action:** Evaluate `f(x)` once outside the main optimization loop. Inside the loop, compute `f_new = f(x + alpha*p)` within the line search, and once the step is accepted, simply cache the result with `fx = f_new` for the next iteration. This eliminates half of the objective function calls and drastically improves performance.

## 2026-11-20 - Cache Constraint Evaluations in Barrier Methods
**Learning:** In the inner Newton loop of log-barrier methods, the inequality constraint function `g_ineq(x)` is evaluated in the backtracking line search to ensure feasibility. Re-evaluating `g_ineq(x)` at the start of every iteration is entirely redundant because `x` does not change if it was just accepted, and the successful `g_new_val` from the previous line search is the current state. This redundancy wastes nearly 50% of constraint evaluations, creating a large bottleneck when `g_ineq(x)` is computationally expensive.
**Action:** Evaluate `g_ineq(x)` once outside the main optimization loop. Inside the line search, compute `g_new_val = g_ineq(x + alpha*p)`. Once the step is accepted, simply cache the result with `g_val = g_new_val` for the next iteration. This eliminates half of the constraint function calls and drastically improves performance.

## 2026-11-20 - SDP Memory Efficient In-Place Hessian Scaling
**Learning:** In computing the Hessian matrix for Semi-Definite Programming (SDP), explicitly defining `W_mat = W_svec[:, None] * W_svec[None, :]` and multiplying it by `M = Vac * Vbd + Vad * Vbc` is memory intensive, allocating dual $O(n^4)$ arrays inside inner iterations, triggering major memory pressure and performance drop due to reallocation logic.
**Action:** Replace explicit dense memory combination matrices by computing `M` completely in-place `M = Vac * Vbd; M += Vad * Vbc`, and use consecutive broadcasting across 1D slices `M *= W_svec[:, None]; M *= W_svec[None, :]` for identical behavior without additional memory allocations.

## 2024-05-25 - Avoid np.all() for scalar line searches
**Learning:** During backtracking line searches in iterative solvers (Conjugate Gradient, BFGS, Newton), evaluating `np.all(f_new <= fx + alpha * expected_decrease)` inside the while loop is extremely slow for scalar functions, which is the 99% use case. Evaluating truth on arrays is much faster if we extract the scalar value first using `.item()`. However, we cannot entirely drop `np.asarray` and fallback logic because some edge cases or internal tests pass lists or vector-valued functions to line searches for testing.
**Action:** Retain `np.asarray` for type safety, but check the array size `f_new.size == 1`. If it is a scalar, extract it with `.item()` and use pure Python scalar inequalities (e.g. `f_new_val <= fx_val + alpha * expected_decrease`). Fall back to `np.all` only for multi-dimensional testing cases. This yields roughly a ~10-15% speedup across all unconstrained solvers.

## 2024-05-25 - Optimize BFGS rank-2 memory allocations
**Learning:** During BFGS updates, `H = H + np.dot(U, V.T)` explicitly allocates a completely new $n \times n$ dense matrix `H` every iteration in the loop, creating memory pressure.
**Action:** Use an in-place update `H += np.dot(U, V.T)` to reuse the existing allocated space, saving time and memory allocations without affecting algorithmic behavior.
## 2024-05-24 - Pre-allocate KKT Block Matrices in Iterative Solvers
**Learning:** In iterative solvers like SQP, replacing convenience functions that allocate memory (like calling a standalone QP solver that constructs a new KKT block matrix every iteration) with inline, pre-allocated buffers provides significant performance benefits. Repeated allocation of $O((n+m)^2)$ matrices creates unnecessary overhead.
**Action:** When implementing iterative optimization algorithms in `karush`, pre-allocate large block matrices outside the loop, evaluate initial dimensions outside the loop if needed, and update sub-blocks in-place before calling `np.linalg.solve`.
## 2024-05-24 - Avoid Redundant Function Evaluations in Memory Optimizations
**Learning:** When trying to optimize memory allocations (like pre-allocating a matrix outside an iteration loop), it's easy to accidentally introduce a severe performance regression if computing the required matrix dimensions forces an extra evaluation of a user-provided function (e.g., a Jacobian `grad_h(x)`). In numerical optimization, function evaluations are typically the most computationally expensive operations, far outweighing the cost of memory allocation.
**Action:** Do not call user-provided functions (like `grad_h(x)`) outside of a loop just to determine array dimensions. Instead, lazily pre-allocate memory *inside* the loop on the first iteration (`if k == 0:`) using the natively evaluated array dimensions. This perfectly combines memory optimization with strict evaluation economy.
## 2024-05-24 - Validate Array Shapes Before Memory Allocation
**Learning:** In optimization problems, a single constraint often causes user-provided functions to return a 1D gradient array (e.g., shape `(n,)`) instead of a 2D matrix (shape `(1, n)`). If array shapes are extracted (like `A.shape[0]`) to calculate dimensions for pre-allocated memory buffers (like a KKT matrix) *before* the input array is properly coerced to 2D (e.g., using `.reshape(1, -1)`), the resulting dimension will incorrectly be `n` instead of `1`, causing fatal linear algebra errors (e.g., singular matrices).
**Action:** Always ensure array inputs are properly dimensioned and validated (e.g., 2D matrices) *before* using their `shape` attributes to determine the size of pre-allocated buffers.

## 2024-06-25 - Avoid Matrix Allocations using svec Linearity
**Learning:** In the inner loops of Semidefinite Programming (SDP) barrier methods, explicitly evaluating gradient matrices like `Grad = C - mu * X_inv` and then applying symmetric vectorization `svec(Grad)` allocates a redundant, dense $O(n^2)$ matrix and performs costly matrix subtraction on every iteration. Since `svec` is a strictly linear operator, the vectorization and subtraction can be mathematically commuted.
**Action:** Use the linearity of `svec` to optimize algebraic expressions before implementation. Precompute constants like `svec_C = svec(C)` outside the loop. Inside the loop, replace the dense matrix evaluation `svec(C - mu * X_inv)` with the equivalent vector expression `svec_C - mu * svec(X_inv)`. This avoids memory allocation entirely for the target matrix, significantly improving inner-loop speed.

## 2024-05-30 - SDP Dynamic State Caching using svec Linearity
**Learning:** In iterative solvers tracking a dynamic matrix state `X`, explicitly evaluating `svec(X)` inside the inner iteration loops allocates a dense vector every time and evaluates matrix extractions. This creates unnecessary memory allocation and redundant computation.
**Action:** Exploit the linearity of `svec`. Precompute the initial state `svec_X = svec(X)` once outside the loop. Inside the loop, replace evaluations of `svec(X)` with `svec_X`, and directly update this cached vector state (`svec_X += alpha * dx_vec`) parallel to the matrix update (`X += alpha * dX`). This completely eliminates inner-loop memory allocations for dynamic tracking.

## 2024-05-21 - Optimize Tight Inner Loop Scalar Function Evaluations
**Learning:** Unconditionally wrapping scalar results returned from user-provided mathematical functions (e.g. `f(x)`) with `np.asarray(...)` introduces significant Python overhead (on the order of 1us per call). In tight inner loops, such as line search backtracking where `f(x + alpha * p)` is called repeatedly, this overhead dominates execution time for fast-evaluating functions, making the algorithm slow. Converting the result with `float()` is much faster but crashes if the function safely returns a vector.
**Action:** Replace `np.asarray()` in tight loop function evaluations with a `try: float(f_raw)` block. This creates a "fast path" for scalar evaluations while falling back to `np.asarray` handling for vectors, achieving the speed of `float()` without sacrificing type safety for vector-valued inputs.
## 2026-11-20 - Unnecessary Deep Copy in Iterative Solver Histories
**Learning:** In iterative solvers, history lists often track state variables like `x` at each iteration using `history.append(x.copy())`. However, if `x` is updated via a reassignment that natively allocates a new array (e.g. `x_new = x + alpha * p; x = x_new`), calling `.copy()` is completely redundant and causes a >10x slowdown in the tracking loop due to duplicate memory allocation.
**Action:** Only use `.copy()` when appending variables that were modified strictly in-place (e.g. `x += p`). Remove explicit `.copy()` calls when the state variable has just been reassigned to a newly created array.
## 2024-05-28 - Fast 3D Array Pre-allocation for Matrix Lists
**Learning:** In optimization problem setups, generating a list of constraint matrices (e.g., `A_list`) using a Python loop that iteratively calls `np.zeros((n, n))` is extremely slow for large `n` due to sequential Python overhead and repeated memory allocation calls.
**Action:** Pre-allocate a 3D NumPy array for the entire stack of matrices at once (e.g., `A_stack = np.zeros((m, n, n))`), use advanced indexing or broadcasting to fill the non-zero elements, and then convert it to a list using `list(A_stack)`. This leverages C-level bulk memory allocation and vectorization, yielding >10x speedup for initialization steps.

## 2024-06-08 - Optimized SDP Barrier Hessian Construction
**Learning:** In the SDP barrier method, the construction of the Hessian matrix `H_mat` involved explicitly allocating multiple intermediate matrices (`Vac`, `Vbd`, `Vad`, `Vbc`) using advanced indexing and storing them in memory before doing element-wise multiplication. This memory allocation overhead caused a significant performance bottleneck due to large temporary variables ($O(n^4)$ storage).
**Action:** When performing complex matrix updates that can be factored into element-wise operations with advanced indexing, evaluate the multiplication directly (`X_inv_a[:, idx_a] * X_inv_b[:, idx_b]`) without assigning the intermediate arrays to explicit local variables. This avoids keeping large dense matrices explicitly in memory and leverages NumPy's efficient internal buffers.

## 2024-06-11 - Pre-allocate and assign matrices to avoid np.column_stack overhead
**Learning:** In iterative solver loops, using `np.column_stack` inside the loop causes repeated memory allocation overhead, which becomes a bottleneck. Pre-allocating `np.empty((n, 2))` outside the loop and assigning columns directly `U[:, 0] = ...` provides a measurable speedup.
**Action:** Pre-allocate memory outside hot loops and assign values in-place rather than relying on NumPy concatenation functions like `np.column_stack` or `np.block`.
## 2024-06-28 - Optimize Matrix Lists Vectorization with Loop Fill
**Learning:** In constraint preprocessing algorithms (like symmetric vectorization `svec`), converting a Python list of matrices into a dense 3D stack via `np.array(A_list)` creates a massive, temporary $O(m \times n^2)$ array. This causes a huge memory spike and often crashes with unhandled Out-Of-Memory exceptions before filtering operations (like `A_stack[:, idx_i, idx_j]`) can extract the needed 2D matrix.
**Action:** Replace `np.array(A_list)` stacking followed by advanced 3D indexing. Instead, pre-allocate the target 2D matrix with `np.empty((m, dim_vec))` and populate it directly inside a fast `for` loop (`A_mat[i, :] = A_list[i][idx_i, idx_j]`). This completely avoids allocating the massive intermediate 3D block, saving substantial peak memory without sacrificing speed.

## 2024-06-29 - Avoid O(n^3) 3D array pre-allocation for constraint matrices
**Learning:** In mathematical problem setups (like SDP relaxations), pre-allocating a dense 3D NumPy array (e.g., `A_stack = np.zeros((n, n, n))`) to generate constraint matrices forces $O(n^3)$ memory allocation and causes severe memory spikes or OOM crashes.
**Action:** Instead, use a loop with repeated `np.zeros()` calls (e.g., `[np.zeros((n, n)) for _ in range(n)]`), which leverages OS-level lazy memory allocation to significantly reduce the resident set size and overhead.

## 2024-07-02 - In-place search direction updates in Conjugate Gradient
**Learning:** In iterative solvers like Conjugate Gradient, calculating the new search direction vector with `p_new = -g_new + beta * p` explicitly allocates a new array on every iteration. This redundant memory allocation creates measurable overhead in tight inner loops.
**Action:** Replace the explicit array allocation with in-place modifications to the existing search direction vector (`p *= beta`, `p -= g_new`). This avoids memory allocation overhead inside the inner loop, providing a significant speedup for large dimensional problems.
