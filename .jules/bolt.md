## 2024-07-06 - Avoid redundant array calculations via algebraic substitution
**Learning:** In numerical interior point solvers (like Primal-Dual QP), explicit calculations of complementarity residuals (e.g., `r_C = x * z`) allocate redundant arrays. This is problematic when the array is only used immediately afterward to compute a normalized residual (e.g., `r_C / x`).
**Action:** Mathematically substitute `r_C` into the subsequent equations. For example, replacing `-(x * z) / x + sigma / x` with `-z + sigma / x` entirely avoids allocating the O(n) array `r_C`, directly speeding up the tight optimization loop.

## 2024-07-06 - Optimize repeated array division with inversion and multiplication
**Learning:** In numerical solvers, dividing multiple large arrays by the same vector (e.g., computing `z / x` and `c / x`) incurs the high performance overhead of repeated floating-point divisions.
**Action:** When a loop requires multiple divisions by the same array, precompute the reciprocal (`inv_x = 1.0 / x`) once and replace subsequent divisions with multiplications (`z * inv_x`). Element-wise array multiplication is significantly faster than division on modern CPUs.

## 2026-07-29 - Avoid explicit step difference calculation in BFGS
**Learning:** In the BFGS optimization loop, calculating the step difference via `s = x_new - x` allocates a new $O(n)$ array and performs a subtraction on every iteration, despite `x_new` being derived immediately beforehand as `x_new = x + step`.
**Action:** Replace `s = x_new - x` with the mathematically equivalent `s = step` to completely bypass the redundant array allocation and subtraction, saving compute overhead in tight inner loops.

## 2024-08-01 - Avoid np.zeros when most blocks are immediately overwritten
**Learning:** Using `np.zeros` to pre-allocate large block matrices (like KKT systems) incurs a performance penalty if most of the matrix is immediately overwritten with dense sub-blocks. The OS must physically zero out the memory pages before they are written to.
**Action:** Use `np.empty` to allocate the memory without zero-initialization, write the dense sub-blocks directly, and explicitly zero out only the required empty sub-blocks (e.g., `KKT[n:, n:] = 0.0`).

## 2024-08-09 - Avoid redundant array calculations via sequential in-place modifications
**Learning:** In NumPy, advanced indexing (e.g., `A[:, idx]`) creates a new array copy. When performing element-wise multiplications on such slices, avoiding combining operations into a single line (e.g., `M = A[:, idx_a] * B[:, idx_b]`) which creates massive intermediate temporary arrays. Instead, use sequential in-place modifications (e.g., `M = A[:, idx_a]; M *= B[:, idx_b]`) to minimize memory allocations and improve inner loop speed.
**Action:** Replace single-line advanced indexing multiplications with sequential in-place modifications.

## 2024-08-09 - Precompute outer product in iterative loops
**Learning:** In iterative mathematical loops, avoid redundant outer product constructions (e.g., `M *= W[:, None] * W[None, :]`) by precomputing the outer matrix (`W_W = W[:, None] * W[None, :]`) outside the loop.
**Action:** Precompute the outer matrix outside the loop and use a single in-place multiplication inside the loop.

## 2025-02-28 - Strided Memory Assignment for KKT Diagonal Updates
**Learning:** In NumPy, updating the diagonal elements of a 2D matrix inside an iterative optimization loop (like updating `M = G + Z/X` in primal-dual interior point KKT matrices) using `KKT[np.diag_indices(n)] = ...` is extremely slow. This is because `np.diag_indices` returns a tuple of index arrays that trigger NumPy's advanced indexing, creating an intermediate array copy before performing the O(n) updates.
**Action:** When updating the main diagonal of a matrix repeatedly within a loop, use a flat strided view instead: `stride = n + m + 1`, `end = n * stride`, and update via `KKT.flat[:end:stride] = ...`. This approach bypasses advanced indexing overhead and avoids allocating a temporary copy of the diagonal, providing a ~3-4x speedup on diagonal block assignments.
