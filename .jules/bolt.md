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
