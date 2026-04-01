## 2024-05-20 - Insecure PRNG in Randomized Rounding
**Vulnerability:** Predictable PRNG (`np.random.randn()`) was used in the `randomized_rounding` algorithm in `karush/convex/relaxations.py`.
**Learning:** Even mathematical and optimization algorithms must use secure PRNGs when deployed in contexts where predictability could be exploited (e.g., when the max-cut problem is applied to cryptography or network security). Using a global predictable PRNG can lead to deterministic outputs across sessions or predictable algorithmic behavior.
**Prevention:** Always use a securely seeded PRNG (e.g., `np.random.default_rng(secrets.randbits(128))`) for random decision-making in algorithms that may be applied to security-sensitive data or systems.
## 2024-05-24 - Secure PRNG Generation

**Vulnerability:** Predictable randomized rounding sequence due to using a cryptographically secure seed (`secrets.randbits`) with a non-cryptographic PRNG (`np.random.default_rng`).
**Learning:** Seeding a non-cryptographic PRNG with a secure seed does not make the generated sequence cryptographically secure. The internal state can still be predicted if enough outputs are observed.
**Prevention:** Use a true Cryptographically Secure PRNG (CSPRNG) like `secrets.SystemRandom()` directly for generating all random numbers when unpredictability is required.
## 2024-05-24 - Unsanitized Input to Solvers

**Vulnerability:** Core mathematical optimization functions lacked input validation for `NaN` and `Inf` values. Passing tainted data into solvers can cause unhandled exceptions during matrix operations or lead to infinite loops inside the line search mechanisms.
**Learning:** In scientific and optimization code, algorithms heavily rely on numeric stability. Unsanitized input propagates rapidly through matrix inversion and updates, acting as an application-level DoS vector or corrupting calculations silently.
**Prevention:** Validate input arrays at the entry point of all solver functions (e.g., using `np.isfinite()`) to ensure the problem data is well-formed before starting algorithmic loops.

## 2023-10-27 - [Configuration Validation for DoS Prevention]
**Vulnerability:** Missing validation for optimization parameters `tol` and `max_iter` could allow users to pass negative tolerances or extremely large/invalid iteration counts, leading to infinite loops or memory exhaustion (due to unbounded history appending).
**Learning:** In applied math libraries like `karush`, parameter validation is critical because the optimization loops depend on them for termination. Leaving them unchecked can result in silent data corruption or DoS vectors.
**Prevention:** Always validate configuration parameters such as `tol > 0` and `max_iter` as positive integers before starting optimization loops.

## 2024-05-20 - Missing bounds checks for tolerance and iterations
**Vulnerability:** DoS (Denial of Service) vulnerability due to missing bounds checking on `tol` (tolerance) and `max_iter` (maximum iterations) parameters in iterative optimization algorithms.
**Learning:** Mathematical solvers often use `tol` to evaluate convergence and `max_iter` to prevent infinite loops. If these inputs aren't sanitized, malicious users could provide a negative `tol` (making convergence impossible) or an enormous `max_iter`, causing the server to exhaust CPU resources in an infinite or extremely long-running loop.
**Prevention:** Always validate that `tol` is strictly positive (`tol > 0`) and `max_iter` is a reasonably bounded positive integer before entering iterative loops.

## 2024-05-24 - Missing bounds checks for integer parameters
**Vulnerability:** Application DoS vulnerability due to missing type and bounds checking on numeric parameters (like `num_trials`).
**Learning:** When iterative limits or sizes depend on user input, failure to validate the input type and bounds can lead to OOM crashes or unhandled exceptions.
**Prevention:** Always validate parameters (like `num_trials > 0`) before using them in algorithms.
