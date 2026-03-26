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
