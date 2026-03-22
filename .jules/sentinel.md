## 2024-05-20 - Insecure PRNG in Randomized Rounding
**Vulnerability:** Predictable PRNG (`np.random.randn()`) was used in the `randomized_rounding` algorithm in `karush/convex/relaxations.py`.
**Learning:** Even mathematical and optimization algorithms must use secure PRNGs when deployed in contexts where predictability could be exploited (e.g., when the max-cut problem is applied to cryptography or network security). Using a global predictable PRNG can lead to deterministic outputs across sessions or predictable algorithmic behavior.
**Prevention:** Always use a securely seeded PRNG (e.g., `np.random.default_rng(secrets.randbits(128))`) for random decision-making in algorithms that may be applied to security-sensitive data or systems.
