# Monte Carlo rules for stochastic kernels in cxr_mc

# Use this skill when

Use this skill when adding or modifying any stochastic kernel, RNG-dependent sampling, or transport algorithm.

# Rules

- Preserve RNG reproducibility — fix seeds in tests; document seed conventions.
- New stochastic processes require regression tests with deterministic seeds.
- Always benchmark new kernels against the CPU baseline.
- Keep CPU and GPU implementations numerically consistent.
