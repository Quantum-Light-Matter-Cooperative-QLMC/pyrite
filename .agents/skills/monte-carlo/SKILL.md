---
name: monte-carlo
description: Work on stochastic kernels, RNG-dependent sampling, or Monte Carlo transport algorithms in cxr_mc. Use when adding or modifying stochastic processes, seeded tests, CPU/GPU kernel parity, or transport benchmarks.
---

# Monte Carlo

## Rules

- Preserve RNG reproducibility: fix seeds in tests and document seed conventions.
- New stochastic processes require regression tests with deterministic seeds.
- Benchmark new kernels against the CPU baseline.
- Keep CPU and GPU implementations numerically consistent.
