---
name: monte-carlo
description: Use when adding or modifying stochastic kernels, RNG-dependent sampling, seeded tests, CPU/GPU parity, Monte Carlo transport, or transport benchmarks in cxr-mc.
---

# Monte Carlo

## Rules

- Preserve RNG reproducibility: fix seeds in tests and document seed conventions.
- New stochastic processes require regression tests with deterministic seeds.
- Benchmark new kernels against the CPU baseline.
- Keep CPU and GPU implementations numerically consistent.
