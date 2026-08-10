---
name: monte-carlo
description: "Use when PyRITE changes involve stochastic-kernel or RNG correctness: sampling, seed/stream behavior, Monte Carlo transport, and CPU/GPU reproducibility; use performance for benchmarks."
---

# Monte Carlo

## Rules

- Preserve RNG reproducibility: fix seeds in tests and document seed conventions.
- New stochastic processes require regression tests with deterministic seeds.
- Benchmark new kernels against the CPU baseline.
- Keep CPU and GPU implementations numerically consistent.

Own stochastic/RNG correctness only. Use `regression-testing` for general test
design, `performance` for measurement protocol, and `physics-review` for model
equations.
