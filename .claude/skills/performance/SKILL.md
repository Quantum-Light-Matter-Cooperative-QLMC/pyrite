---
name: performance
description: Use when profiling, benchmarking, or investigating runtime, memory, GPU utilization, I/O, or plotting bottlenecks in cxr-mc.
---

# Performance

Measure a representative workload before proposing optimization.

## Workflow

1. Identify the narrowest real entry point and representative input.
2. Record the exact command, environment, data size, seed, and CPU/GPU backend.
3. Measure wall time and peak memory; include GPU utilization when relevant.
4. Profile cumulative and exclusive time, separating Python, NumPy/CuPy, I/O,
   and plotting costs.
5. Compare against the same baseline and report uncertainty or run-to-run spread.
6. Recommend changes only where the measured hotspot justifies them.

## Report

| Metric | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |

Include commands, workload, backend, repeated-run count, and the top hotspots.

## Common mistakes

- Comparing different inputs, backends, seeds, or warm-up states.
- Timing compilation, cache fill, or file loading in only one arm.
- Optimizing a microbenchmark that is not material to an end-to-end workload.
