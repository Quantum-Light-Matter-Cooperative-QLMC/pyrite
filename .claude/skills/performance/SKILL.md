---
name: performance
description: Use when profiling, benchmarking, or investigating PyRITE runtime, memory, GPU use, I/O, plotting, or remote latency.
---

# Performance

Measure representative workload before optimization.

1. Record command, environment, data size, seed, backend, warm-up/cache state,
   repeat count, wall time, peak memory, and GPU use when relevant.
2. Profile cumulative/exclusive time; separate Python, NumPy/CuPy, I/O,
   plotting, and remote transport.
3. Compare identical workloads and report spread/uncertainty.
4. Recommend changes only for measured material hotspots.

Report baseline, candidate, delta, commands, workload, backend, repeats, and top
hotspots. Never compare different inputs/backends/seeds or one warm arm against
one cold arm. Do not generalize microbenchmarks beyond measured scope.
