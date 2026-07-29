# Compute performance optimization

Branch: `feature/compute-performance-optimization`

TODO scope: former P1 item 1, remote compute utilization and stable non-OOM
throughput.

## Goal

Use representative remote measurements to find and remove material bottlenecks.
Improve end-to-end throughput while keeping CPU, GPU, host memory, and VRAM
within stable limits. Keep the GPU fed when feed-wait evidence identifies host
transport as the constraint.

The requested roughly 85% utilization is a capacity target, not an independent
success metric: unused RAM is not a defect, and forcing every resource to the
same percentage can reduce throughput or OOM margin.

## Current evidence

- `--performance-profile` emits `cxr.performance.v1` NDJSON and
  `cxr profile analyze` excludes incomplete sessions.
- Complete `sub_100keV` sessions attributed about 42.4% of aggregate wall time
  to checkpointing and 57.1% to spectrum work; driver transport feed-wait was
  about 0.27%.
- Those observations argue against CPU transport starvation in that cohort.
  They do not yet distinguish checkpoint contamination, serialized spectrum
  work, synchronization, or fragmented GPU kernels.
- `Sweep.spec_chunk` and `Sweep.brem_chunk` are the existing fine-grained VRAM
  controls. `run_cases` intentionally uses one CUDA context with CPU transport
  pipelined behind serial GPU spectrum/bremsstrahlung work.

## Decisions

- Measure matched workloads before changing scheduling or memory policy.
- Compare identical profile, material grid, seed/backend, allocation, worker
  count, chunks, warm-up/cache state, and checkpoint state. Report repeats and
  spread.
- Run heavy or GPU-bound experiments through `cxr remote`; never locally.
- Do not add one GPU process per case. Consider long-lived GPU sharding only if
  measured work proves the single context is the material limiter and VRAM
  admission can be bounded.
- Preserve checkpoint correctness and resumability. Optimize write frequency or
  representation only after isolating checkpoint cost from compute activity.
- Treat OOM absence, throughput, tail latency, checkpoint recovery, and stable
  resource headroom as joint acceptance criteria.

## Owning paths

- GPU/CPU pipeline: `src/cxr_mc/montecarlo/runner.py`
- Spectrum and chunking: `src/cxr_mc/montecarlo/spectrum.py`
- Checkpoint callback/write path: `src/cxr_mc/run.py`,
  `src/cxr_mc/_checkpoint_store.py`, `src/cxr_mc/_checkpoint_io.py`
- Telemetry and analysis: `src/cxr_mc/performance_profile.py`,
  `src/cxr_mc/performance_analysis.py`
- Remote experiment plumbing: `src/cxr_mc/_remote/scripts.py`,
  `src/cxr_mc/_remote/lifecycle.py`, `src/cxr_mc/_remote/cli.py`
- Sweep/profile controls: `src/cxr_mc/sweep.py`, `src/cxr_mc/profiles.py`

## Implementation path

1. Reproduce a complete matched baseline, including a controlled 300 keV
   cohort with the same six-worker allocation used for comparison.
2. Validate activity labels against direct timings. Separate checkpoint I/O,
   CPU transport, GPU spectrum/bremsstrahlung, GPU feed wait, synchronization,
   and OOM retry time.
3. Profile checkpoint write amplification. If confirmed material, implement
   the smallest recovery-safe reduction and measure it independently.
4. Profile the remaining GPU path for kernel fragmentation, synchronization,
   serial host work, and transfer stalls. Change only a measured hotspot.
5. Sweep bounded worker/chunk settings. Record throughput, peak RSS/VRAM,
   utilization distributions, feed-wait, and failures; retain safe defaults or
   expose a validated profile-owned control.
6. Repeat the matched baseline/candidate comparison and document workload,
   commands, environment, repeats, uncertainty, hotspots, and limitations.

## Verification

- Focused CPU tests for timing aggregation, checkpoint cadence/recovery, chunk
  invariance, OOM retry behavior, and runner ordering.
- Likely suites: `tests/test_performance_profile.py`,
  `tests/test_performance_analysis.py`, `tests/test_run.py`,
  `tests/test_montecarlo.py`, plus affected remote CLI/script tests.
- Run scoped lint/type checks before any remote experiment.
- Remote benchmark must complete without OOM and leave resumable checkpoints.
- Candidate must show a repeatable end-to-end improvement on the matched
  workload; report regressions or neutral results rather than generalizing.

## Non-goals

- Maximizing allocation percentages without a throughput benefit.
- Heavy local WSL benchmarks.
- Unbounded GPU process multiplication.
- Physics, seed, grid, or fidelity changes disguised as optimization.
- Scheduler policy changes or progress-dashboard presentation.

## Dispatch

Worker skill: `lead-task`

Required skills: `performance`, `remote-gpu-jobs`, `scientific-library`,
`regression-testing`; add `monte-carlo` when changing stochastic or
RNG-sensitive paths.

Authority: checkpoint commits; no push; no TODO writing; delegation allowed
only for isolated read-only profiling or non-overlapping test investigation.

Stop on incomparable baseline/candidate inputs, unavailable remote allocation,
telemetry schema gaps that prevent attribution, unrelated dirty work, or any
proposal that changes physics outputs to improve speed.
