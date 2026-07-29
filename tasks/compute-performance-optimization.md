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

## Historical experiment artifacts

Artifacts remain unchanged under `scratch-perf/artifacts/compute_test_300keV*`.
They predate the current branch candidate and were copied from completed remote
job directories for offline diagnosis. They are outdated: do not use them as
the current baseline, candidate acceptance evidence, or a source of current
throughput claims. Their only present use is identifying the repeated-OOM
fallback hypothesis that fresh matched runs must test.

Every full session used MoS2, full fidelity, parameter SHA
`bb0c36a6693522f15682f6b0ab6f7ac360b69a451345a18aa55e47bdc2df7986`,
18 uncached cases, one GPU process, and requested/effective six-worker
allocation on the same host. Submission argv is not embedded in the NDJSON;
chunk values and all other comparison fields below come directly from terminal
telemetry.

| Line chunk | Successful runs | Wall seconds | OOM retries/run | Result |
| ---: | ---: | --- | ---: | --- |
| 20,000 | 3/3 | 94.313, 94.324, 94.789 | 0 | Historical zero-OOM cohort |
| 25,000 | 3/3 | 233.566, 233.687, 233.823 | 11 | Stable completion after repeated fallback |
| 22,000 | 1/1 | 234.757 | 11 | Same repeated-fallback regime |
| 30,000 | 0/1 | failed at 11/18 after 100.728 | 1 recorded | Not viable |

Historically, the 20,000 cohort averaged 94.475 seconds (SD 0.272, CV 0.29%) versus
233.692 seconds (SD 0.129, CV 0.055%) at 25,000: 2.474x throughput, or 59.6%
lower wall time. It had zero swap growth, 10.59-12.07 GiB peak VRAM, and
4.65-4.81 GiB peak process-tree RSS. The 25,000 cohort peaked at 13.80 GiB
VRAM. Its 11 caught OOMs per repetition motivate, but do not validate, the
current adaptive candidate.

Checkpointing was only 0.284-0.305 seconds per successful run (0.13-0.30% of
wall). Driver GPU-feed wait was 0.53-1.49%. Neither checkpoint write
amplification nor CPU transport supply is material for this workload.

Nsight attempts `compute_test_300keV-5`, `-6`, and `-7` are unusable:
the reports contain one 9-17 microsecond NVTX range, no CUDA kernel records,
and one `cuModuleGetLoadingMode` call. The two associated telemetry sessions
stop after about 2.5 seconds with zero completed cases. They support no
kernel-fragmentation or synchronization conclusion.

## Provisional local candidate

- `002b8e8` carries a successful line-OOM fallback forward as a non-increasing,
  run-local line-chunk cap. Original cases, order, results, and brem chunks stay
  unchanged.
- `ee26601` restores phase-specific correctness: line OOM halves only line
  chunk and may teach the cap; brem OOM halves only brem chunk; unattributed
  catchable OOM retains legacy dual-halving without teaching. Telemetry exposes
  total plus line/brem/generic retry counts and attempted/effective/learned
  chunks.
- `bfa1381` distinguishes structurally complete sessions from successful
  `done` sessions. Failed terminals remain visible in CSV/timelines but no
  longer contaminate bottleneck classification or comparable-session counts.

Local evidence:

- `tests/test_gpu_oom_retry.py`, `tests/test_chunk_invariance.py`,
  `tests/test_performance_profile.py`: 23 passed before the analysis change.
- `tests/test_montecarlo.py tests/test_run.py`: 131 passed.
- `tests/test_performance_analysis.py`, `tests/test_performance_profile.py`,
  `tests/test_gpu_oom_retry.py`, `tests/test_chunk_invariance.py`: 31 passed.
- Full lint passed. Full typecheck reaches only pre-existing unresolved
  optional imports `imageio.v3` and `kaleido` in
  `src/cxr_mc/plots/render_trajectories.py`.

Remote validation remains required before calling the adaptive candidate an
end-to-end improvement. Remote compute is currently occupied by another
user's long job; no sync, submit, stage, pull, stop, or clear action was taken.
When authorized and capacity is free, run three uncached default-chunk
repetitions from this branch against a fresh matched control, require zero
uncaught OOM/swap growth, verify resumable checkpoints, and compare wall/cost
rate using the fresh control/candidate spread. The historical 0.29% spread is
not an acceptance threshold.

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
