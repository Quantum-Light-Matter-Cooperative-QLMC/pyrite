# Analyze performance profiles

Use performance profiles to diagnose slow runs and compare changes to PyRITE's
execution settings. Optimize **uncached, compute-weighted throughput**
(`done_cost / elapsed_seconds`), not CPU or GPU utilization alone. Keep the
scientific workload fixed when comparing execution configurations.

For an editable introduction to history counts, convergence and runtime, use
[Sampling, convergence and performance](../notebooks/designing/07-sampling-convergence-performance.md).
This recipe diagnoses an existing run using its execution telemetry.

## Collect and analyze a run

`PROFILE` must be an existing catalog profile (`pyrite profile list`), not an
arbitrary label. For example, to profile one material on a remote worker:

```bash
pyrite-dev perf mos2_heavy -m mos2 --remote --workers 6 -i 1 --detach
# After the job finishes:
pyrite remote performance pull mos2_heavy
pyrite-dev performance analyze mos2_heavy --sample-period 1
```

`pyrite-dev perf` runs without shared-cache reads or writes, so production
checkpoints are unaffected. Remote profiling supports one live job per named
profile; finish the previous job before submitting another. Repeat each
configuration at least three times. Use `--performance-dir PATH` with the
analysis command if the logs are outside the default performance directory.

The analyzer writes to `pyrite-output/performance/<profile>/analysis/`:

| File | Purpose |
| --- | --- |
| `sessions.csv` | One row per session: throughput, wall time, timings, and resource peaks. |
| `intervals.csv` | Rates calculated from adjacent valid telemetry samples. |
| `summary.md` | Session comparison and interpretation template. |
| `timelines/*.png` | Time-aligned phase, utilization, memory, and progress plots. |

Keep the source `.ndjson` logs unchanged. A `paused` or `failed` session is not a
completed benchmark; inspect its status before comparing it with successful
runs. Missing GPU counters are **unknown**, not zero.

## Make comparisons reproducible

Use the same profile, material, beam, grids, electron counts, seeds, software
commit, host/GPU allocation, and warm-up treatment. Check that
`parameter_sha256` matches. Record the actual worker and chunk settings; a
resource policy may admit fewer workers or smaller chunks than requested.
Change **one execution setting at a time**, then compare median and spread
across repetitions.

Distinguish complete-session rates from steady-state rates: transport startup,
JIT compilation, memory allocation, and pipeline fill can dominate short
runs. For GPU pipelines, allow at least one batch of completed work per
worker before evaluating steady state when the run is long enough.

The principal measurements are:

| Measurement | Interpretation |
| --- | --- |
| `done_cost / elapsed_seconds` | Primary compute-weighted throughput; higher is better for comparable workloads. |
| `completed_new_cases / elapsed_seconds` | Case throughput; meaningful only if case costs are comparable. |
| `transport_seconds_total`, `spectrum_seconds_total` | Which calculation phase dominates. |
| `driver_wait_seconds_total`, `gpu_feed_wait_fraction` | Time the spectrum driver waits for prefetched transport work. |
| `checkpoint_seconds_total` | Time spent writing results. |
| Worker RSS, available RAM, swap | Whether more CPU workers are safe. |
| VRAM, allocator pool peaks, `gpu_oom_retry_count` | GPU memory headroom and retry pressure. |

For pipeline runs, an additional check is
`(transport_seconds_total / timed_case_count) / effective_workers` versus
`spectrum_seconds_total / timed_case_count`: if transport supply is slower,
the GPU can run out of work. The feed-wait counter does **not** capture host
launch, transfer, or synchronization delays *inside* spectrum processing.

## Diagnose the bottleneck

| Observed evidence | Likely limitation | Controlled test |
| --- | --- | --- |
| `gpu_feed_wait_fraction` near or above 0.25, repeated `transport_wait` | CPU transport cannot feed the GPU | Increase workers if RAM and swap permit. |
| Worker cap reached, measured RSS well below the admission budget | Conservative worker admission | Test adjacent worker counts or a measured per-worker memory estimate. |
| Spectrum dominates; GPU stays busy | GPU spectrum compute | Tune line/bremsstrahlung chunks or inspect kernels. |
| Spectrum dominates; GPU appears idle but feed-wait is low | Launch, synchronization, transfer, or short GPU bursts | Capture an Nsight timeline before changing algorithms. |
| Checkpoint time, disk writes, and iowait rise together | Storage | Compare storage or output settings. |
| RAM runs low, swap grows, or GPU OOM retries appear | Memory pressure | Reduce workers/chunks or concurrency. |
| Both CPU and GPU appear idle while unfinished work remains | Scheduling or control flow | Inspect phases, in-flight cases, and job logs. |

These are hypotheses, not diagnoses from utilization alone. Sampled GPU
utilization can miss brief kernels, and `gpu_feed_wait_fraction` describes
only transport-to-spectrum waiting. Inspect a timeline before interpreting a
low-utilization interval as a stall.

A useful tuning order is **workers → spectrum chunks → material
concurrency**. Start with one material process per GPU. For each candidate,
require repeatable throughput improvement without swap, OOM retries, or
unbounded memory growth. More concurrent processes also duplicate GPU
contexts and allocator pools; they are not automatically faster.

## Profile a suspicious phase

Use Nsight Systems when telemetry suggests launch gaps or GPU-side
bottlenecks. Collect one uncached, single-material, single-process remote run
with a monolithic allocation:

```bash
pyrite-dev perf mos2_heavy -m mos2 --remote --nsys --detach
# After completion:
pyrite remote performance pull mos2_heavy
```

Pulled artifacts include `<material>.nsys-rep`, `<material>.sqlite`, and
`<material>.nsys-stats.txt`. Inspect CUDA kernels and the `pyrite.transport.*`
and `pyrite.spectrum_case:*` NVTX ranges. A host-side launch range is not the
same as GPU kernel execution time. Nsight adds overhead, so do **not** use its
wall time for throughput comparisons.

For Python-level CPU attribution, run a separate serial pass:

```bash
pyrite-dev perf mos2_heavy -m mos2 --remote --cpu-only --detach
pyrite remote performance pull mos2_heavy
```

This produces `<material>.cpu.prof` and `<material>.cpu.txt`. The `--cpu`
option also exists to run the primary profiling pass followed by a CPU pass.
Both options require remote execution; `--cpu-only` cannot be combined with
`--nsys` or GPU chunk pins.

## Understand memory recovery

When a profile slows down after an OOM or changes its effective chunks,
inspect `gpu_oom_retry_count`, `recovery_events`, `recovery_path`,
`backend_fallback_reason` and the resolved runtime plan. A resource
admission rejection is not necessarily an allocator OOM. The
[memory and scheduling contract](../computation/memory-and-scheduling.md#interpreting-recovery-in-performance-logs)
explains when retries, host-resident replay or CPU scoring may occur and how
these routes affect reproducibility.

## Record the conclusion

In `summary.md`, record the profile and commit, fixed scientific inputs,
resource allocation, repetitions, median/spread of throughput, phase timing,
peak memory, suspected bottleneck, and the **single next experiment**. Keep
the raw logs so a claimed improvement can be reproduced. Do not infer a
cross-engine speed ranking without separately validating numerical agreement.
