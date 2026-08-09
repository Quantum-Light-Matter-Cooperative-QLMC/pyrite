# Performance-profile analysis playbook

Use this guide after collecting `cxr run PROFILE --remote --perf` logs.
Goal: identify throughput constraint from measured phase and resource behavior,
then test one change at a time. High CPU or GPU utilization is supporting
evidence, not optimization target; compute-weighted throughput is target.

## Outputs

Write analysis artifacts under:

```text
performance-profiles/<profile>/analysis/
├── sessions.csv
├── intervals.csv
├── summary.md
└── timelines/
    └── <job>-<material>-<session>.png
```

`sessions.csv` holds one row per `session_id`. `intervals.csv` holds adjacent
sample pairs after validation and counter differencing. `summary.md` reports
evidence, bottleneck classification, uncertainty, and next controlled
experiment.

Do not modify raw `.ndjson` files.

Runtime records include `backend`, `backend_vendor`, `backend_device`,
requested/resolved resource policy, device budget/reserve, backend-neutral
`allocator_*_mib`, OOM attempts, effective chunks, and any
`backend_fallback_reason`. Legacy `cupy_pool_*_mib` fields remain emitted and
readable for existing `cxr.performance.v1` artifacts. NVIDIA device utilization
comes from `nvidia-smi`; unavailable vendor counters remain null rather than
being mislabeled as NVIDIA data.

## Collect comparable runs

Use one named catalog profile, one slow material, and one faster comparison
material. Run at least three repetitions per configuration. Keep these fixed:

- `parameter_sha256`, fidelity, beam parameters, grids, electron counts, and
  seeds;
- host, SLURM CPU/GPU allocation, worker request, and parallel-material count;
- software commit and environment;
- warm-up treatment and checkpoint state.

The performance-profile name selects an existing catalog profile; it is not an
arbitrary experiment label. Pulled remote logs retain job separation:

```bash
cxr run sub_100keV --remote --perf --detach
cxr remote performance pull sub_100keV
```

Cached cases contain no new compute and must not be compared with uncached
runs. Prefer a separate dataset identity or archive/restore workflow over
deleting checkpoints solely for profiling.

Start with one material process per GPU. Test `--parallel-materials` only after
single-process pipeline behavior is understood.

Generate analysis artifacts after local collection or remote pull:

```bash
cxr profile analyze sub_100keV
```

Use `--performance-dir PATH` for a non-default log root and `--sample-period
SECONDS` when collection did not use the default five-second interval.

### Repeat one uncached remote workload

Run three comparable MoS2 sessions with one-second telemetry and a fixed
six-worker allocation. Submissions go through `cxr run PROFILE -R/--remote`;
`cxr remote run` still resolves but is deprecated and hidden (removal in 0.3.0,
see [`cli-deprecations.md`](cli-deprecations.md)). The supported form has no
repetition flag — submit once per repetition, waiting for each to finish, since
one live job per named profile is allowed. `--perf-reps`, `--chunk-minutes`, and
`--parallel-materials` exist only on the deprecated command:

```bash
cxr run mos2_heavy -m mos2 --remote \
  -p \
  -i 1 \
  --workers 6 \
  --detach

cxr remote performance pull mos2_heavy
cxr profile analyze mos2_heavy --sample-period 1
```

`-p/--perf` runs without shared-cache reads or writes, so every repetition is
uncached and existing production checkpoints stay untouched; profiling
checkpoints are not automatically pulled. `PROFILE` must name a catalog profile
(`cxr profile list`) — it is not a free-form experiment label.

After the baseline completes, test a smaller line-spectrum chunk while keeping
every other option fixed:

```bash
cxr run mos2_heavy -m mos2 --remote \
  -p \
  -i 1 \
  --workers 6 \
  --spec-chunk 20000 \
  --detach
```

Wait for the baseline job to finish before submitting the candidate: one live
job per named profile is allowed. Pull and analyze again; job directories keep
the baseline and candidate sessions separate. Change `--brem-chunk` only in a
later experiment if line-spectrum chunking does not explain the measured gap.

### Capture one Nsight Systems timeline

When one-second telemetry shows a spectrum-dominated run with bursty GPU use,
capture CUDA API calls, kernels, NVTX phases, OS runtime activity, and native
CPU samples for one full uncached session:

```bash
cxr run mos2_heavy -m mos2 --remote \
  -p \
  -i 1 \
  --workers 6 \
  --spec-chunk 20000 \
  --nsys \
  --detach

cxr remote performance pull mos2_heavy
```

Every NVTX range in the tree is GPU-side (`cxr.spectrum_case:*`, `cxr.lines*`,
`cxr.brem`, `cxr.interpolate`); `transport.py` pushes none. A capture therefore
shows a transport-side wait only as an unlabelled gap, and attributing one needs
the `cxr.performance.v1` activity labels (`transport_wait` vs `spectrum`), which
carry per-tick phase identity. Do not expect `cxr.transport.*`.

`--nsys` requires exactly one material, one material process, one performance
repetition, and a monolithic allocation. It uses an isolated job-local
checkpoint so the trace is uncached and never changes or pulls a production
checkpoint. The lab worker must provide `nsys`; no local CUDA toolkit is
required to collect the trace. Nsight sessions launch the synchronized virtual
environment's Python directly and use the `spawn` multiprocessing context;
ordinary scans retain the platform-default context. This avoids Nsight's known
fork-without-exec deadlock mode.

Python-level stack sampling and CUDA backtraces stay opt-in: nsys 2025.6.x
stack-walkers SIGSEGV while unwinding CPython 3.14's frame layout, so the
session omits `--python-sampling`/`--python-backtrace`/`--cudabacktrace` by
default. Set `CXR_MC_NSYS_PYSTACK=1` in the submit environment to add them back
only on a supported Python/nsys pair. Without them the trace still carries the
CUDA kernel timeline, NVTX phases, OS runtime, and process-tree CPU samples --
enough to diagnose bursty-GPU/low-CPU behavior; only Python-frame attribution
is lost.

If an older submitted script remains at 0% with both CPU and GPU idle and its
log ends with `Waiting for termination of re-parented processes`, cancel that
job, sync the updated code, and submit again. It cannot recover useful work.

The existing profile pull fetches these files beside the NDJSON:

```text
<material>.nsys-rep
<material>.sqlite
<material>.nsys-stats.txt
```

The text report summarizes CUDA APIs, kernels, launch-to-execution delay, and
NVTX ranges. Open `.nsys-rep` in an equal-or-newer Nsight Systems GUI for the
timeline. The runner emits one `cxr.spectrum_case:<configuration>` range per
case with nested `cxr.lines`, `cxr.brem`, and `cxr.interpolate` ranges.

Nsight instrumentation adds overhead. Use it to explain gaps, not as a
throughput measurement or a replacement for the three-repetition comparison.

## Validate and normalize

For every input line:

1. Require `schema == "cxr.performance.v1"`.
2. Group by `profile`, job directory, `material`, and `session_id`.
3. Sort by `elapsed_seconds`; verify timestamps and elapsed time are monotonic.
4. Require one `start` record and one terminal `done`, `paused`, or `failed`
   record. Mark incomplete sessions instead of silently dropping them.
5. Confirm one `parameter_sha256` and one execution topology per comparison.
6. Mark intervals longer than twice expected sampling period as sampling gaps.
7. Preserve `null` GPU fields as missing data, never zero.

`start` precedes runtime-plan resolution, so topology and phase fields may
first appear on a later sample.

For adjacent valid samples, compute:

```text
dt = elapsed_seconds[i] - elapsed_seconds[i-1]
case_rate = Δcompleted_new_cases / dt
cost_rate = Δdone_cost / dt
read_rate = Δio_read_bytes / dt
write_rate = Δio_write_bytes / dt
swap_in_rate = Δswap_in_bytes / dt
swap_out_rate = Δswap_out_bytes / dt
```

Treat negative deltas from process-tree CPU or I/O counters as process-turnover
discontinuities. Exclude those intervals from rate calculations; do not clamp
them to zero.

Useful normalized CPU metrics:

```text
host_core_occupancy =
    process_cpu_percent / (100 * cpu_affinity_count)

worker_cpu_efficiency =
    Δ(process_cpu_user_seconds + process_cpu_system_seconds)
    / (dt * effective_workers)
```

`worker_cpu_efficiency` includes driver CPU time, so values slightly above one
are possible. Use it for comparisons, not accounting.

## Session summary metrics

Take rolling totals from last valid terminal sample:

- wall time: `elapsed_seconds`;
- throughput: `done_cost / elapsed_seconds`;
- case throughput: `completed_new_cases / elapsed_seconds`;
- transport time: `transport_seconds_total`;
- spectrum time: `spectrum_seconds_total`;
- driver feed-wait: `driver_wait_seconds_total`;
- checkpoint share: `checkpoint_seconds_total / elapsed_seconds`;
- transport feed-wait fraction: `gpu_feed_wait_fraction` (legacy field name);
- peak worker RSS: maximum `child_process_rss_max_bytes`;
- peak process RSS and system memory percentage;
- peak VRAM and CuPy reserved/peak memory;
- total GPU OOM retries;
- median and 90th-percentile CPU/GPU utilization, clocks, power, and iowait.

For GPU-pipeline sessions:

```text
mean_transport_per_case =
    transport_seconds_total / timed_case_count

mean_spectrum_per_case =
    spectrum_seconds_total / timed_case_count

transport_supply_time =
    mean_transport_per_case / effective_workers
```

Compare `transport_supply_time` with `mean_spectrum_per_case`.
`gpu_feed_wait_fraction` records only driver time blocked on prefetched CPU
transport results. It does not include host launch, synchronization, allocation,
or transfer gaps inside the spectrum phase; use an Nsight trace to separate
those.

Warm-up inflates initial transport waits and CUDA allocation/JIT work. Report
both full-session and steady-state results. Derive steady-state totals by
differencing rolling counters after pipeline fill; use at least
`effective_workers` completed cases as initial warm-up when session length
permits.

## Timeline

Produce one synchronized timeline per session with:

1. `phase`, `active_case`, and `in_flight_case_count`;
2. host and process-tree CPU;
3. GPU compute and memory utilization;
4. GPU SM clock, P-state, power, and temperature;
5. process RSS, max worker RSS, system RAM, swap, VRAM, and CuPy pool;
6. read/write rates and CPU iowait;
7. cumulative completed cost and interval `cost_rate`;
8. cumulative transport, spectrum, feed-wait, and checkpoint time.

Shade intervals where CPU and GPU are both below 20%. Classify each shaded
interval by active phase and resource evidence; do not label it a stall merely
because no case completed during one sample interval.

## Bottleneck decision table

| Evidence | Likely constraint | Next experiment |
|---|---|---|
| `gpu_feed_wait_fraction >= 0.25`, frequent `transport_wait`, falling in-flight count | CPU transport cannot feed GPU | Increase workers within measured RAM headroom |
| Effective workers equal memory cap; max worker RSS far below `worker_memory_budget_mib` | Conservative worker admission | Test lower `CXR_MC_WORKER_MEM_MB` (`CXR_MC_PIPELINE_WORKER_MEM_MB` on the `gpu-pipeline` engine) or explicit `--workers` |
| Low feed-wait, spectrum dominates, GPU busy | GPU spectrum compute | Test spectrum algorithm or safe chunk increase |
| Low feed-wait, spectrum dominates, GPU below 20%, clocks active | Launch/synchronization or host work inside spectrum phase | Sweep `spec_chunk`/`brem_chunk`; profile spectrum internals if unchanged |
| Low GPU utilization plus low clocks, low power, idle P-state during `spectrum` | Power-state, scheduling, or burst sampling | Compare longer cases; inspect GPU clock/throttle policy |
| Checkpoint share material; write rate and iowait align with low utilization | Checkpoint/storage path | Reduce save cost or test faster storage |
| Swap counters rise or available RAM collapses | Memory pressure | Reduce workers/chunks; do not increase concurrency |
| CuPy reserved peak approaches pool limit or OOM retries rise | GPU memory limit | Reduce chunks or concurrency |
| CPU pool phase, low worker CPU efficiency, low iowait | Scheduling, affinity, process priority, or serial Python | Inspect allocation/affinity and worker activity |
| Both devices low during `idle` while work remains | Dispatch/control-flow fault | Reproduce with logs and inspect stop/deadline state |

The 20–25% feed-wait boundary matches runner's existing pipeline decision
rule. Treat every other threshold as an experiment trigger, not proof.

## Controlled experiment order

### 1. Worker admission

Hold chunks and workload fixed. Compare current auto worker count against
adjacent safe counts. Before relaxing memory policy, require:

- sampled worker RSS comfortably below configured per-worker budget;
- no swap activity;
- adequate system available memory across repetitions.

Win condition: higher `cost_rate`, lower feed-wait, stable memory, and no
regression in spectrum time.

### 2. Spectrum and bremsstrahlung chunks

Hold workers fixed. Test smaller, current, and larger chunks within VRAM
headroom. Record effective `spec_chunk` and `brem_chunk` from each active case.

Win condition: lower spectrum time and higher end-to-end throughput without
OOM retries or unsafe CuPy/VRAM growth.

### 3. CuPy pool-release cadence

Only test after chunk behavior is understood. Compare pool-release cadence
while watching CuPy reserved peak and spectrum time.

Win condition: less allocation/synchronization time with bounded reserved
memory.

### 4. Material concurrency

Only if one scan cannot keep GPU busy after worker/chunk tuning. Compare one
and two material processes, accounting for duplicated CUDA contexts, CuPy
pools, and transport workers.

Win condition: higher aggregate `done_cost / elapsed_seconds`, not merely
higher GPU utilization.

## Report template

```markdown
# Performance analysis: <profile>, <commit>, <date>

## Workload
- Materials:
- Parameter SHA:
- Host/GPU:
- Allocation:
- Repetitions:
- Warm-up rule:

## Baseline
| Metric | Median | Spread |
|---|---:|---:|
| Wall time | | |
| Cost rate | | |
| GPU feed-wait fraction | | |
| Mean transport/case | | |
| Mean spectrum/case | | |
| Checkpoint share | | |
| Peak worker RSS | | |
| Peak VRAM/CuPy pool | | |

## Evidence
- Observed:
- Ruled out:
- Remaining uncertainty:

## Classification
- Primary bottleneck:
- Secondary bottleneck:
- Confidence:

## Next controlled experiment
- One changed variable:
- Expected signature:
- Safety limit:
- Acceptance criterion:
```

Optimization recommendation requires repeatable throughput improvement on
identical inputs. Report run-to-run spread and retain raw logs with analysis
artifacts.
