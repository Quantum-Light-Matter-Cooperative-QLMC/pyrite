# Memory, chunking, and scheduling

PyRITE limits memory use by splitting spectrum reductions into chunks, transport into electron batches, and sweeps into independently scheduled cases. Backend and transport-core selection are covered in [execution and acceleration](execution-and-acceleration.md).

These partitions preserve the physical calculation. Floating-point sums can change when terms are grouped differently, especially for coherent fields; see [precision and tolerances](precision-and-tolerances.md).

## Resource policies

A named resource policy sets device budgets, CPU-fallback limits, and retry rules. It resolves from `PYRITE_MC_RESOURCE_POLICY`, and `auto` — the default — picks by device size: `conservative` below 8 GiB of device memory, `balanced` at or above it, and `balanced` when there is no device at all.

```{list-table} Resolved limits per resource policy.
:name: tbl-memory-policies
:header-rows: 1

* - Policy
  - Device fraction
  - Device reserve
  - Host fraction (fallback)
  - Host reserve (fallback)
  - OOM retries
  - Pool release cadence
* - `conservative`
  - 0.50
  - 2 GiB
  - 0.60
  - 4 GiB
  - 3
  - every 4 cases
* - `balanced`
  - 0.70
  - 1 GiB
  - 0.75
  - 2 GiB
  - 3
  - every 8 cases
* - `throughput`
  - 0.85
  - 0
  - 0.85
  - 1 GiB
  - 3
  - every 16 cases
```

The device budget is $\min(\text{fraction} \times \text{total},\ \text{total} - \text{reserve})$, floored at zero. The fraction limits the run's allocation; the reserve leaves memory for the CUDA context and other processes. On small devices, the reserve can be the tighter constraint.

The host columns govern CPU-fallback admission: whether the host has enough memory for a full-case worker when an accelerator path falls back to the CPU. Insufficient memory raises a resource error naming the shortfall. Ordinary worker-pool sizing uses the separate cap in {eq}`eq-memory-workers`.

A policy is distinct from a simulation sweep profile. It governs execution resources only and never appears in a case's physical parameters.

## Host transport step capacity

CPU transport preallocates up to `n_electrons * max_steps` segment rows,
even when actual trajectories are much shorter. The initial allocation
uses a 20,000-step budget by default. Set
`PYRITE_MC_TRANSPORT_INITIAL_MAX_STEPS` to a smaller positive integer
(no greater than 160,000) to start with fewer rows. If a trajectory
exceeds the allocated capacity, PyRITE deterministically replays the
transport with a doubled budget, capped at the original ceiling. This
is an allocation limit, **not** a physical stopping criterion or a
truncation of trajectories; it may trade memory savings for more work.

The disposable Sphinx/Marimo tutorial exports start at 128 steps to
reduce peak allocations on Read the Docs without changing the history counts.
Ordinary PyRITE simulations retain their existing default.

## Chunking the spectrum reductions

The line and bremsstrahlung spectrum kernels reduce over segments into energy bins. The reduction is chunked over segments, because the peak transient is a dense $(\text{chunk}, n_{\text{bins}})$ intermediate — and in fact about three of them concurrently, for the argument, its `sinc`, and that expression's internal temporary.

Chunk size is therefore chosen to hold a **byte product** constant rather than to hold the chunk constant:

```{math}
:label: eq-memory-chunk

\mathrm{chunk} \;\approx\; \frac{B}{3\, n_{\text{bins}}\, w},
```

where $B$ is the transient budget in bytes and $w$ is the item size of the working real type. The result is clamped into $[1000, 100\,000]$ and then capped again against the device budget.

As the number of bins grows, the chunk shrinks to stay within the same byte budget. A `float32` working type permits roughly twice as many segments per chunk as `float64` for the same budget.

Precedence runs: an explicit per-case `spec_chunk` / `brem_chunk` wins, then a pinned `PYRITE_MC_CXR_CHUNK` / `PYRITE_MC_BREM_CHUNK`, then the adaptive size. Whichever wins is capped by device admission.

### Blocking the bremsstrahlung photon grid

Segment chunking leaves one cost proportional to the full grid: each pass stages state that does not depend on the segments. For EEDL that is one unit-base panel row per photon node, about 327 binary64 panels per element, plus a REAL device copy on an accelerator. The BremsLib route stages the same rows as its fallback. A 1 keV–3 MeV grid at 10 eV spacing has 299,901 nodes, which is 0.8 GB per element.

The runner therefore scores a broad continuum grid in contiguous photon blocks. Each block's staging, together with one `mu(E)` per absorbing layer and the grid/output vectors, must fit `PYRITE_MC_CXR_BUDGET_MB`. This budget is separate from the chunk transients above, which are already sized for the whole grid and only shrink inside a block. Bethe–Heitler stages no panels. A grid that fits is scored in one block, so its result is unchanged.

Measured host costs (CPU fp64, #819) agree with that split:

| Cost | Measured | Assumed by |
|---|---|---|
| EEDL staging (C, 327 panels) | 2640 B/node | block width: 8 B/panel + 40 B/node |
| Bethe–Heitler staging | 16 B/node | block width: 40 B/node |
| Dense transient, EEDL and Bethe–Heitler | 81 B/(segment·node) | chunk admission: 8 × 8 B = 64 B |
| Dense transient, BremsLib and coupled | 88 B/(segment·node) | chunk admission: 64 B, plus the BremsLib $2^{21}$-cell cap |

The block estimate holds. The chunk admission's eight dense intermediates undercount the host transient by about a third. Correcting that changes default chunk sizes, and with them the summation order of default outputs, so it is left to a separate change.

Every photon node is scored independently: attenuation, cross sections and dielectric suppression are pointwise in photon energy, and the EEDL cutoff normalization and escape paths depend on segments only. Blocking is therefore exact in the same sense as segment chunking: a per-node segment sum is a BLAS matrix-vector product whose summation order depends on the block width, so blocked and one-pass spectra agree to rounding (#819). The block width is not a user knob. It is reported as the run diagnostic `brem_photon_block` beside `effective_brem_chunk` and does not enter identity. Direct `mc_brem_spectrum` calls score in one pass unless they pass `photon_block`.

### Admission

Before allocation, admission computes how many segments fit from the bin count, item size, and device budget. It returns the smaller of that count and the requested chunk.

If the minimum chunk of 1000 does not fit, admission raises a resource error naming the budget and bin count.

## Out-of-memory handling

Device out-of-memory errors are handled at three levels.

* The CuPy pool is capped at the configured fraction of device memory so allocations beyond that limit raise a catchable error.
* The coherent-line phase retries with progressively halved chunks, up to the policy's retry count. Later cases in the same run reuse a successful smaller line chunk. The original case and its bremsstrahlung chunk remain unchanged for checkpoint and recompute compatibility. Errors in other phases propagate without changing the line chunk.
* If resident transport segments exceed device memory, transport is replayed with the same seed and the segments downloaded to the host. Counter-addressed streams make the replay exact.

The pool is released every 4, 8, or 16 cases, depending on policy, or when a reserved-pool watermark is crossed. The pool cap and retries handle allocation growth between releases.

### Interpreting recovery in performance logs

Retries reduce eligible chunks without resampling physical histories, as
explained above. Under automatic **engine and backend** selection, exhausted
spectrum retries or device admission rejection can fall back to CPU scoring
when host memory admission succeeds. Explicit accelerator selection cannot
silently switch to CPU. Resident CUDA transport failure has its own same-core
replay using host-resident segments.

For profiling, examine `gpu_oom_retry_count`, `recovery_events`, `recovery_path`,
`backend_fallback_reason` and **effective** rather than requested chunks to
interpret performance logs. A resource admission rejection and an allocator
OOM are different failures. Recovery preserves seeds and scientific settings,
although different reduction grouping and device arithmetic can change
floating-point roundoff. For the measurement workflow see
[analyze performance profiles](../guides/performance-profile-analysis.md#understand-memory-recovery).

## Execution engines

A sweep resolves to one of three topologies.

`serial` : One process, one case at a time. Chosen when there is a single case, when workers are explicitly disabled, when only one worker fits, or whenever the run transports on the device.

`cpu-pool` : A process pool of full-case workers, each running transport and spectrum for its own cases. The CPU topology.

`gpu-pipeline` : A pool of **transport-only** workers feeding a driver process that owns all device state and runs every spectrum kernel. Its premise is that CPU transport can be hidden behind GPU work, so it needs at least two workers to be worth choosing over `serial`.

### Why device transport forces serial

When transport runs on the device, it shares that device with the spectrum kernels. A CPU transport pool cannot overlap those stages. Keeping execution in the driver also avoids additional CUDA contexts and lets spectrum kernels read resident segments without transferring them between processes.

Device-transport routing applies across the run's cases. Sweeps hold electron count fixed across the grid, so mixed CPU/CUDA transport runs are uncommon.

## Host-memory admission

Both worker pools are bounded by host RAM, not just by core count. The cap is

```{math}
:label: eq-memory-workers

n_{\max} = \frac{\min\bigl(\text{MemAvailable},\ 0.9 \times \text{MemTotal}\bigr)}
                {\text{per-worker budget}} .
```

The pools use different per-worker budgets:

* Full-case CPU workers: 6144 MB for transport and spectrum calculations.
* GPU-pipeline transport workers: 1536 MB; the driver owns spectrum and device state.

The pipeline also **prefetches** two cases beyond its worker count, so a worker always has the next case queued. Each in-flight case is a host-resident segment payload the driver holds, so prefetch depth is charged against host RAM exactly like a worker rather than being treated as free.

## Reading the runtime plan

Before a sweep starts, the resolved topology and budgets are available as a runtime plan — the engine, the effective worker count, the per-worker memory budget, the prefetch depth, the representative chunk sizes, the backend and its device, any fallback reason, and both the requested and resolved resource policy.

This is the artifact to inspect first when a run is slower or larger than expected, because it distinguishes the three failure modes that look alike from the outside:

* the **backend** fell back to NumPy — check `backend_fallback_reason`;
* the **engine** resolved to `serial` when a pool was expected — usually host RAM admission or an all-CUDA transport run;
* the **chunk** collapsed to its floor — a grid wide enough that the device budget barely admits the minimum.

## Validation

Tests cover chunk invariance of spectrum reductions, capacity-replay invariance of transport, memory admission, and pool-release cadence. Admission and cadence checks have no device dependency and run without a GPU.

The measurement records behind the defaults are in [compute-performance optimization](../repo-design/compute/compute-performance-optimization.md).
