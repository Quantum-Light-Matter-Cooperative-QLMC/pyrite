# Memory, chunking, and scheduling

Where a run's arithmetic happens is [the previous page](execution-and-acceleration.md).
This page is about how much of it happens at once: how a sweep's cases are
distributed across processes and devices, and how every large allocation is
sized so that a bigger grid or a bigger material does not become an
out-of-memory kill.

The governing principle is that **none of it changes results**. Chunking
partitions a sum over segments, batching partitions a loop over electrons, and
worker assignment partitions a list of independent cases. All three are exact
decompositions; they trade memory against speed and nothing else. The one place
that needs care is the coherent field sum, where partitioning a *complex*
accumulation is exact in real arithmetic but not in floating point — that is
treated in [precision and tolerances](precision-and-tolerances.md).

## Resource policies

A single named policy sets every budget in a run. It resolves from
`PYRITE_MC_RESOURCE_POLICY`, and `auto` — the default — picks by device size:
`conservative` below 8 GiB of device memory, `balanced` at or above it, and
`balanced` when there is no device at all.

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

The device budget is
$\min(\text{fraction} \times \text{total},\ \text{total} - \text{reserve})$,
floored at zero. The fraction bounds the run's own appetite; the reserve
protects whatever else is on the card — a display server, another process, the
CUDA context itself. On a small card the reserve is the binding constraint,
which is why `conservative` pairs a low fraction with a large reserve rather
than relying on either alone.

The two host columns govern a narrower question than their names suggest:
they budget **CPU-fallback admission** — whether the host can seat one
full-case worker when an accelerator path steps aside — and raise a resource
error naming the shortfall when it cannot. Ordinary worker-pool sizing uses
the separate, policy-independent cap in {eq}`eq-memory-workers`.

A policy is distinct from a simulation sweep profile. It governs execution
resources only and never appears in a case's physical parameters.

## Chunking the spectrum reductions

The line and bremsstrahlung spectrum kernels reduce over segments into energy
bins. The reduction is chunked over segments, because the peak transient is a
dense $(\text{chunk}, n_{\text{bins}})$ intermediate — and in fact about three
of them concurrently, for the argument, its `sinc`, and that expression's
internal temporary.

Chunk size is therefore chosen to hold a **byte product** constant rather than
to hold the chunk constant:

```{math}
:label: eq-memory-chunk

\mathrm{chunk} \;\approx\; \frac{B}{3\, n_{\text{bins}}\, w},
```

where $B$ is the transient budget in bytes and $w$ is the item size of the
working real type. The result is clamped into $[1000, 100\,000]$ and then capped
again against the device budget.

The adaptive rule replaced a fixed default for a concrete reason. Widening the
line grid to 30 000 eV grew the bin count roughly threefold and silently tripled
the per-matmul transient, because the fixed chunk had been tuned on the older,
narrower grid. Holding the byte product constant instead means the chunk shrinks
as the grid widens and grows as it narrows, automatically.

Two consequences of {eq}`eq-memory-chunk` are worth noting:

* $w$ is the working real type's size, so a GPU run at `float32` admits roughly
  twice the chunk a CPU run at `float64` does, from the same budget.
* The default budget reproduces the historical fixed chunk of 40 000 exactly on
  the pre-2026-07 line grid, so the change was a generalization rather than a
  retuning.

Precedence runs: an explicit per-case `spec_chunk` / `brem_chunk` wins, then a
pinned `PYRITE_MC_SPEC_CHUNK` / `PYRITE_MC_BREM_CHUNK`, then the adaptive size.
Whichever wins is capped by device admission.

### Admission

Admission is the check that runs *before* allocation rather than after the
failure. Given the requested chunk, the bin count, the item size, and the device
budget, it computes how many segments actually fit and returns the smaller of
that and the request.

If even the minimum chunk of 1000 does not fit, it raises a resource error
naming the budget and the bin count. That is the correct behavior: a grid so
wide that 1000 segments will not fit in the device budget is a configuration
problem, and reporting it before allocation gives a diagnosable message instead
of an allocator failure deep inside a kernel.

## Out-of-memory handling

Device OOM is treated as a recoverable, *tagged* condition rather than a crash,
in three layers.

**Cap the pool so the error is catchable.** The CuPy default pool is capped at a
fraction of device memory, so an over-budget allocation raises a catchable
out-of-memory error before the process hard-OOMs. Without the cap the allocator
would keep growing until the driver killed the process, which is not something a
retry can help with.

**Retry the line phase with halved chunks.** The GPU spectrum phase retries an
OOM with progressively halved chunk sizes, up to the policy's retry count. A
successful fallback is carried forward within one run so later cases start from
the size that worked, but it only ever *lowers* the resolved line chunk, and the
original case and its bremsstrahlung chunk stay pristine for checkpoint and
recompute compatibility.

OOMs outside the coherent-line phase propagate rather than teaching a false cap
— an OOM in a different phase says nothing about the line chunk.

**Replay transport on the host.** If the device cannot hold a resident segment
payload, the OOM is caught and the same seed replayed with segments downloaded.
Counter-addressed random streams make that replay exact, so the fallback costs
bus time and not the result. This is the one silent fallback in the system, and
it is safe precisely because of that exactness.

The pool is also released on a cadence — every 4, 8, or 16 cases by policy, or
when a reserved-pool watermark is crossed. The cadence is no longer
load-bearing for correctness: the pool cap and the retry catch growth between
frees. Freeing every case measured about 4 % of GPU-phase wall time on an 8 GiB
card, which is what the cadence buys back.

## Execution engines

A sweep resolves to one of three topologies.

`serial`
: One process, one case at a time. Chosen when there is a single case, when
  workers are explicitly disabled, when only one worker fits, or — importantly —
  whenever the run transports on the device.

`cpu-pool`
: A process pool of full-case workers, each running transport and spectrum for
  its own cases. The CPU topology.

`gpu-pipeline`
: A pool of **transport-only** workers feeding a driver process that owns all
  device state and runs every spectrum kernel. Its premise is that CPU transport
  can be hidden behind GPU work, so it needs at least two workers to be worth
  choosing over `serial`.

### Why device transport forces serial

A device-transported run gives up the `gpu-pipeline` engine, for two independent
reasons. The pipeline's premise no longer holds — there is nothing left to hide,
since transport is on the same device as the spectrum work — and there is no
room for a second CUDA context on a card this process is already driving.

Such a run is kept in the driver process, serially, and every worker process is
pinned off the device. Serial is also simply faster once transport is on the
device: the case pays one kernel launch instead of segment-proportional pipe
traffic, and the segments then stay resident for the spectrum kernels.

The switch is **all-or-nothing** across a run's cases. A mixed run would strand
its CPU-core cases in the driver with nothing overlapping them. Sweeps hold the
electron count fixed across the grid, so mixed runs are the exception rather
than the rule.

## Host-memory admission

Both worker pools are bounded by host RAM, not just by core count. The cap is

```{math}
:label: eq-memory-workers

n_{\max} = \frac{\min\bigl(\text{MemAvailable},\ 0.9 \times \text{MemTotal}\bigr)}
                {\text{per-worker budget}} .
```

The two pools pass different budgets, because their workers are not comparable:

* **Full-case CPU pool: 6144 MB per worker.** Sized from a measured worker
  footprint of roughly 5.5 GB of anonymous RSS at 200 keV on a wide grid,
  rounded up.
* **GPU-pipeline transport pool: 1536 MB per worker.** These workers run
  transport *only* — the driver owns all spectrum and device state — so their
  measured peak child RSS was 552–1033 MB. The budget leaves about 50 %
  headroom.

Giving both pools the same budget was a real bug, not a theoretical one: it
capped the pipeline at two workers on a 23 GB box and silently clamped an
explicit worker count along with it.

The pipeline also **prefetches** two cases beyond its worker count, so a worker
always has the next case queued. Each in-flight case is a host-resident segment
payload the driver holds, so prefetch depth is charged against host RAM exactly
like a worker rather than being treated as free.

## Reading the runtime plan

Before a sweep starts, the resolved topology and budgets are available as a
runtime plan — the engine, the effective worker count, the per-worker memory
budget, the prefetch depth, the representative chunk sizes, the backend and its
device, any fallback reason, and both the requested and resolved resource
policy.

This is the artifact to inspect first when a run is slower or larger than
expected, because it distinguishes the three failure modes that look alike from
the outside:

* the **backend** fell back to NumPy — check `backend_fallback_reason`;
* the **engine** resolved to `serial` when a pool was expected — usually host
  RAM admission or an all-CUDA transport run;
* the **chunk** collapsed to its floor — a grid wide enough that the device
  budget barely admits the minimum.

## Validation

Nothing on this page is a physics claim, and none of it carries a ledger row.
The properties that matter are invariance properties, and they are pinned as
ordinary tests: chunk-invariance of the spectrum reductions, capacity-replay
invariance of transport, and the purely arithmetic admission and free-cadence
predicates, which are written without any device dependency so they unit-test on
a machine with no GPU.

The measured basis for the defaults — the profiling rounds, the OOM incidents
that motivated adaptive chunking and split worker budgets, and the throughput
tables — is recorded in
[compute-performance optimization](../repo-design/compute/compute-performance-optimization.md).
