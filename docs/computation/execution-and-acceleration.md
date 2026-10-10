# Execution and acceleration

PyRITE selects an array backend for spectrum calculations and a transport core for electron trajectories. This page explains those selections, the kernels they use, and their effect on reproducibility. See [memory, chunking, and scheduling](memory-and-scheduling.md) for memory budgets and worker pools, and [precision and tolerances](precision-and-tolerances.md) for floating-point policy.

The cores implement the same physical models, but changing cores can change the sampled trajectories. See [what acceleration does not change](#what-acceleration-does-not-change) before comparing results across cores.

## Two independent selections

A run makes two separate choices:

```{list-table} The two runtime selections and what each governs.
:name: tbl-execution-selections
:header-rows: 1

* - Selection
  - Environment variable
  - Governs
* - **Array backend**
  - `PYRITE_MC_BACKEND`
  - Which array module the *spectrum* kernels run on: `auto`, `cpu`, `cuda`, `rocm`, `sycl`.
* - **Transport core**
  - `PYRITE_MC_TRANSPORT_CORE`
  - Which electron-transport implementation runs: `auto`, `lockstep`, `per-electron`, `cuda`.
```

The CUDA transport core requires a CUDA array backend for its `cupyx.jit` kernel. A ROCm or SYCL backend runs its spectrum kernels on the device and its transport on the CPU.

### Backend resolution

`auto` probes in a fixed order — CuPy for CUDA or ROCm, then Intel SYCL through `dpnp`, then NumPy — and takes the first that yields a usable device. The probe is recorded: a backend that fell back to NumPy carries a `fallback_reason` string naming why, which appears in the runtime plan.

An explicit accelerator request raises if it cannot be satisfied. Asking for `PYRITE_MC_BACKEND=cuda` on a machine with a ROCm CuPy build raises rather than running somewhere else, and naming a backend whose device lacks native `float64` support raises for the same reason, whatever `PYRITE_FP64` says. Only `auto` is permitted to substitute, and only with a logged warning; such a device falls back to NumPy with `fallback_reason` `unsupported_fp64: <device>`.

A device without native `float64` is refused even though `REAL` would be `float32`: some spectrum reductions still accumulate in `float64` on the device, and such a queue fails mid-run instead of at selection. Intel Arc (DG2) and integrated Intel GPUs fall in this class, so on them the SYCL backend is currently unavailable; on-device support is tracked in issue #224.

Selection happens once at import and applies to the whole process: the module-level `xp`, `REAL`, and `_GPU` bindings are read by every kernel, so a mid-run switch would leave arrays from two backends in the same reduction. A different backend requires a separate process. Worker pools select their backend this way (see [memory, chunking, and scheduling](memory-and-scheduling.md)).

### Transport-core resolution

`auto` is the default. It selects the CUDA core when all three conditions hold:

1. the process has a CUDA device and the kernel module imports;
2. the run is ungrooved, since the CUDA core does not support periodic facet exit and re-entry;
3. the electron count exceeds `CUDA_TRANSPORT_MIN_ELECTRONS`, currently 1000.

Otherwise it resolves to the lockstep CPU core. The count tested is transport's own, $\max(N_e, N_{e,\mathrm{brem}})$, since one transport serves both populations.

An explicit `transport_core=` selects that core or raises an error. Use it to pin the core for reproducibility or investigate a host/device difference.

The threshold avoids CUDA launch and staging overhead for small electron populations.

## The three transport cores

All three implement the same physics. They differ in how they traverse the electron ensemble, and therefore in what order they consume randomness.

`lockstep` : An outer step loop over an inner electron loop, with a single NumPy `Generator` serving every electron. Draws are consumed in a fixed interleaved order. This is the default whenever `auto` does not select CUDA.

`per-electron` : One electron transported to completion before the next. This is the CPU reference for the CUDA kernel: both use the same algorithm and address random draws identically, allowing comparisons of host/device arithmetic. It is intended for validation, not CPU acceleration.

`cuda` : The `cupyx.jit` kernel. One thread per electron, run to completion.

### Why run-to-completion

Each CUDA thread transports one electron independently. It computes random draws from the electron index and draw address, without a shared generator. This keeps the random streams independent of thread scheduling and launch geometry. The construction and bit-for-bit host/device agreement of the draws are documented in [random number streams](random-streams.md).

### Output addressing and capacity replay

Segment $s$ of local electron $i$ is written to slot $i \times \mathrm{cap} + s$ — a pure function of the electron index, with no atomics. After the kernel, a boolean mask built from the per-electron segment counts compacts the batch into electron-major, step-minor order.

The counter records the **true** count even when it exceeds the capacity: the thread keeps transporting but stops writing. The driver then grows the capacity and replays the batch. Replay is exact because the streams are counter-addressed, so an overflow costs time and nothing else.

Replay permits a default capacity of 512 slots instead of reserving the `max_steps` worst case of 20 000 for every electron on the device.

Because the true count is always reported, every batch — including one that overflowed — states exactly what the material needs. Two policies use that:

* **Replay at the measured capacity.** One replay at the reported requirement always suffices, so no growth factor is applied. Capacity is reset after every batch to the running maximum times a headroom factor of 1.25, in both directions. Capacity trades against batch size out of a fixed byte budget, so an over-provisioned capacity costs launches exactly as an under-provisioned one costs replays.
* **A short first batch.** A probe of 1024 electrons caps the width of the first batch — the one batch still sized by the initial guess. Its segments are kept, since output slots are addressed by electron index, so a short first batch is simply a short batch. This bounds what a wrong initial capacity can cost.

None of `seg_capacity`, `scratch_budget_bytes`, `capacity_headroom`, `max_batch`, or `probe_electrons` changes results. They set memory and launch behavior only.

## The transport lookup table

The ungrooved transport cores read a per-run table of every energy-dependent scalar in the transport hot loop, rather than recomputing transcendentals per event.

The table is built once per run from the layer compositions and covers, per layer and energy:

* the total elastic rate,
* the stopping power $dE/ds$,
* $1/\beta$,
* the collision-element cumulative distribution,
* the elastic screening parameter $\alpha$.

The grid is uniform in kinetic energy. A uniform grid turns table lookup into one multiply, one integer conversion, and a linear interpolation:

```{math}
:label: eq-execution-lut-index

x = (E - E_{\min})\,\Delta E^{-1}, \qquad
i = \lfloor x \rfloor, \qquad
f = x - i,
```

with the interpolated value $T_i + f\,(T_{i+1} - T_i)$. A logarithmic grid would track the physics more naturally — rates vary as powers of $E$ — but would cost a logarithm per lookup in the innermost loop, which is the operation the table exists to remove. Indices are clamped at both ends, so an energy outside the grid extrapolates flat rather than reading out of bounds.

Grid spacing targets 0.025 keV, bounded below by 256 points and above by 16 384 to limit memory use for wide energy ranges. The alpha column is the one entry still built by interpolation rather than closed form: Mott screening parameters come from tabulated $(\log E, \log \alpha)$ pairs, interpolated in log space and exponentiated onto the linear grid.

The builder fails closed on a non-positive or non-finite total elastic rate rather than emitting a table that would produce an infinite mean free path.

The table's own interpolation error is a systematic, not a statistical, effect, and it sits well below the step-control effects quantified under `energy-step-convergence`; see [statistical methods](statistical-methods.md#null-results).

## The coherent streaming kernel

For reflection/orientation row $g$, polarization $p$, and photon energy $E$, the kernel accumulates a complex field

```{math}
:label: eq-execution-coherent-field

F_{gpE} = \sum_j c_{gpj}\,
  \operatorname{sinc}\!\bigl(a_{gj}(E - E_{r,gj})\bigr)\,
  \exp\!\bigl[i\,(s_j E - \varphi_{gj})\bigr],
```

and the spectrum contribution is $\sum_g w_g\,(|F_{gsE}|^2 + |F_{gpE}|^2)$.

Streaming rests on one identity: the field is linear in the segment index, so

```{math}
:label: eq-execution-streaming-identity

F_{gpE} = \sum_{\text{blocks}} F^{\text{block}}_{gpE}.
```

**Intensity is never reduced per block.** Four persistent field planes — sigma and pi, real and imaginary — are updated across segment blocks and squared only after the final block. Different $g$ rows are never added as fields, so reflection and mosaic-orientation coherence semantics are unchanged.

The pipeline is three stages:

1. **Prologue.** One thread owns one $(g, \text{segment})$ pair and computes the resonance kinematics, the shared-bracket interpolation of $\chi$, $U$, and $\mu$, both complex polarization amplitudes, the Beer–Lambert amplitude attenuation, the finite-time width, and the propagation phase. Output is fixed-order, g-major scratch. Rejected pairs get zero field coefficients, so no compaction step is needed and the layout remains deterministic.
2. **Field accumulation.** One CUDA block owns one $(g, \text{energy group})$. Threads stride over the current segment block, reduce the four field components in shared memory, and add one partial to each persistent cell. There is exactly one writer per field cell per launch, so no atomics are required.
3. **Finalize.** One thread owns a photon-energy bin, squares each completed $g$ row, applies its mosaic weight, and adds the incoherent row sum.

The streaming path holds prologue scratch for a fixed pair target plus four persistent $(N_g, N_E)$ field planes. It releases the scratch after each accumulation launch so the memory pool can reuse it.

CPU, `float64`, layered, grooved, and explicitly disabled paths use non-streaming implementations. CUDA-fp32 coherent `sinc_cutoff` requests use the same streaming kernel with a launch-uniform window branch. The `float32` reassociation this kernel introduces is [tracked as validation debt](precision-and-tolerances.md#float32-reassociation).

## Device residency

A case's incoherent line, coherent line, and bremsstrahlung kernels share one staged device copy of their segment arrays. Staging preserves the dtypes, elements, and order used by each kernel, so sharing the copy leaves the spectra bit-for-bit unchanged.

With `keep_segments_on_device=True`, the per-segment arrays remain on the CUDA device after transport. The driver concatenates compacted batches on the device, and staging casts the resident arrays as needed. This avoids downloading the segments and uploading them again for spectrum calculations.

Residency requires `transport_core="cuda"`. Incident phase-space diagnostics, groove-gap arrays, and scalar counts remain on the host; the spectrum kernels do not read them.

Resident segments increase peak device memory use and retain an allocation after transport returns. If the device cannot hold the payload, the driver catches the out-of-memory error and replays the same seed with segments downloaded to the host. Counter-addressed streams make that replay exact.

(what-acceleration-does-not-change)=

## What acceleration does not change

```{list-table} What is and is not bit-for-bit across cores.
:name: tbl-execution-bitwise
:header-rows: 1

* - Property
  - Status
  - How it is checked
* - Random stream, host versus device
  - bit-for-bit
  - integer-only arithmetic; asserted against an independent pure-Python reference
* - Control flow, branch structure, output addressing
  - identical
  - shared algorithm plus a signature-drift test
* - Result versus CUDA launch geometry, batch size, capacity replay
  - bit-for-bit
  - asserted across `nthreads` 32/128/512 and capacities 4/256/4096
* - CPU per-electron core versus CUDA kernel, per trajectory
  - **not** bit-for-bit
  - first-step agreement to `rtol=1e-12`
* - CPU per-electron core versus CUDA kernel, in aggregate
  - agrees
  - 8-seed observable comparison at 4σ
* - Per-electron core versus lockstep core
  - **not** bit-for-bit, by construction
  - 8-seed observable comparison at 4σ
* - Shared device staging and residency versus per-kernel upload
  - bit-for-bit
  - same dtypes, elements, and order; the cast happens exactly once
```

CPU and GPU trajectories can differ even with identical random draws. CUDA's `log`, `exp`, `pow`, `sin`, and `cos` use different implementations from the host libm. Transport is sensitive to rounding: a last-bit difference in one scattering angle diverges over the several hundred events an electron undergoes. The first recorded segment already passes through a logarithm, so even it agrees only to rounding rather than exactly.

The per-electron/lockstep difference is likewise structural. They draw from differently-ordered streams, so they realize different samples of the same distribution.

One deliberate measure-zero difference is documented in the code: elastic element selection recomputes per-element rates rather than buffering them, so the kernel needs no per-thread local array. A recomputation landing exactly on the sampled cumulative boundary could select a neighbouring element. The CPU reference recomputes identically, so the two cores agree with each other; only a buffered implementation would differ, and only at a boundary.

Enabling a new core changes numerical output, which under the [validation methodology](../validation/methodology.md) requires a `Validation:` id, a ledger row, and a golden regeneration. A spectrum pinned at $N_e > 1000$ on a CUDA machine is not reproduced by the lockstep core and must be regenerated or compared statistically.

## Fail-closed boundaries

The following unsupported combinations and invalid inputs raise errors:

* **Explicit CUDA transport for a grooved run.** The CUDA core does not support facet exit and re-entry. Automatic selection uses lockstep for grooved runs.
* **Flight-grouped incoherent CXR on a non-NumPy backend.** The segmented complex accumulation has no device equivalent, so substepped rows raise. The fallback would be the row-incoherent sum, which is a different physical claim.
* **Layered or component-resolved substepped grouping.** The per-layer refractive decrement along the escape path is not modelled, so these raise too.
* **Device residency on a non-CUDA core.** Only CUDA transport produces resident segment arrays.
* **An explicit accelerator backend that is unavailable, or whose device lacks native `float64`.** Raises; only `auto` may substitute.
* **A transport lookup table with a non-positive elastic rate.** Raises at build time rather than producing an infinite mean free path at run time.

Device out-of-memory handling can replay transport with downloaded segments. Counter-addressed streams preserve the result during that fallback.

## Validation

The ledger row `gpu-transport-core` records the core-selection and reproducibility checks. Reproduce the host-side checks with:

```bash
pyrite-dev test \
  tests/montecarlo/test_transport_per_electron.py \
  tests/montecarlo/test_transport_core_default.py \
  tests/montecarlo/test_segment_staging.py
```

`test_transport_per_electron.py` checks the transport kernel's properties. `test_transport_core_default.py` checks selection and routing with a patched device probe, so those checks need no GPU.
