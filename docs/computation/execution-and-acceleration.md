# Execution and acceleration

How a case actually runs, once the physics is fixed. This page covers the
choices a run makes about *where* and *how* it computes; the memory budgets and
scheduling topology those choices operate inside are
[their own page](memory-and-scheduling.md), and the floating-point policy is
[a third](precision-and-tolerances.md).

Nothing here changes a physical model. What it does change is the
**realization** — which sample of the modelled distribution a given run
produces. That distinction is load-bearing and is treated explicitly in
[what acceleration does not change](#what-acceleration-does-not-change).

## Two independent selections

A run makes two separate choices, and conflating them is the most common source
of confusion when reading a runtime plan.

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

The two interact in one direction only: the CUDA transport core requires the
resolved backend to *be* a CUDA device, because it is the only backend with a
`cupyx.jit` transport kernel. A ROCm or SYCL backend runs its spectrum kernels
on the device and its transport on the CPU.

### Backend resolution

`auto` probes in a fixed order — CuPy for CUDA or ROCm, then Intel SYCL through
`dpnp`, then NumPy — and takes the first that yields a usable device. The probe
is recorded: a backend that fell back to NumPy carries a `fallback_reason`
string naming why, which surfaces in the runtime plan rather than disappearing.

An **explicit** accelerator request never silently degrades. Asking for
`PYRITE_MC_BACKEND=cuda` on a machine with a ROCm CuPy build raises rather than
running somewhere else, and naming a backend whose device lacks `float64`
support while `PYRITE_FP64=1` is set raises for the same reason. Only `auto`
is permitted to substitute, and only with a logged warning.

Selection happens **once, at import**, and is process-scoped. This is deliberate:
the module-level `xp`, `REAL`, and `_GPU` bindings are read by every kernel, so
a mid-run switch would leave arrays from two backends in the same reduction. A
process that needs a different backend is a different process — which is exactly
how the worker pools pin their own choice (see
[memory, chunking, and scheduling](memory-and-scheduling.md)).

### Transport-core resolution

`auto` is the default and the only value that resolves to something else. It
selects the CUDA core when three conditions hold simultaneously:

1. the process has a CUDA device and the kernel module imports;
2. the run is **ungrooved** — grooved transport carries periodic facet
   exit/re-entry machinery that was never ported, and the combination is
   rejected rather than silently degraded;
3. the electron count exceeds `CUDA_TRANSPORT_MIN_ELECTRONS`, currently 1000.

Otherwise it resolves to the lockstep CPU core. The count tested is transport's
own, $\max(N_e, N_{e,\mathrm{brem}})$, since one transport serves both
populations.

Every **explicit** `transport_core=` is honored verbatim: a caller that names a
core gets that core or an error, never a substitution. This is what makes
`PYRITE_MC_TRANSPORT_CORE` usable to reproduce a pre-threshold result or to
bisect a host/device difference.

The threshold of 1000 sits at the conservative end of a measured crossover band
of $N_e \approx 500\text{–}1000$, established for both a light and a heavy
material. Below it, launch and staging overhead outweigh the kernel — at
$N_e = 250$ for MoSe₂ the device core runs at $0.46\times$ the CPU core. Above
it the device pulls away monotonically. Choosing the conservative end means no
run that would have been faster on the CPU core is moved off it. The
`--ne-line` default of 450 therefore stays on the CPU core; a 20 000-electron
run does not.

## The three transport cores

All three implement the same physics. They differ in how they traverse the
electron ensemble, and therefore in what order they consume randomness.

`lockstep`
: The historical core. An outer step loop over an inner electron loop, with a
  single NumPy `Generator` serving every electron. Draws are consumed in a fixed
  interleaved order. This is the default whenever `auto` does not select CUDA,
  and it is the core every pinned golden result was generated on.

`per-electron`
: One electron transported to completion before the next. Same models, same
  draw semantics, different consumption order. It exists as the **CPU reference
  for the CUDA kernel** — a host implementation that addresses randomness
  identically, so a host/device difference can be attributed to floating-point
  arithmetic rather than to algorithm. It is measurably *slower* than lockstep
  on the CPU (0.27–1.05× across the measured grid): counter-addressed draws and
  worse cache locality cost more than the restructuring saves. The win is the
  device, not the algorithm.

`cuda`
: The `cupyx.jit` kernel. One thread per electron, run to completion.

### Why run-to-completion

The lockstep ordering is exactly what a GPU cannot reproduce. The two
conventional fixes both destroy verifiability: per-thread cuRAND states make
results depend on how threads were assigned, and an atomic output counter makes
segment order depend on scheduling. Either would force a full golden
regeneration on every launch-configuration change.

The kernel instead **addresses** randomness rather than streaming it, so a
thread computes its own draws from its electron index without any shared
generator. That construction, and the bit-for-bit host/device identity it buys,
is documented in [random number streams](random-streams.md).

### Output addressing and capacity replay

Segment $s$ of local electron $i$ is written to slot $i \times \mathrm{cap} + s$
— a pure function of the electron index, with no atomics. After the kernel, a
boolean mask built from the per-electron segment counts compacts the batch into
electron-major, step-minor order.

The counter records the **true** count even when it exceeds the capacity: the
thread keeps transporting but stops writing. The driver then grows the capacity
and replays the batch. Replay is exact because the streams are
counter-addressed, so an overflow costs time and nothing else.

This is what permits a modest default capacity of 512 slots instead of the
`max_steps` worst case of 20 000. The CPU path gets away with allocating the
worst case only because Linux never faults in the untouched pages; a device
allocator has no such luxury.

Because the true count is always reported, every batch — including one that
overflowed — states exactly what the material needs. Two policies use that:

* **Replay at the measured capacity.** One replay at the reported requirement
  always suffices, so no growth factor is applied. Capacity is reset after every
  batch to the running maximum times a headroom factor of 1.25, in both
  directions. Capacity trades against batch size out of a fixed byte budget, so
  an over-provisioned capacity costs launches exactly as an under-provisioned
  one costs replays.
* **A short first batch.** A probe of 1024 electrons caps the width of the first
  batch — the one batch still sized by the initial guess. Its segments are kept,
  since output slots are addressed by electron index, so a short first batch is
  simply a short batch. This bounds what a wrong initial capacity can cost.

The probe is nearly free on a GPU, where a batch costs about one electron
lifetime whatever its width, and worth a third of the runtime on the CPU
per-electron core, where the loop is serial and a discarded batch costs its
electrons.

None of `seg_capacity`, `scratch_budget_bytes`, `capacity_headroom`,
`max_batch`, or `probe_electrons` changes results. They set memory and launch
behavior only.

## The transport lookup table

Both ungrooved cores read a per-run table of every energy-dependent scalar in
the transport hot loop, rather than recomputing transcendentals per event.

The table is built once per run from the layer compositions and covers, per
layer and energy:

* the total elastic rate,
* the stopping power $dE/ds$,
* $1/\beta$,
* the collision-element cumulative distribution,
* the elastic screening parameter $\alpha$.

The grid is **uniform in kinetic energy**, deliberately. A uniform grid turns
table lookup into one multiply, one integer conversion, and a linear
interpolation:

```{math}
:label: eq-execution-lut-index

x = (E - E_{\min})\,\Delta E^{-1}, \qquad
i = \lfloor x \rfloor, \qquad
f = x - i,
```

with the interpolated value $T_i + f\,(T_{i+1} - T_i)$. A logarithmic grid would
track the physics more naturally — rates vary as powers of $E$ — but would cost
a logarithm per lookup in the innermost loop, which is the operation the table
exists to remove. Indices are clamped at both ends, so an energy outside the
grid extrapolates flat rather than reading out of bounds.

Grid spacing targets 0.025 keV, bounded below by 256 points and above by 16 384
so an unusually wide energy range cannot blow up memory. The alpha column is the
one entry still built by interpolation rather than closed form: Mott screening
parameters come from tabulated $(\log E, \log \alpha)$ pairs, interpolated in log
space and exponentiated onto the linear grid.

The builder fails closed on a non-positive or non-finite total elastic rate
rather than emitting a table that would produce an infinite mean free path.

The table's own interpolation error is a systematic, not a statistical, effect,
and it sits well below the step-control effects quantified under
`energy-step-convergence`; see
[statistical methods](statistical-methods.md#null-results).

## The coherent streaming kernel

The coherent line sum is the one reduction whose *conditioning*, not just its
cost, motivated its structure.

For reflection/orientation row $g$, polarization $p$, and photon energy $E$, the
kernel accumulates a complex field

```{math}
:label: eq-execution-coherent-field

F_{gpE} = \sum_j c_{gpj}\,
  \operatorname{sinc}\!\bigl(a_{gj}(E - E_{r,gj})\bigr)\,
  \exp\!\bigl[i\,(s_j E - \varphi_{gj})\bigr],
```

and the spectrum contribution is
$\sum_g w_g\,(|F_{gsE}|^2 + |F_{gpE}|^2)$.

Streaming rests on one identity: the field is linear in the segment index, so

```{math}
:label: eq-execution-streaming-identity

F_{gpE} = \sum_{\text{blocks}} F^{\text{block}}_{gpE}.
```

**Intensity is never reduced per block.** Four persistent field planes — sigma
and pi, real and imaginary — are updated across segment blocks and squared only
after the final block. Different $g$ rows are never added as fields, so
reflection and mosaic-orientation coherence semantics are unchanged.

The pipeline is three stages:

1. **Prologue.** One thread owns one $(g, \text{segment})$ pair and computes the
   resonance kinematics, the shared-bracket interpolation of $\chi$, $U$, and
   $\mu$, both complex polarization amplitudes, the Beer–Lambert amplitude
   attenuation, the finite-time width, and the propagation phase. Output is
   fixed-order, g-major scratch. Rejected pairs get zero field coefficients, so
   no compaction step is needed — a deliberate trade of a little wasted
   bandwidth for a fully deterministic layout.
2. **Field accumulation.** One CUDA block owns one $(g, \text{energy group})$.
   Threads stride over the current segment block, reduce the four field
   components in shared memory, and add one partial to each persistent cell.
   There is exactly one writer per field cell per launch, so no atomics are
   required.
3. **Finalize.** One thread owns a photon-energy bin, squares each completed $g$
   row, applies its mosaic weight, and adds the incoherent row sum.

Bounded memory is the point of the arrangement. The predecessor retained roughly
eight values for every kept $(\text{segment}, g)$ pair until the end of the case;
the streaming path holds prologue scratch for a fixed pair target plus the four
persistent $(N_g, N_E)$ planes, and releases the scratch after each accumulation
launch so the pool can reuse it.

The CPU, `float64`, layered, grooved, and explicitly disabled paths retain their
prior implementations. CUDA-fp32 coherent `sinc_cutoff` requests use the same
streaming kernel with a launch-uniform window branch. The `float32`
reassociation this kernel introduces is
[tracked as validation debt](precision-and-tolerances.md#float32-reassociation).

## Device residency

A device-transported run keeps its segments where they were produced.

The motivation is measured rather than assumed. Before residency, transport
copied every segment down and each spectrum kernel copied what it needed
straight back up. A case runs three such kernels over the same segment
dictionary — one incoherent line sum, one coherent line sum, one bremsstrahlung
sum per radiating layer — and none of them shared a device copy. The accounting
was exact: 82 bytes per segment down, then 116 bytes per segment back up, for
data that was already on the device.

Two fixes landed, cheapest first.

**Share one device copy across a case's kernels.** The three calls uploaded
overlapping arrays 24 times; their union is 48 bytes per segment uploaded once.
Staging that copy needed nothing from transport, because every kernel already
reached for its arrays through a cast that returns a correctly-typed device
array untouched. The cast is the same one each kernel performed, hoisted to
happen once, so the spectra are bit-for-bit identical. Measured: −39 % on the
spectrum phase.

**Never come down.** With `keep_segments_on_device=True`, the per-segment arrays
of the frozen schema stay where the CUDA kernel made them, and staging only
casts them in place. The driver joins batches with one concatenate instead of
copying each compacted batch into host buffers. Measured over a whole case:
−61 % end to end, with the segment traffic in both directions falling to zero
and the spectra bit-for-bit identical to the downloading arm.

Two requirements follow, and both fail closed rather than pretending:

* Residency requires `transport_core="cuda"` and refuses anything else.
* The incident phase-space diagnostics, groove-gap arrays, and every scalar
  count stay NumPy, because they are per-electron or scalar and no kernel reads
  them.

The cost is device memory: the segments remain resident, so peak pool usage
grows and a substantial allocation is still held when the call returns. If the
device cannot hold the resident payload, the OOM is caught and the same seed is
replayed with the segments downloaded. Counter-addressed streams make that
replay exact, so the fallback costs bus time, not the result.

Pinned host staging is **not** a cheaper substitute on the upload path: copying
into a pinned buffer first measured 3.8–4.0 GB/s against 6.1–6.3 GB/s pageable,
because the staging copy costs more than the pageable-transfer overhead it
removes.

(what-acceleration-does-not-change)=
## What acceleration does not change

This is the section to read before comparing two runs.

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

The CPU/GPU trajectory difference is not a defect and cannot be engineered away.
CUDA's `log`, `exp`, `pow`, `sin`, and `cos` are accurate to a few ulp but are
not the host libm's implementations, and transport is chaotic: a last-bit
difference in one scattering angle diverges over the several hundred events an
electron undergoes. The first recorded segment already passes through a
logarithm, so even it agrees only to rounding rather than exactly.

The per-electron/lockstep difference is likewise structural. They draw from
differently-ordered streams, so they realize different samples of the same
distribution.

One deliberate measure-zero difference is documented in the code: elastic
element selection recomputes per-element rates rather than buffering them, so
the kernel needs no per-thread local array. A recomputation landing exactly on
the sampled cumulative boundary could select a neighbouring element. The CPU
reference recomputes identically, so the two cores agree with each other; only a
buffered implementation would differ, and only at a boundary.

**What this means in practice.** Enabling a new core changes numerical output,
which under the [validation methodology](../validation/methodology.md) requires
a `Validation:` id, a ledger row, and a golden regeneration. A spectrum pinned at
$N_e > 1000$ on a CUDA machine is not reproduced by the lockstep core and must
be regenerated or compared statistically.

At sweep scale the port is stronger than "statistically indistinguishable". Over
72 cases at $N_e = 20\,000$, the CUDA core and the CPU `per-electron` core agree
to 0.000 % on every recorded field, and the difference against lockstep is
*exactly* the difference the CPU per-electron core already produces. The stream
change itself is not small — median 0.43 % on the line spectrum, 4.01 % on
bremsstrahlung, 8.20 % on the coherent spectrum, each tracking $1/\sqrt{N}$ for
its own population — which is precisely why it must be read as a realization
change rather than a physics change.

## Fail-closed boundaries

Several reductions raise rather than silently substituting a different
algorithm. Each is a deliberate refusal, and the pattern is worth recognizing:
the alternative would silently answer a different question.

* **Grooved transport on a device.** Not ported; the combination is rejected
  rather than degraded to the CPU without saying so.
* **Flight-grouped incoherent CXR on a non-NumPy backend.** The segmented
  complex accumulation has no device equivalent, so substepped rows raise. The
  fallback would be the row-incoherent sum, which is a different physical claim.
* **Layered or component-resolved substepped grouping.** The per-layer
  refractive decrement along the escape path is not modelled, so these raise
  too.
* **Device residency on a non-CUDA core.** Refuses rather than pretending.
* **An explicit accelerator backend that is unavailable, or lacks `float64`
  under `PYRITE_FP64=1`.** Raises; only `auto` may substitute.
* **A transport lookup table with a non-positive elastic rate.** Raises at build
  time rather than producing an infinite mean free path at run time.

The one place a fallback *is* silent-by-design is the device OOM path, and it is
safe precisely because counter-addressed streams make the replay exact.

## Validation

The behavior on this page is pinned by the ledger row `gpu-transport-core`,
which records what the core selection changes and what it does not. Reproduce
the host-side checks with:

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test \
  tests/montecarlo/test_transport_per_electron.py \
  tests/montecarlo/test_transport_core_default.py \
  tests/montecarlo/test_segment_staging.py
```

The kernel's own properties live in `test_transport_per_electron.py`; the
selection policy and its routing consequences live in
`test_transport_core_default.py`, which patches the device probe and needs no
GPU, because what is under test there is the decision rather than the kernel.
