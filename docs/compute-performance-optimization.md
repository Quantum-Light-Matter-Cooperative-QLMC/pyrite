# Compute performance optimization

Broad-strokes summary of the GPU/throughput work. Method, the levers found,
what landed, and what is left. Written in rounds; each round records its own
hardware, workload, and numbers.

- [Round 1 (2026-07-31)](#round-1-2026-07-31) — `xp.fuse` line-chain fusion and
  incremental checkpoint shards.
- [Round 2 (2026-08-06)](#round-2-post-rawkernel-re-profile-2026-08-06) —
  re-profile after the `cupyx.jit.rawkernel` reductions landed; the eager
  prologue is now the constraint.
- [Round 3 (2026-08-06)](#round-3-fused-line-prologue-implementation-2026-08-06)
  — fused deterministic line prologue, combined interpolation gather, and the
  measured brem launch adjustment.
- [Round 4 (2026-08-08)](#round-4-gpu-transport-2026-08-08) — transport moves to
  the device and stays there; the CUDA core becomes the default above 1000
  electrons and the run gives up the transport pool to keep it.

## Review verdict

The central diagnosis is correct: after the line and brem reduction kernels
landed, the next material GPU lever is the incoherent line prologue, not more
Numba around transport and not CuPy's reduction autotuner. The checkpoint-shard
design and the decision to keep transport on the CPU for now are also sound.

Three corrections were made during implementation:

1. `__fmaf_rn` does **not** reproduce a non-contracted
   `f0 + frac * (f1 - f0)` sequence; it deliberately performs one rounding
   after the multiply-add. Preserving the former blend requires separate
   `__fsub_rn`, `__fmul_rn`, and `__fadd_rn` operations.
2. Atomic survivor compaction makes output order scheduler-dependent. Since the
   downstream float32 reduction is order-sensitive, the implementation keeps a
   deterministic `(segment, g)` order and marks rejected pairs with zero
   weights; the reduction skips those slots before evaluating `sin`.
3. A 3x line-only speedup does not imply six transport workers. Using the
   measured line+brem totals gives roughly two workers at `Ne=2000` and three at
   `Ne=450`, before headroom. Re-profile the completed kernel before setting a
   new default.

## Round 1 (2026-07-31)

Work on `feature/compute-performance-optimization`. Numbers are from `nsys`
profiles on the lab box (`qlmc`, RTX-class single GPU) over the `hopg` test
profile (189 cases) unless noted; see `performance-profiles/hopg_test-*/`.

### Method

Profiling drove every change — no speculative optimization. Each candidate was
isolated with `nsys` NVTX ranges (`cxr.lines`, `cxr.brem`,
`cxr.lines.tab/accum`; **not** `cxr.transport.*` — this round claimed transport
ranges that did not exist in the tree, so a transport-side gap in a capture was
unlabelled and could not discriminate phases on its own. Round 4's attribution
used the `cxr.performance.v1` activity labels instead; Round 5 added the real
ranges) plus `cuLaunchKernel` counts
from `nsys-stats.txt`, then verified bit-for-bit or tolerance-bounded against
the golden suite before landing. See `docs/performance-profile-analysis.md` for
the analysis playbook. Guiding rule: **GPU utilization is evidence, not the
target — compute-weighted wall time is.**

### Where the time was going

Two dominant, unrelated levers surfaced on the `hopg` profile:

1. **Launch-bound spectrum kernels.** The batched incoherent line path fired a
   storm of tiny elementwise CuPy kernels — `cupy_multiply`, `cupy_add`,
   `take`, `where`, `searchsorted`, `clip` — ~82k `cuLaunchKernel` calls total.
   At single-digit GPU utilization the wall was dispatch-bound, not
   compute-bound: the GPU was starved, not busy.
2. **O(N²) checkpoint I/O.** `run_sweep` re-serialized the *entire growing*
   per-material store to `line.pkl`/`brem.pkl` on **every** finished config, so
   total bytes written over a sweep grew quadratically — ~16 s of a ~28 s wall
   (~55%), all serial CPU I/O with the GPU idle.

`cxr.brem` is genuinely compute-bound and was deliberately left alone.

### What landed

#### 1. Line-chain kernel fusion (`montecarlo/spectrum.py`)

Collapsed the batched incoherent line launch-storm into a handful of fused
kernels, all **bit-for-bit** with the prior op-by-op path:

- `_line_kin_core` (`xp.fuse`) — resonance frequency + photon kinematics
  (~15 elementwise launches → 1). Exact op order preserved (`x*x`, not `x**2`).
- **Shared interpolation index** — `chi`/`U` (real+imag) and `mu` all sample the
  same `E_res` on the same `E_tab_g` grid, so the bracket is computed **once**
  and each table is gathered off the shared index. Removes 4× redundant
  `searchsorted`/`clip`.
- `_line_weight_core` (`xp.fuse`) — escape factor + PXR prefactor with the
  absorption term folded in.

**Impact:** `cuLaunchKernel` 82,457 → 63,765 (−23%); `cxr.lines` 4.43 s →
3.61 s on `hopg`. Bigger on high-reflection materials (e.g. `mos2`, where
`cxr.lines` was ~18 s). `cxr.brem` unchanged, as expected. No new validation
debt — pure kernel-merge, existing `Validation: line-hkl-batch` marker intact.

This builds on the earlier GPU line-absorption tabulation fix (already on
`main`), which had already cut `cxr.lines` from ~70 s to ~8 s by ending a
per-hkl CPU `xraydb` sync stall.

#### 2. Incremental checkpoint shards (`run.py`, `_checkpoint_store.py`)

Records are immutable once a config completes, so there is no reason to rewrite
the whole store each time. Each finished config now writes exactly one
crash-safety **shard** (`<stem>/parts/<hash>.pkl`) — O(1) per config, O(N) per
sweep. At the end of the sweep (clean finish *or* budget stop) the shards are
folded **once** into the authoritative `<stem>/{line,brem}.pkl` monolith and the
`parts/` directory is cleared, so `slim`/`prune`/`archive`/`remote` and the
analysis app still see the unchanged on-disk contract.

Crash-safety improves rather than regresses: a hard kill leaves shards behind,
and `load` unions them over any stale monolith (shards win per `name`/`E0`).
Legacy `<stem>.pkl` callers keep the old whole-file path.

**Impact:** per-sweep checkpoint write goes from O(N²) whole-store re-pickle to
O(N) shards + one consolidation — an estimated 16 s → ~1–2 s on `hopg`
(pending an `nsys` rerun on the box to confirm).

### Verification

- Line fusion: 122/122 line goldens (chunk-invariance, montecarlo, mosaic,
  multilayer, coherent, faceting).
- Checkpoint: `test_run`/`profiles`/`prune`/`archive`/`slim`/`analyze`/`remote`
  green, incl. new shard-clear-on-consolidate, unconsolidated-shard recovery,
  and shard-over-stale-monolith override tests.
- `lint` + `typecheck` clean on both.

### Remaining levers (not yet taken)

- **`cxr.lines.tab`** — per-hkl `chi`/`U` `f1`/`f2` tabulation (~5.5 s on
  `hopg`). Next target, but carries the `f1` edge cusp + coherent goldens, so
  trickier than the absorption (`f2`/`mu`-only) fix.
- **Transport aggregate** — large in wall-clock terms but runs pipelined across
  the CPU worker pool; watch the memory-aware worker cap on high-beam-energy
  refine steps (OOM risk on the single-GPU box).
- **`cxr.brem`** — compute-bound; leave unless the kernel math itself changes.

Round 2 supersedes the first two of these: `cxr.lines.tab` was fixed by the
stacking prologue (host→device uploads hoisted out of the reflection loop), and
the transport/GPU balance is now quantified.

## Round 2: post-rawkernel re-profile (2026-08-06)

Investigation only; nothing landed. Re-profile of the incoherent line + brem
path after the `cupyx.jit.rawkernel` reductions
(`spectrum_jit_kernel.py`, `brem_jit_kernel.py`, `coherent_jit_kernel.py`)
became the default (`_USE_JIT_*_REDUCTION = True`). Question asked: is there
more to win from (a) more raw/elementwise/reduction kernels or wider fusion,
(b) `cupyx.optimizing.optimize`, (c) more Numba JIT or a GPU transport port?

Answer: (a) yes, ~3× on the line phase, but the target has moved — the
rawkernels are no longer the cost, the **eager-CuPy prologue that feeds them
is**. (b) and (c) are dead ends, with numbers below.

### Environment and workload

Local box `ALEX-DESKTOP` (not `qlmc`): RTX 3060 Ti 8 GiB, driver 610.47,
24 cores, 22 GiB RAM, `nsys` 2026.4.1. Single case from the `compute_test`
catalog profile, `mos2`, `MoS2 1mm pol=45 az=120 footprint=5x5mm`:
22 reflections × 5 mosaic-MC nodes (`N_g = 110`), 1018 line bins, 4001 brem
bins, incoherent path, `REAL = float32`, no layers, finite 5×5 mm footprint.

Method: `_transport_case` once, then repeated `_lines_for_segments` /
`_brem_wide_from_segments` on the same segments with explicit
`Device().synchronize()` around each call. **The 3060 Ti idles at 210 MHz SM
and takes seconds to reach boost**, which alone produced ±25% spread; every
timing below is preceded by a 4 s GPU burn-in, reported as min and median of
7–9 reps, with A/B arms interleaved across separate processes.

### Where the time goes now

Phase split, warm, seconds per case:

| `Ne` | `n_seg` | transport (1 core) | lines | brem |
| --- | --- | --- | --- | --- |
| 450 | 1.07 M | 0.35 | 0.200 | 0.055 |
| 2000 | 4.50 M | 1.05 | 0.79 | 0.27 |
| 10000 | 22.6 M | 5.46 | 4.08 | 1.50 |

All three scale linearly in `n_seg`, and single-core transport holds at
≈0.98 × the whole GPU phase at every `Ne` — the ratio is `Ne`-independent, so
there is no electron count at which the balance flips on its own.

`nsys` at `Ne = 450` (3 reps of lines + brem): **3703 `cuLaunchKernel` calls per
lines call**, 166 ms/rep GPU-busy against 270 ms/rep wall. Kernel time share:

| Kernel | Share | What it is |
| --- | --- | --- |
| `_kernel_2e` (brem) | 24.3% | brem reduction rawkernel |
| `_kernel_3e` (lines) | 22.1% | line reduction rawkernel |
| `cupy_prepare_array_indexing` | 11.1% | fancy indexing in `_interp_gather2d` |
| `cupy_add` | 6.5% | `A2`/`A2_pxr`/`A2_cbs` accumulation |
| `cupy_take` | 6.0% | fancy indexing in `_interp_gather2d` |
| `cupy_multiply` | 4.7% | polarization dot products |
| `cupy_subtract__int64` | 3.9% | `idx - 1` interp bookkeeping |
| `_line_amp_sq_core` | 3.9% | fused amplitude core |
| `cupy_where` | 3.3% | interp endpoint clamps |
| `cupy_subtract__float32` | 3.0% | interp blend |

**The two rawkernels are only 46% of GPU-busy; ~50% is the eager per-seg-block
prologue in `mc_spectrum`'s batched incoherent branch** — interp gathers,
polarization dots, and boolean-mask compaction, none of which existed as a
target when round 1 was written.

### Confirmed lever: fuse the prologue

Prototype (in-memory source patch, not landed): replace `_interp_gather2d` /
`_interp_gather1d` with one `cupy.ElementwiseKernel` doing flat-index
gather+blend, eliminating all fancy indexing. Three interleaved A/B runs at
`Ne = 2000`, min of 7 reps:

| Arm | Run 1 | Run 2 | Run 3 |
| --- | --- | --- | --- |
| base | 0.837 s | 0.826 s | 0.805 s |
| gather EK | 0.642 s | 0.607 s | 0.596 s |

**−25% on the line phase**, reproducible. Not bit-for-bit: 4.8e-9 relative,
from `nvcc` contracting `a + b*c` into an `fma`. Either force separate
`__fsub_rn`/`__fmul_rn`/`__fadd_rn` operations to retain the former rounding,
or carry it as the same class of debt as
`Validation: line-amplitude-fusion` (ledger row + golden regen).

Re-profiled with the gather EK in place (`Ne = 2000`, per rep, burn-in
excluded): the line reduction rawkernel is 158 ms of 352 ms GPU-busy (45%),
and the remaining 194 ms is still prologue — `cupy_add` 30 ms,
`_line_amp_sq_core` 26 ms, `cupy_multiply` 18 ms, residual `_interp_index`
bookkeeping (`searchsorted` + `clip` + `take` + int64 subtract) ~35 ms, mask
compaction (`scan_naive` + `getitem_mask` + `bitwise_and` + `greater` +
`isfinite`) ~30 ms.

**Recommended next change:** one `cupyx.jit.rawkernel` covering the *whole*
prologue — per `(segment, g)` pair compute kinematics → interpolation → both
polarizations → weight and feed the existing `E_r`/`aw`/`w` reduction interface.
The original prototype proposed atomic survivor compaction; Round 3 instead
uses deterministic fixed-order buffers with zero-weight rejection. That
collapses the eager prologue to one launch per segment block and removes its
full-size intermediates without introducing run-to-run ordering noise.
Extrapolating the measured split, the earlier target was 0.83 s → 0.25–0.30 s
at `Ne = 2000` (≈3×), but that remains a hypothesis until the implemented
fixed-order variant is timed on the profiling hosts.

### Measured non-levers

Each of these was prototyped and timed; none is worth taking.

- **`seg_block` sizing.** The path is memory-traffic-bound, not launch-bound,
  so the obvious "batch harder" knob goes the wrong way. Against a base median
  of 0.827 s, a 16 M-element budget costs **+27%** (1.05 s) and 250 k costs
  **+23%** (1.01 s). The current `max(1, 1_000_000 // N_g)` sits at the
  optimum — leave it alone.
- **Dropping the `components=False` PXR/CBS dead work.** `_line_amp_sq_core`
  always computes and returns `a2_pxr`/`a2_cbs`, and the caller always
  accumulates them. A `components`-gated variant measured 0.806/0.804 s against
  a base of 0.795/0.914 s — inside the noise. A naive skip-the-accumulate patch
  was 13% *slower* (memory-pool reuse). Not worth the branch.
- **`cupyx.optimizing.optimize`.** It only tunes CuPy's own
  `ReductionKernel`-backed routines, and this path has almost none: `cub_any`
  0.1%, `argmin` 0.2%, `min` 0.2%, `bsum_shfl` ~0.2% — **under 1% of GPU-busy
  in total**, so the ceiling is ~0.5%. It also needs `optuna`, which is not a
  dependency. Skip.
- **More Numba JIT.** `cProfile` of `_transport_case` at `Ne = 2000`: 0.911 s of
  0.912 s inside `simulate_trajectories`, 645 Python calls for the whole phase.
  Coverage is already complete; there is no interpreter overhead left to
  remove. The only levers are `prange` or a GPU port, below.

### Marginal: rawkernel launch configs

12-point sweep of `nthreads × energies_per_block` at `Ne = 2000`, min of 5:

- Lines (`SpectrumKernelConfig(512, 3)` today): best 256/3 at 0.717 s vs 0.747 s
  — ~4%, at the noise floor. Not worth touching.
- Brem (`BremKernelConfig(256, 2)` today): best 512/3 at 0.219 s vs 0.244 s —
  ~10% of the brem phase, ≈2% of a case. Real but small.

Caveat: changing `nthreads` changes the shared-memory reduction tree and is
**not** bit-for-bit; `energies_per_block` alone is. If only one is taken, take
brem `energies_per_block` 2 → 3.

### GPU transport: not yet

> **Superseded by [Round 4](#round-4-gpu-transport-2026-08-08).** Both revisit
> conditions below were met and the port landed; the CUDA core is now the
> default above 1000 electrons. The reasoning is kept as the Round 2 record.

Not warranted now, and the reason is the pipeline rather than the kernel.
Single-core transport is ≈1× the GPU phase per case, so the `gpu-pipeline`
engine hides it completely at two or more transport workers; the
`_PIPELINE_WORKER_MEM_MB` split (transport-only workers no longer charged the
full-case `_WORKER_MEM_MB` budget) is what makes that reliably reachable. It
becomes worth revisiting only when either

1. the prologue rawkernel lands and a fresh end-to-end profile shows the
   available 2–3 transport workers no longer keep the GPU fed; or
2. a run is core-starved (single-core SLURM allocation, or an interactive
   single case, where transport is ~50% of wall).

If it is taken: `_transport_core_ungrooved` is already written lockstep over
electrons, so a `numba.cuda` port maps directly (one thread per electron,
lockstep loop dropped). The costs are per-electron cuRAND streams replacing the
single shared `Generator`, and an atomic `nseg` counter for segment output.
Both change segment order, so neither is bit-for-bit — ledger row plus a full
golden regen. `numba.prange` over the electron loop buys the core-starved case
far more cheaply, with the identical RNG-stream problem and the caveat that it
oversubscribes against the process pool.

### Reproducing

No new tooling was added. The harness is `_transport_case` once followed by
repeated `_lines_for_segments` / `_brem_wide_from_segments` on the returned
`tp["segs"]`, with `coherent=False`. Anyone repeating this must keep the GPU
burn-in and the interleaved A/B arms — without them the clock ramp swamps every
effect reported above. Prototype variants were applied by re-`exec`-ing a
patched `spectrum.py` source into the imported module before importing
`runner`, so the working tree stayed clean.

### Suggested order

1. Gather `ElementwiseKernel` (−25% on lines, small and isolated) — use separate
   round-to-nearest arithmetic intrinsics if the existing blend order must be
   retained.
2. Full prologue rawkernel (est. ≈3× on lines) — the actual prize.
3. Brem `energies_per_block` 2 → 3 (~2% of a case, bit-for-bit).
4. Re-measure the pipeline balance; only then reconsider transport `prange` or
   a GPU port.

### Scope limits

Measured on one material, one geometry, one GPU, and the incoherent line path
only — the coherent path was under concurrent work and deliberately left out.
The prologue result is shared code and should generalize; the −25% and ≈3×
figures are for this workload and should be re-measured on `qlmc` before being
quoted as production numbers.

## Round 3: fused line prologue implementation (2026-08-06)

Implementation pass following the Round 2 review. No CUDA device was available
in the editing environment, so this round records code structure and static
verification only. Performance and GPU numerical validation remain release
gates; none of the Round 2 estimates are promoted to measured results here.

### What changed

#### 1. Deterministic full line prologue (`line_prologue_jit_kernel.py`)

The new float32 CUDA JIT kernel assigns one thread to each `(segment, g)` pair
and evaluates, in one pass:

- resonance and photon kinematics;
- the shared energy bracket plus chi/U/mu interpolation;
- both polarization amplitudes;
- Beer-Lambert attenuation, finite-time width, and mosaic-weighted line weight.

It emits three fixed-order arrays (`E_r`, `aw`, `w`). Rejected pairs retain
their deterministic slot with `w=0`; no atomic counter or mask compaction is
used. This costs three arrays of at most the existing one-million-pair block
budget, but eliminates the roughly forty eager temporaries and preserves stable
pair ordering. The path is staged behind `_USE_JIT_LINE_PROLOGUE = False` until
the GPU release gates below pass. When opted in, it is further gated to the
existing fast-path domain: CuPy, float32, incoherent, `components=False`,
`sinc_cutoff=None`, and the single-slab absorber branch (finite transverse
footprint remains supported). Every other case stays on the prior
implementation.

#### 2. Reduction kernel streaming (`spectrum_jit_kernel.py`)

The line reduction kernels now load the weight first and skip zero-weight pairs
before loading `E_r`/`aw` or evaluating `sin`. They can accumulate directly
into a caller-provided output, avoiding one extra CuPy add launch per segment
block. Fresh calls still allocate and return a zero-initialized spectrum, so
the previous API remains valid. The impossible 2048-thread launch option was
removed; CUDA blocks are capped at 1024 threads.

#### 3. One-launch interpolation fallback (`spectrum.py`)

The eager batched path now gathers chi/U real+imag and mu in one
`ElementwiseKernel`. Separate round-to-nearest subtract, multiply, and add
intrinsics retain the pre-existing interpolation operation order. This path is
still useful for the coherent batch branch, which deliberately does not use the
new incoherent prologue.

#### 4. Brem launch setting (`brem_jit_kernel.py`)

`DEFAULT_BREM_KERNEL_CONFIG.energies_per_block` changed from 2 to 3, the
bit-for-bit launch-only improvement measured in Round 2. The thread count stays
at 256, so the reduction tree is unchanged.

#### 5. Shared-transport repair consistency (`runner.py`)

The live runner already transports `max(Ne, Ne_brem)` once and selects the line
and brem populations by electron id, so the earlier duplicate-transport concern
has been resolved. The brem-only repair helper still used the obsolete
`seed + 1` convention, however; it now uses `case["seed"]` to reproduce the
shared live transport's brem population.

### Required GPU verification

Before enabling this in a release branch:

1. Compile each JIT specialization on the minimum and current supported CuPy
   versions.
2. Compare the prologue's `E_r`, `aw`, and nonzero `w` against the eager path on
   one-block synthetic inputs, edge-bracketing inputs, and full MoS2 cases.
3. Run the line golden suite and repeated-run determinism checks. Record max
   absolute error, max error/peak, significant-bin relative error, and integral
   drift.
4. Re-run the interleaved burn-in A/B harness at `Ne=450`, 2000, and 10000 on
   both ALEX-DESKTOP and `qlmc`. Capture line wall time, kernel count, survivor
   fraction, and fixed-order zero-slot overhead.
5. Re-measure transport/GPU overlap before changing process counts or starting
   a `prange`/CUDA transport project.

## Round 4: GPU transport (2026-08-08)

The round Round 2 deferred. Its two revisit conditions — the prologue lands and
a fresh profile shows the transport workers no longer keeping the GPU fed — were
both recorded in Round 3's remeasurement: transport came out at **2.4–3.4× the
whole GPU phase** for the TMDs and **6–12×** for hopg on an RTX 5080, linear in
`n_seg`, with zero OOM retries and 4 ms of checkpoint I/O. Transport, not the
line phase, was the constraint.

Design, kernel structure, RNG, and the full measurement tables live in
[`docs/gpu-transport-rawkernel.md`](gpu-transport-rawkernel.md). This is the
summary and the verdict.

### What landed

1. **A per-electron transport core**, run-to-completion, each electron drawing
   from its own counter-addressed SplitMix64 stream instead of one shared
   step-major `Generator`. Written as an `njit` CPU function first — it is the
   executable specification, so the port can be checked against it rather than
   only against statistics.
2. **The `cupyx.jit` port**: one thread per electron, textually identical
   arithmetic, output slots addressed by electron index (`i * cap + s`), no
   atomics. Same algorithm, two backends.
3. **Measured segment capacity.** The cores report an electron's true segment
   count even when it overflows its slots, so a batch that overflowed says
   exactly what the material needs. `cap` is reset from the running maximum
   after every batch, in both directions, and a short probe batch bounds what a
   wrong initial guess can cost. MoSe2 at `Ne=16000`: **0.727 s → 0.467 s
   (−32%)** with no tuning, settling tighter (1593) than the hand-swept optimum.
4. **One device copy of the segments per case.** Every spectrum kernel reached
   for its own slice with `xp.asarray(segments[k], dtype=REAL)`: 24 uploads,
   116 B/segment, where the union of what they read is 48. Staging one copy for
   the case is **−39% on the spectrum phase**, and bit-for-bit — it is the same
   `REAL` cast, hoisted to happen once.
5. **Device-resident segments.** `keep_segments_on_device=True` returns the eight
   per-segment arrays where the kernel made them. Whole case, hopg, `Ne=16000`,
   5.65 M segments: **0.265 s → 0.104 s (−61%)**, transport −73% and spectrum a
   further −43%. 464 MB D2H and 284 MB H2D become **zero and 1.5 MB** — not one
   segment byte crosses the bus. Costs device memory: peak pool 810 → 1176 MB.

### The default, and what it costs

`transport_core="auto"` now ships as the default and takes the CUDA core when
the process has a CUDA device, the run is ungrooved, and `Ne > 1000`. The
threshold is the measured crossover: below it the device core loses (0.46× at
`Ne=250` on MoSe2), above it it wins monotonically — 4.1× at `Ne=4000`, 4.9–15.9×
at `Ne=16000`. The `--ne-line` default of 450 is unaffected; the 20000-electron
runs that started this whole investigation are not.

A device-transported run **gives up the `gpu-pipeline` engine**, and this is the
point rather than a concession: the pipeline exists to hide CPU transport behind
GPU work, there is nothing left to hide, and a pool of worker processes would put
N CUDA contexts on the card the driver is already using. Such a run stays in the
driver process, serially, with the segments resident across the handoff. Workers
are pinned to the lockstep core in `_worker_init`, so the pin holds for every
call site inside a worker, in both pools.

The cost is that this **changes numerical output**: per-electron streams are a
different realization of the same distribution, and CUDA's libm differs from the
host's by a few ulp on a chaotic trajectory. It is ledgered as
`Validation: gpu-transport-core` (`filtered`) on aggregate agreement across
seeds, first-step agreement at `rtol=1e-12`, bitwise repeat-run determinism, and
invariance to batch size, launch geometry, and capacity replay. Any pinned
spectrum taken above the threshold on a CUDA box has to be regenerated;
`CXR_MC_TRANSPORT_CORE=lockstep` restores the old core process-wide.

### Verification: whole-sweep A/B on qlmc (2026-08-08)

Everything above is a per-case or per-phase measurement. This is the flip itself,
end to end, on `qlmc` (RTX 5080 16 GB, 32 logical cores, 45 GB RAM, idle box, no
SLURM allocation so the pipeline arm gets every core it asks for). Both arms run
the same `cxr run` invocation and differ only in `CXR_MC_TRANSPORT_CORE`: unset
(so `auto` resolves to the CUDA core, serial in the driver, segments resident)
against `lockstep` (the historical `gpu-pipeline` arm, 16 transport workers).
Fresh checkpoint directory per arm — `--perf` already bypasses the shared cache,
but a resumed arm would otherwise measure nothing. One warm-up rep per arm, then
three interleaved reps.

`coh_test` / hopg, 72 cases, `Ne=20000`, `Ne_brem=150`:

| | device arm | pipeline arm |
| --- | --- | --- |
| case loop, 3 reps | 12.03 / 12.02 / 12.14 s | 18.64 / 18.70 / 18.78 s |
| wall incl. startup | 13.88 s (median) | 20.48 s (median) |
| engine / workers | serial, 1 | gpu-pipeline, 16 |
| CPU burned | 11.8 core-s | 53.2 core-s |
| GPU utilization | 84% median | 37.5% median |
| GPU feed-wait | n/a | 0.103–0.110 |
| peak VRAM | 2751 MiB | 5039 MiB |
| peak host RSS (tree) | 0.76 GB | 12.7–13.4 GB |
| cases in flight | 1 | 18 |

**1.55× on the case loop, 1.48× on wall, for 4.5× less CPU and 17× less host
memory.** The spread is under 1% across reps, so the difference is not close.

The reason is not the one the design predicted. Feed-wait in the pipeline arm is
only ~11% — the transport pool *does* keep up — yet the card sits at 37% median
utilization against the device arm's 84%. What the pipeline pays for is
everything around the overlap: 18 cases in flight, each holding a host-side
segment payload (hence 13 GB of RSS), each uploading it before the spectrum
kernels can touch it. The device arm never forms those payloads. Overlap was
hiding a cost that residency removes outright.

**Numerical agreement, at sweep scale.** Comparing the arms' 144 (case, energy)
records field by field looked alarming at first — `brem` differing by up to 26%,
`spec_coherent` by 38%. A third arm settles it. Run serially with the CPU
`per-electron` core, the sweep matches the CUDA arm to **0.000% on every field**,
and differs from lockstep by *exactly* the figures above, digit for digit. All of
the change is the lockstep → per-electron stream change; none of it is the
device. Sized against `1/√N`: `line.spec` median 0.43% / max 2.31% at
`Ne=20000`, `brem` median 4.01% / max 25.74% at `Ne_brem=150`. Both arms are
run-to-run bitwise reproducible, and serial-lockstep matches pipeline-lockstep
exactly, so the engine does not enter the result.

That control also caught a defect in the pin: `_worker_init` overwrote
`CXR_MC_TRANSPORT_CORE` unconditionally, so a worker pool silently ran lockstep
even when the run was pinned to `per-electron` — the pin was a no-op for every
pooled run. It now redirects only `auto` and `cuda`, which are the values that
would open a second CUDA context; a deliberate CPU core is honored.

**MoSe2 device memory.** `promising` / mose2, 432 cases, `Ne=15000`, thickness to
50000 Å — the heaviest resident payload in the catalog, ~2.5× hopg's segment
count. Peak VRAM **6873 MiB of 16303** (median 4215), against a CuPy pool cap of
11412 MiB, with **zero OOM retries** and host RSS of 0.82 GB. The residency
fallback never fired; the margin on a 16 GB card is real but it is the case to
watch, and a smaller card would want `REAL` compaction at the join (below).

### Corrections to earlier rounds

- Round 1's method section credited NVTX ranges `cxr.transport.line` /
  `cxr.transport.brem`. **They did not exist**; `transport.py` pushed no
  `_nsys_push`/`_nsys_range` at all, so a transport-side gap in an `nsys`
  capture was unlabelled. The MoSe2 attribution used the shipped
  `cxr.performance.v1` activity labels, which do carry per-tick phase identity.
  Round 5 added real ranges, under different names than Round 1 invented — see
  that round below.
- Round 2's "single-core transport is ≈1× the GPU phase, so two transport
  workers hide it" was measured on a 3060 Ti at `N_g = 110`. On an RTX 5080 at
  `N_g = 66` the GPU phase shrank and transport did not; the ratio flipped.
  Balance claims do not survive a GPU generation.
- The MoSe2 `--ne-line=20000` "stall" is **not a stall**: it is the
  pipeline-fill transient plus compute-bound transport, linear in `n_seg`, with
  per-segment cost within 3.5% of hopg's. Feed-wait falls 0.530 → 0.299 as cases
  become available to overlap. Documented as expected; no default was changed
  for it — Round 4 removes the underlying cost instead.

### Still open

- NVTX ranges in `transport.py`. **Done in Round 5**; the rest of this list is
  unchanged and Round 5 turns it into a sequenced plan.
- **The `gpu-pipeline` engine's memory sizing.** Running `promising`/mose2 on the
  pipeline arm drove `qlmc`'s 45 GB to 46.8 GB of tree RSS and 8 GB of swap.
  That arm was left running to get a pipeline-vs-device total for MoSe2 and
  **never produced one — I stopped it**. It reached 319 of 432 cases in 5212 s
  while decelerating hard (21 s/case at case 282, 34 s/case by 298, 71 s/case by
  319, against the device arm's 1154 s for all 432), driving peak tree RSS to
  **50.3 GB and swap to 12.9 GB on a 45 GB box**, load average 49–76, and leaving
  the machine unreachable over ssh for ~15 minutes. It was `SIGTERM`ed at case
  319 rather than allowed to finish: it is a shared box. Treat those numbers as
  evidence about memory behavior, not as a timing — the measured MoSe2 figures
  elsewhere in this document are the device arm's.

  The partial run does carry one clean result. At MoSe2 scale the pipeline is
  genuinely **feed-starved**, which at hopg scale it was not: `gpu_feed_wait_fraction`
  runs a median of **0.569** (max 0.671) against hopg's 0.103–0.110. So the two
  materials fail the pipeline for different reasons — hopg by payload overhead at
  11% feed-wait, MoSe2 by a transport pool that cannot keep 16 workers ahead of
  the card once `n_seg` triples. It also holds *more* device memory than the
  resident arm it was supposed to undercut: peak VRAM **9405 MiB vs 6873**, with
  18 cases in flight throughout and 0 OOM retries.
  `_gpu_pipeline_workers`
  budgets *workers* (`_PIPELINE_WORKER_MEM_MB`) but nothing budgets the 18 cases
  in flight, each holding a host-side segment payload, and the worker count comes
  from `os.cpu_count()` — which ignores a SLURM cgroup, so a `--cpus-per-task=8`
  allocation on this box still spawns 16. The flip routes the heavy runs away
  from this path rather than fixing it; anything still using the pipeline (no
  GPU, `Ne ≤ 1000`, grooved) is still exposed.
- The host-side remainder of a transport call — scratch allocation, mask
  construction, output assembly — is now 38% of hopg's wall and 52% of MoSe2's.
  The kernel is no longer the bottleneck; the driver around it is.
- Compacting the resident segments to `REAL` at the join would nearly halve the
  device memory they hold, at the price of changing the dtype the function
  documents.
- `numba.prange` over the per-electron core, for the core-starved CPU case. Note
  it starts from a deficit: the per-electron restructuring is **0.27–1.05×** of
  lockstep on the CPU, so it needs more than two cores just to break even.
- Grooved transport, which stays on the lockstep core.

## Round 5 (2026-08-09)

Instrumentation, bookkeeping, and a plan. No kernel or numerical change: nothing
in this round moves a spectrum.

### What landed

1. **NVTX ranges in `transport.py`**, closing the oldest correction in this
   document. Round 1 credited `cxr.transport.line` / `cxr.transport.brem`, which
   never existed; the real names are per *stage*, not per radiation channel,
   because that is what the unattributed remainder needed splitting into.
   `simulate_trajectories` pushes `cxr.transport.tables`, `.sample`, `.alloc`,
   `.core`, `.output`; inside the CUDA driver, `.core` nests `.upload`,
   `.scratch`, `.launch`, `.capsync`, `.compact`, `.exitcodes`, `.join`.
   `.launch` and `.capsync` are deliberately split: on CUDA the core call
   returns immediately, so `.launch` is host-side dispatch and the batch's
   device time lands in the `seg_count.max()` read inside `.capsync`. Every
   range is host-side wall clock, and all are no-ops off a profiled run
   (`_nsys_push`/`_nsys_pop` from `runner.py`, imported lazily because `runner`
   imports `transport`). The playbook is corrected in
   [`performance-profile-analysis.md`](performance-profile-analysis.md).
2. **The four orphan line-path `Validation:` markers are ledgered**:
   `line-hkl-batch`, `line-amplitude-fusion`, `line-gemv-elementwise` (all
   `filtered`) and `line-absorption-tabulation` (**`discrepancy`**). The last is
   a real finding, not bookkeeping: the in-code justification that tabulated `μ`
   is "at least as accurate as the chi/U interpolation already accepted here" is
   measurably false near an absorption edge — hopg (002) at the C K-edge is off
   `2.72e-01` against `5.82e-05` for `χ_g` on the same grid and points, because
   `μ` depends on `f₂` alone so the edge jump is the whole of its variation.
   Edge-localized and median `1.6e-6`, but it propagates to `exp(−μL_esc)` at
   `5.23e-01` for `L_esc = 1e4 Å`. Resolutions are in the ledger row; none is
   taken here.
3. **`-c/--cpu` and `--cpu-only` on `cxr run -R/--remote`.** The CPU-profiling
   phase existed only on `cxr remote run`, which is deprecated and hidden with
   removal at 0.3.0, so the feature would have disappeared silently. Both flags
   require `-R`, imply `--perf`, and forward to the same job machinery;
   validation stays in `start_command`, unduplicated.

### Is the gated line prologue still worth anything? (`_USE_JIT_LINE_PROLOGUE`)

Re-derived rather than re-measured — **arithmetic on recorded numbers, not a new
measurement**, because it needs a CUDA box. Round 3 measured the prologue at
**4.2–8.9% of the line phase** (`Ne` 450 → 10000, mos2, `N_g = 66`, `qlmc`).

Before Round 4, the single-process split at `Ne=10000` was transport 1.273 s,
lines 0.3356 s, brem 0.1564 s — a 1.765 s case in which the line phase is
**19.0%** and the prologue's best case is worth **1.7% of case wall**. Round 4's
whole-case measurement (hopg, `Ne=16000`) is 0.265 s → 0.104 s with transport
−73% and spectrum −43%. Applying those factors to the mos2 split gives transport
0.344 s, lines 0.191 s, brem 0.089 s: the line phase's share of the case
**roughly doubles, 19.0% → 30.6%**.

What that does to the prologue depends on how its win scales:

- If the win scales with the line phase, it is **~2.7% of case wall** (up from
  1.7%).
- If it is unchanged in absolute terms, **~4.8%** — but this is the optimistic
  bound and probably wrong in the optimistic direction. The fused prologue's
  saving is largely gather and launch overhead, and device-resident segments
  already removed a chunk of exactly that.

**Answer: yes, still low single digits, and the honest range moved up rather
than down — roughly 1.6–2.8× its old share, not a different order of magnitude.**
That does not change the recommendation: 2–5% of wall does not buy a fifth
physics claim on a path that already owes four, one of which is a `discrepancy`.
The gate on any flip is a re-measured interleaved A/B under the Round-4
device-resident default, not this arithmetic.

### Plan for the remaining levers

Sequenced, because two of them are gated on the first:

1. **Use the new NVTX ranges** on a `--nsys` capture of hopg and MoSe2 to split
   the host-side remainder (38% of hopg's transport wall, 52% of MoSe2's) into
   `.tables` / `.sample` / `.alloc` / `.scratch` / `.compact` / `.join`. This is
   pure measurement and it decides the next two items. Needs one authorized
   remote job.
2. **`REAL` compaction at the join**, gated on (1) showing `.join` or `.compact`
   is material. It roughly halves the 509 MB held / 1176 MB peak pool, at the
   price of changing the dtype the function documents — so it is an API change,
   not just a perf change, and wants the 16 GB-card MoSe2 case (peak 6873 MiB) as
   its acceptance workload.
3. **`gpu-pipeline` memory sizing.** The only correctness-adjacent item on this
   list: a box can be driven into swap. Two independent defects, both cheap to
   state and neither fixed by Round 4's flip — nothing budgets the ~18 cases in
   flight (only workers are budgeted, via `_PIPELINE_WORKER_MEM_MB`), and worker
   count comes from `os.cpu_count()`, which ignores a SLURM cgroup. Round 4
   routed heavy runs off this path; everything still on it (no CUDA device,
   `Ne ≤ 1000`, grooved) remains exposed. Fix the cgroup read first — it is
   local, testable, and bounds the worst case on a cluster.
4. **Grooved transport on the CUDA core.** Scope, not difficulty: grooved runs
   still take the lockstep path.
5. **`numba.prange` over the per-electron core — deferred**, by decision. It
   starts from a 0.27–1.05× deficit against lockstep, so it needs more than two
   cores just to break even.

The missing MoSe2 pipeline arm stays deferred until (3), and should be rerun only
with an explicit memory cap on a box that is not shared.
