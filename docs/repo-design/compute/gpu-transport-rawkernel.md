# GPU electron transport RawKernel

## Why now

[the performance history](compute-performance-optimization.md) ("GPU transport: not yet") deferred this on the grounds that single-core transport was ≈1× the GPU phase per case, so the `gpu-pipeline` engine hid it at two or more transport workers. It named the condition for revisiting: *the prologue rawkernel lands and a fresh end-to-end profile shows the available transport workers no longer keep the GPU fed.*

The Round 3 remeasurement recorded that condition. Transport is now **2.4–3.4× the GPU phase for the TMDs and 6–12× for hopg**, per-segment cost is material-independent and linear in `n_seg` with zero OOM retries, and the MoSe2 `--ne-line=20000` investigation attributed the long GPU-0% window to transport compute plus transport result transfer, in roughly equal parts.

Two consequences shaped this design:

- Transport is genuinely compute-bound, not scheduling-bound, so moving the compute is a real lever rather than a reshuffle.
- Roughly half the idle window is *returning the segment payload* to the driver. Transport that already runs on the device puts the segments where the line and brem kernels want them, which attacks the transfer half as well. This module does not yet exploit that (see "Not done"), but the design leaves room for it.

## Scope

Ported: `_transport_core_ungrooved`, including the multilayer stack, the finite `(width, height)` prism footprint, both elastic models (Browning/Mott and analytic screened Rutherford), and per-electron energy cutoffs.

Not ported: grooved transport. `_transport_core_grooved` carries the periodic facet exit/re-entry machinery and its own event budget; it stays on the CPU and `simulate_trajectories` rejects the combination rather than silently degrading.

## The ordering problem, and the RNG

`_transport_core_ungrooved` is written lockstep over electrons — an outer step loop, an inner electron loop — for one reason: a single NumPy `Generator` serves every electron, so the draws must be consumed in a fixed interleaved order. That order is exactly what a GPU cannot reproduce. One thread per electron runs each electron to completion, so it consumes its stream contiguously.

The usual fixes are per-thread cuRAND states (which makes results depend on how threads were assigned) or an atomic output counter (which makes segment order depend on scheduling). Both would make the port unverifiable except statistically, and both would need a full golden regen on every launch-config change.

Instead, randomness is *addressed* rather than *streamed*. Draw `c` of electron `e` is

```text
key_e   = splitmix64(seed + PHI * (e + 1))
u(e, c) = (splitmix64(key_e + PHI * (c + 1)) >> 11) * 2**-53
```

with `PHI = 0x9E3779B97F4A7C15` and SplitMix64's finalizer (Steele, Lea & Flood, OOPSLA 2014), which is a bijection on 64 bits and passes BigCrush in counter mode. Keys are built on the host so device code needs no 64-bit integer casts, and so both cores provably address the same streams.

This is pure integer arithmetic plus one exact `uint64 -> double` conversion (the shifted value is below `2**53`), so the generator is **bit-for-bit identical on host and device**. It also makes a run independent of:

- CUDA launch geometry (`nthreads`);
- batch size;
- capacity replays.

`tests/montecarlo/test_transport_per_electron.py` asserts all three, plus the generator against an independent pure-Python SplitMix64.

## Output addressing

Segment `s` of local electron `i` is written to slot `i * cap + s` — a pure function of the electron index, with no atomics. After the kernel, a boolean mask built from `seg_count` compacts the batch into electron-major, step-minor order.

`seg_count[i]` records the *true* count even when it exceeds `cap`; the thread keeps transporting but stops writing. The driver then grows `cap` and replays the batch. Replay is exact because streams are counter-addressed, so an overflow costs time and nothing else. This is what lets `cap` default to a modest 512 slots instead of the `max_steps = 20000` worst case: the CPU path gets away with `np.empty((Ne * max_steps, 3))` only because Linux never faults in the untouched pages, and `cupy.empty` has no such luxury.

`PerElectronTransportConfig` sets `seg_capacity` and a `scratch_budget_bytes` that bounds the dense `(batch, cap)` grid; neither changes results.

## What is and is not bit-for-bit

| Property | Status | How it is checked |
| --- | --- | --- |
| RNG stream, host vs device | bit-for-bit | integer-only; asserted against a pure-Python reference |
| Control flow, branch structure, output addressing | identical | shared algorithm, signature-drift test |
| CPU per-electron core vs CUDA kernel, per trajectory | **not** bit-for-bit | first-step agreement to `rtol=1e-12` |
| CPU per-electron core vs CUDA kernel, in aggregate | agrees | 8-seed observable comparison at 4σ |
| Per-electron core vs lockstep core | **not** bit-for-bit by construction | 8-seed observable comparison at 4σ |

The CPU/GPU trajectory difference is not a defect and cannot be engineered away. CUDA's `log`, `exp`, `pow`, `sin` and `cos` are accurate to a few ulp but are not the host libm's implementations, and transport is chaotic: a last-bit difference in one scattering angle diverges over the several hundred events an electron undergoes. The first recorded segment already passes through `log`, so even it agrees only to rounding rather than exactly.

The per-electron core differing from the lockstep core is likewise structural — they draw from differently-ordered streams, so they realize different samples of the same distribution.

**Consequence for validation.** Enabling either new core changes numerical output. That needs a `Validation:` id, a physics ledger row, and a golden regen, per `AGENTS.md`. The default remains `transport_core="lockstep"`, which stays bit-for-bit; nothing in this change alters it.

One deliberate measure-zero difference is documented in the code: the elastic element selection recomputes per-element rates rather than buffering them, so the kernel needs no per-thread local array. A recomputation landing exactly on the sampled cumulative boundary could select a neighbouring element. The CPU reference recomputes identically, so the two cores agree with each other; only a buffered implementation would differ, and only at a boundary.

## Kernel structure

One thread per electron; the outer lockstep loop is gone. Control flow uses only `while` with an explicit `running` flag, so nothing depends on `break` / `continue` support in the `cupyx.jit` transpiler. Device helpers return a single scalar each; `_first_prism_exit_scalar` and `_rotate_direction_scalar` return tuples on the CPU and are therefore inlined.

Arithmetic is fp64 and textually identical to the CPU core, so any divergence is attributable to libm alone. fp32 would roughly double throughput on consumer cards but would change trajectories qualitatively rather than at rounding scale; it is not offered.

Arrays are flattened to 1-D and indexed manually, matching the house style set by `line_prologue_jit_kernel`. Per-layer element data is padded to `(n_layers, max_elements)` with live lengths in `L_nel`.

## Expected behaviour, and what is not yet measured

The kernel compiles and its tests pass on an RTX 5080 (driver 610.47, CuPy 14.1.1, which is simultaneously the `cupy-cuda13x` floor and the current pin): 37 tests in 54.81s, including determinism, invariance to `nthreads` 32/128/512, capacity replay at cap 4/256/4096, first-step agreement with the CPU reference at `rtol=1e-12`, and aggregate agreement.

One porting note worth keeping. `cupyx.jit` follows C promotion in CUDA mode, so mixing the uint32 launch indices with a signed int32 scalar yields uint32 and then fails the `same_kind` cast of the signed operand. The index is made signed once at its source; the alternative — carrying uint32 through the whole kernel, as `line_prologue_jit_kernel` does — would have collided with the int32 layer tables and int8 exit codes.

## Measured throughput

RTX 5080, 30 keV into a stopping slab, catalog compositions, 3 repeats, medians, compile excluded. hopg runs 354 segments/electron here, MoSe2 880.

| material | Ne | lockstep CPU | per-electron CPU | cuda | cuda vs lockstep |
| --- | --- | --- | --- | --- | --- |
| hopg | 250 | 0.020 s | 0.022 s | 0.018 s | 1.11x |
| hopg | 1000 | 0.096 s | 0.091 s | 0.041 s | 2.32x |
| hopg | 4000 | 0.404 s | 0.963 s | 0.099 s | 4.10x |
| hopg | 16000 | 2.387 s | 4.416 s | 0.151 s | 15.86x |
| mose2 | 250 | 0.039 s | 0.082 s | 0.086 s | 0.46x |
| mose2 | 1000 | 0.163 s | 0.347 s | 0.102 s | 1.60x |
| mose2 | 4000 | 0.766 s | 2.838 s | 0.187 s | 4.10x |
| mose2 | 16000 | 6.154 s | 11.459 s | 1.269 s | 4.85x |

The crossover is `Ne` ≈ 500–1000 for both materials, so launch overhead stops mattering below production sizes. The 450-electron default sits on it.

Two results worth carrying forward. The per-electron core is *slower than lockstep on CPU* (0.27–1.05x): counter-addressed draws and worse locality cost more than the restructuring saves, so the win is the device, not the algorithm — which is what a `numba.prange` version would have to overcome. And lockstep's own cost per segment degrades with `Ne` (226 → 421 ns for hopg), so part of the headline 15.86x is the baseline getting worse.

MoSe2's weaker 4.85x is not divergence. Its per-electron tail exceeds the default 512-slot capacity, so the first batch — most of the run — completes, overflows, and replays at a larger capacity. Capacity and batch size trade against a fixed byte budget, so a bigger static default is not the fix (cap=4096 starves the GPU, and hopg's optimum is 1024 against MoSe2's 2048). See the next section: the run now measures the capacity it needs instead of being told.

## Capacity, without a tuning knob

The cores report the true segment count even for an electron that overflowed its slots, so every batch — including one that has to be replayed — says exactly what the material needs. Two changes use that.

**Replay at the measured capacity, not a growth factor.** The replay used to jump to `max(needed, 4 * cap)`. Because `needed` is exact, one replay at `needed` always suffices, and the `4x` only over-provisioned: capacity trades against batch size out of a fixed byte budget, so an over-provisioned capacity costs launches exactly as an under-provisioned one costs replays. `cap` is now reset after every batch to the running maximum times `capacity_headroom` (1.25), in both directions.

**A short first batch.** `probe_electrons` (default 1024) caps the width of the first batch, the one batch still sized by the initial guess. Its segments are kept — output slots are addressed by electron index, so a short first batch is just a short batch — which bounds what a wrong `seg_capacity` can cost.

Ne=16000, 3 repeats, medians. `tuned` is the best a human got by sweeping the static capacity; `full` and `probed` both start from the unchanged default of 512.

| material | arm | median | launches | electrons transported | settled cap |
| --- | --- | --- | --- | --- | --- |
| hopg | old static default | 0.139 s | 2 | 16000 | 512 |
| hopg | tuned (cap 1024) | 0.129 s | 2 | 16000 | 544 |
| hopg | full | 0.146 s | 2 | 16000 | 560 |
| hopg | probed | 0.150 s | 3 | 16000 | 560 |
| MoSe2 | old static default | 0.727 s | 7 | 28787 | 2048 |
| MoSe2 | tuned (cap 2048) | 0.467 s | 5 | 16000 | 1593 |
| MoSe2 | full | 0.491 s | 5 | 28787 | 1593 |
| MoSe2 | probed | 0.499 s | 6 | 17024 | 1593 |

MoSe2 goes from 0.727 s to 0.491 s (−32%) with no tuning, and lands on a tighter capacity (1593) than the hand-swept optimum, which is why it beats it on launches.

**The probe buys nothing on the GPU and a great deal on the CPU.** A GPU batch costs about one electron lifetime whatever its width — 16000 electrons fit in one concurrent wave on this card — so a discarded full batch and a probe cost the same single launch, and the probe is a ~2% premium (one extra launch when the guess was already right, as for hopg). On the CPU per-electron core the loop is serial, so a discarded batch costs its electrons: MoSe2 at Ne=8000 goes 2.571 s → 1.712 s (−33%), transporting 9024 electrons instead of 16000. One shared driver, one default, priced on the arm that cares.

## Where the time actually goes

Ne=16000, capacity 2048 so nothing replays, six launches per run. Kernel and device-compaction totals are from `nsys`; wall times are from unprofiled runs because `nsys` inflates host-side time several-fold. The kernel's own median launch duration is the same under both capacities (37.8 vs 37.9 ms for MoSe2), which is the cross-check that the device-side numbers are trustworthy.

| | hopg | MoSe2 |
| --- | --- | --- |
| wall | 0.188 s | 0.927 s |
| transport kernel | 0.050 s (27%) | 0.224 s (24%) |
| device-side compaction | 0.005 s (3%) | 0.072 s (8%) |
| device-to-host payload | 0.061 s (32%) | 0.146 s (16%) |
| unattributed host-side remainder | 0.072 s (38%) | 0.485 s (52%) |

The payload is 82 bytes per recorded segment — 0.46 GB for hopg, 1.15 GB for MoSe2 — moved over pageable memory at a measured 7.6–7.9 GB/s. The remainder is scratch allocation, mask construction, and NumPy assembly of the output arrays; it has not been attributed further. Round 5 added the NVTX ranges that split it (`cxr.transport.scratch` / `.compact` / `.join` and siblings); a capture using them is what closes this out.

**The kernel is no longer the bottleneck; the driver around it is.**

## Occupancy and divergence

`ncu` could not be used — GPU performance counters are admin-only on the box (`ERR_NVGPUCTRPERM`) and there is no passwordless sudo. Both quantities were obtained without counters instead.

Occupancy comes from the CUDA occupancy API against the compiled module. The kernel uses **92 registers per thread** and 40 B of local memory, so it is register-limited to 640 of the 1536 threads an SM can hold — **41.7% theoretical occupancy**, flat for `nthreads` ≤ 128 and falling to 33.3% above it. Wall time tracks that exactly (MoSe2, medians: 1.08 s at 64, 1.12 s at 128, 1.27 s at 256, 1.64 s at 512), so the default of 128 is already at the optimum.

Divergence is computed exactly rather than sampled. Thread `i` owns electron `e_start + i`, so a warp covers a contiguous block of 32 electrons and costs its longest-lived member; the wasted fraction is `1 - sum(n) / (32 * max(n))` over each block, where `n` is the per-electron segment count the run already reports.

| material | mean segments | p99 | max | warp-wasted | ceiling if removed |
| --- | --- | --- | --- | --- | --- |
| hopg | 353 | 406 | 448 | 11.9% | 1.14x |
| MoSe2 | 879 | 1220 | 1274 | 27.4% | 1.38x |

**This closes the persistent-thread work queue as an option.** Perfect load balancing would recover 27.4% of a kernel that is 24% of MoSe2's wall time — about 7% end to end, and half that for hopg — in exchange for a work-queue scheme that would break the index-addressed output slots the reproducibility guarantees rest on. Electron lifetimes are simply not spread widely enough: transport at fixed energy into a stopping slab has a fairly tight range distribution, so 32 neighbouring electrons finish at similar times.

The ranking that followed was: keep segments on the device, which attacks the D2H, most of the host remainder, and the spectrum phase's re-upload at once. That is done — see the handoff section. Pinned host memory was *not* the cheap partial it looked like, and is now moot.

## Selecting the core

`transport_core="auto"` is the default, and `resolve_transport_core` decides: the CUDA core when the process has a CUDA device, the run is ungrooved, and `Ne > CUDA_TRANSPORT_MIN_ELECTRONS` (1000); the lockstep CPU core otherwise. The count is the transport's own, `max(Ne, Ne_brem)` — one transport serves both populations. Every explicit `transport_core=` is honored verbatim, so a caller that names a core gets that core or an error, never a substitution; `PYRITE_MC_TRANSPORT_CORE` pins the choice for a whole process, which is how a run reproduces a pre-threshold result or bisects a host/device difference.

**Why 1000.** The measured crossover is `Ne ≈ 500–1000` for both a light and a heavy material (see "Measured throughput"): below it the launch and staging overhead outweighs the kernel — at `Ne=250` MoSe2 the device core is *0.46×* the CPU one — and above it the device pulls away monotonically, 4.1× at `Ne=4000` and 4.9–15.9× at `Ne=16000`. 1000 is the conservative end of that band, so no run that would have been faster on the CPU core is moved off it. The `--ne-line` default of 450 stays on the CPU core; a 20000-electron run does not.

**Where it runs.** A device-transported run gives up the `gpu-pipeline` engine: its premise is CPU transport hidden behind GPU work, and there is neither anything left to hide nor room for a second CUDA context on the card this process is already driving. `run_cases` therefore keeps such a run in the driver process, serially, and every worker process is kept off the device in `_worker_init` — the pin covers both pools and every call site inside a worker. `_worker_init` redirects only `auto` and `cuda`; it used to overwrite the variable unconditionally, which made `PYRITE_MC_TRANSPORT_CORE=per-electron` a no-op in any pooled run. The switch is all-or-nothing across a run's cases, since a mixed run would strand its CPU-core cases in the driver with nothing overlapping them; sweeps hold `Ne` fixed across the grid, so mixed runs are the exception.

Serial is also simply the faster arrangement once transport is on the device: the case pays one kernel instead of `n_seg`-proportional pipe traffic, and the segments then stay resident for the spectrum kernels (`keep_segments_on_device`, set on this branch alone — device arrays cannot be pickled out of a worker). If the device cannot hold the resident payload, `_transport_case` catches the OOM and replays the same seed with the segments downloaded: counter-addressed streams make that replay exact, so the fallback costs bus time, not the result.

**What it changes.** The realization, not the distribution — see `Validation: gpu-transport-core` in the ledger, and "What is and is not bit-for-bit" above. A pinned spectrum taken at `Ne > 1000` on a CUDA box is not reproduced by the lockstep core and must be regenerated or compared statistically.

At sweep scale this is now measured rather than argued. Over `coh_test`/hopg — 72 cases at `Ne=20000`, 144 (case, energy) records — the CUDA core and the CPU `per-electron` core agree to **0.000% on every recorded field**, and the difference against lockstep is *exactly* the difference the CPU per-electron core already produces. The port is not merely statistically indistinguishable from its specification; at production scale it reproduces it. The stream change itself is not small: `line.spec` median 0.43% / max 2.31%, `brem` median 4.01% / max 25.74% at `Ne_brem=150`, `spec_coherent` median 8.20% / max 37.91% — all tracking `1/√N` for their own population. See "Verification: whole-sweep A/B on qlmc" in [compute-performance optimization history](compute-performance-optimization.md).

## Not done

- **A capture that uses the transport NVTX ranges.** Round 5 added them (`cxr.transport.tables` / `.sample` / `.alloc` / `.core` / `.output`, with `.upload` / `.scratch` / `.launch` / `.capsync` / `.compact` / `.exitcodes` / `.join` nested inside the CUDA driver), replacing the `cxr.transport.line` / `cxr.transport.brem` names Round 1 invented and never shipped. Nothing has been profiled with them yet; that run is what attributes the host-side remainder above.
- **`numba.prange` over the per-electron core.** The same restructuring makes the CPU core trivially parallel, which is the cheap answer for the core-starved case the perf doc flagged. Not wired up here.
- **Grooved transport.**

## The transport → spectrum handoff

`simulate_trajectories` used to always return NumPy, so the CUDA core copied every segment down and each spectrum kernel copied what it needed straight back up with `xp.asarray(segments[k], dtype=REAL)`. `_spectrum_case` calls three such kernels over the *same* `segs` dict — one incoherent line sum, one coherent line sum when the profile asks for it, one brem sum per radiating layer — and none of them shared a device copy. The measurements below are that state; both fixes under them have since landed.

hopg, Ne=16000, 5.65M segments, RTX 5080, `REAL = float32`:

| phase | wall | segment H2D | share of the call |
| --- | --- | --- | --- |
| transport (cuda core) | 0.348 s | — (82 B/seg *down*) | — |
| lines, incoherent | 0.364 s | 226 MB / 0.207 s | 57% |
| lines, coherent | 0.088 s | 271 MB / 0.046 s | 53% |
| brem | 0.069 s | 158 MB / 0.026 s | 38% |

The per-segment bytes account exactly: 40 B/segment for the incoherent lines (`r_mid`, `v_hat`, `L_ang`, `E_keV`, `elec_id`), 48 for the coherent sum (adds `t_ang`, `t0_ang`), 28 for brem. So a case pushes **116 B/segment up** on top of the **82 B/segment the transport pulled down**, for data that was on the device already. The re-upload alone is 0.279 s, 80% of a whole transport. Steady-state bandwidth is ~6 GB/s pageable; the incoherent call's 0.207 s is inflated by first-touch pool growth, so ~0.12 s of the 0.279 s is bus time and the rest is allocation.

Two independent fixes, cheapest first:

1. **Share one device copy across a case's kernels.** *Done.* The three calls uploaded overlapping arrays 24 times; their union is 48 B/segment uploaded once. `_segments_on_device` stages that copy and `_spectrum_case` hands it to all three kernels, which needs nothing from transport: every kernel already reached for its arrays through `xp.asarray(..., dtype=REAL)`, and that returns a device array of the right dtype untouched, so the kernels are unchanged and the ones still called on host segments behave exactly as before. The cast to `REAL` is the same one each kernel did, hoisted to happen once — the spectra are bit-for-bit identical.

Same segment set, arms interleaved after a discarded warm-up, medians of 5:

   | arm | three kernels | H2D | copies |
   | --- | --- | --- | --- |
   | per-kernel upload | 0.176 s | 656 MB / 0.113 s | 24 |
   | one shared copy | 0.108 s | 283 MB / 0.047 s | 16 |

−39% on the spectrum phase, and the bytes land where the accounting predicted: 116 → 50 B/segment, the 48 B/segment union plus the two arrays the staging uploads whether or not this particular case reads them. The remaining 16 copies are the energy tabulations, not segments.
2. **Never come down.** *Done.* `simulate_trajectories(..., keep_segments_on_device=True)` returns the eight per-segment arrays of the frozen schema where the CUDA core made them, and staging then only casts them to `REAL` in place. `energy_model="midpoint"` adds `E_end_keV`, `t_end_ang`, `E_repr_keV`, the `flight_id`/`substep_id` identifiers, and the row-end `event_kind` to that set; the byte accounting above is quoted for the frozen default. The driver stops copying each compacted batch into the caller's host buffers and instead joins the batches with one `concatenate` — which is also what sizes the output, since the segment total is not known until the last batch has run. It requires `transport_core="cuda"` and refuses anything else rather than pretending; the incident phase-space diagnostics, the groove-gap arrays, and every count stay NumPy, because they are per-electron or scalar and no kernel reads them.

Whole case — CUDA transport plus the three staged kernels — hopg, Ne=16000, 5.65M segments, arms interleaved after a discarded warm-up, medians of 5:

   | arm | transport | spectrum | total | D2H | H2D | PCIe |
   | --- | --- | --- | --- | --- | --- | --- |
   | download, then stage | 0.158 s | 0.108 s | 0.265 s | 464 MB | 284 MB | 0.105 s |
   | device-resident | 0.042 s | 0.062 s | 0.104 s | 0 | 1.5 MB | 0.002 s |

−61% end to end: transport −73%, spectrum a further −43% on top of the shared copy. The bus goes quiet — the 1.5 MB left is energy tabulations, and not one segment byte moves in either direction. Values are untouched: same dtypes, same elements, same order, and the `REAL` cast still happens exactly once, so the spectra are bit-for-bit identical to the downloading arm.

The cost is device memory. The segments now stay resident: peak pool 810 → 1176 MB, and 509 MB is still held when the call returns against 0 before. Peak grows by less than the payload because the join runs while the batch scratch is being freed. MoSe2's segment count is ~2.5× hopg's, so budget accordingly, and note the arrays are `float64` — compacting to `REAL` at the join would nearly halve this, at the price of changing the dtype the function documents.

Pinned host staging is *not* a fix on the way up: copying into a pinned buffer first measured 3.8–4.0 GB/s against 6.1–6.3 GB/s pageable, because the staging copy costs more than the pageable-transfer overhead it removes. It would only pay if the arrays were produced into pinned memory in the first place — which, with residency, is now moot.

Both fixes are what the live sweep now gets: `run_case` transports on the CUDA core above the threshold in "Selecting the core", serially in the driver process, with the segments never leaving the device between transport and the three spectrum kernels.

## Reproducing the CPU-side checks

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test \
  tests/montecarlo/test_transport_per_electron.py \
  tests/montecarlo/test_transport_core_default.py \
  tests/montecarlo/test_segment_staging.py
```

The kernel's own properties are in `test_transport_per_electron.py` (51 tests: 36 run on the CPU, 15 skip without a device). The selection policy and the routing that follows from it are in `test_transport_core_default.py`, which patches the device probe and needs no GPU at all — what is under test there is the decision, not the kernel.
