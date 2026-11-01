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

The batch size costs each slot at the enabled schema's allocated columns: 82 bytes frozen, 115 midpoint, 125 midpoint with coupled photons or shell hard events, 135 with both, and 24 more for transported secondary directions, plus 9 bytes of per-history count and exit code. Before [#611](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/611) every schema was costed at 82 bytes, so wider schemas overran the budget by up to 1.8×. The budget covers that scratch grid only; the per-column compaction copy is transient and the retained output is outside it. If one history's capacity exceeds the budget, the driver still runs it alone, which it must do to make progress, and logs one warning per request.

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

Arithmetic is fp64 and textually identical to the CPU core, so any divergence is attributable to libm alone. FP32 would change every trajectory at rounding scale and above, and it is not offered; its measured speed-of-light gain is in "Mixed precision (#831)" below.

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

(bounded-continuation-opt-in-prototype)=
## Bounded continuation (automatic production schedule)

Full-history batches tie live scratch to the longest track: every electron of a launch reserves `cap` dense rows, and a launch lasts as long as its longest member. `PerElectronTransportConfig(continuation=TransportContinuationConfig(...))` ([#789](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/789)) instead keeps a fixed population of *active* histories, launches each for at most `rows_per_round` rows, and between launches retires finished histories and refills their lanes from pending primaries. The production process default is `auto`: continuation for qualified exact-CUDA passes, full-history with a logged reason elsewhere. Set `PYRITE_MC_TRANSPORT_SCHEDULE=full-history` to retain the reference schedule.

**Resumable state.** The exact kernel yields only at the top of its step loop, after the launch's row quantum, and never in place of the step limit. There, every loop-carried value is per-history: `(step, flight_id, substep_id)`, the elastic, hard and radiative stream counters, and the three open optical depths. Position, direction, energy, clock and the straggling accumulator are already global. `continuation.py` holds three of each per history (int32, uint64, float64), and a history with a nonzero `step` resumes from them. Because every draw is addressed by (stream, per-history counter), a resumed history executes the same arithmetic, draws and rows as one uninterrupted run: no extra integration steps, draws or replayed rows. Thread `i` owns `slots[i]`, not `e_start + i`, so compaction and refill change only which lane a history occupies.

**Canonical output.** Each round's rows are compacted lane by lane and retained in launch order. At the end, a per-history cursor starting at each history's exclusive row prefix scatters them into the electron-major, step-minor order the full-history driver produces. Active population, row quantum and refill order therefore change performance only. Bounded device staging (`device_staging=True`, [#842](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/842)) is the default; host reassembly was the default from [#825](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/825) until [#866](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/866). Once host reassembly used the parallel run copy, staging was no faster on any measured scene: job 3000 measured staged/host transport ratios of 0.81–1.05 on Si 300 keV, W 300 keV and W 1 MeV at 8,192 histories. The #842 gain had been measured against the earlier per-row host scatter. With staging on, rounds stay on the device and are joined there column by column ([#866](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/866)). A CuPy run-copy kernel, driven by the same `(destination, source, length)` run table as the host copy, places one column of every round; that column's round slices are then freed. Each joined column is downloaded through a double-buffered pinned ring on its own stream while the next column is joined, and the pinned chunks are copied in parallel into the reserved host output. Device memory peaks at the staged rows plus their largest joined column and the run table, so staging is bounded. While that peak fits `staging_budget_bytes`, rounds stay on the device. In A/B job 3018 (8,000 histories, seeds 1–3, arm order alternated, digests identical), staged transport ran at 0.47–0.80× host reassembly on Si 300 keV, W 300 keV and Si 1 MeV. Assembly wall was within ±25%: both paths are limited by first-touch faults on the host output, not PCIe. W 1 MeV keeps 122 M rows (about 15 GB). The payload plus one joined column exceeds the 16 GB card, so staging spills (round 59 of 128) and ties host reassembly. Staging is therefore the default again: it is faster wherever the rows fit and no slower when they spill. When the next round would exceed that limit, the staged rounds spill to the host and the run finishes with host reassembly. The spill then frees the CuPy pool's cached blocks: pageable downloads ran at about half speed while the pool still held most of the card ([#825](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/825), jobs 2994–2995). By default the budget is free device memory after the round scratch is allocated (driver free plus the CuPy pool's cached blocks), less 10% headroom. A run that succeeds with staging off therefore does not fail for lack of device memory with staging on. The `phase="assembly"` diagnostic reports `staged`, the budget and `spilled_round`. Host reassembly packs each round's columns into per-column blocks of at least 64 MiB, then copies runs column by column with a parallel run copy, releasing each column's blocks once copied. Every round holds rows of every active history, so a round-by-round copy faults in the whole output while every round is still resident. Small per-round arrays also stay resident in the C heap after release. W 1 MeV with 8,000 histories (15 GB of rows) peaked at 30–33 GiB with round-by-round copies and peaks at 25 GiB with column-major copies; transport time did not increase ([#825](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/825), A/B job 3013). Passes that keep segments on the device never spill. Sweep 2885 measured staging at 1.22–1.66× request time against the per-row host scatter, with identical digests. The CUDA regressions require bitwise-identical arrays against full-history runs across quanta 1–4096, populations 5–all, a permuted refill order, layered/side-exit geometry, frozen and midpoint schemas, coupled hard-photon events, device-resident output and step-limit errors.

**Memory accounting.** Live scratch is `active × rows_per_round` rows. The population is the largest that fits `scratch_budget_bytes` at `rows_per_round × (2 × row_bytes + 24) + 13` bytes per lane, capped by `max_batch` and `active_histories`. `row_bytes` is measured from the enabled schema's columns (125 bytes for midpoint with coupled photons), the second copy covers compaction, and 24 bytes cover the int64 gather indices. If one lane does not fit, the request raises rather than shrinking silently. Ne-sized per-history state (72 bytes) and retained output are outside the budget: bounded live scratch does not bound total trajectory storage, which remains [#611](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/611)'s scope.

**Production coverage boundaries.** The CUDA comparisons cover SR, ELSEPA and Mott elastic scattering on the exact core, layered geometry, coupled radiation, resident rows and an electron-block slice. The step-limit error, end-to-end retry and block-monitor comparisons also pass on GPU (jobs 2670/2672 at revisions `bbf20ec6`/`1850f099`). Mott's comparison explicitly disables its default transport LUT in both schedules; continuation does not support that LUT. Grooves select lockstep, and `auto` logs that fallback. GDF and launched channeling passes also fall back under `auto`; their continuation combinations are not qualified. Forced continuation rejects host cores and the CUDA LUT instead of silently changing the request.

**Supported scope.** The exact CUDA core under continuous losses, `inelastic_model="shell-soft-hard"` (hard-collision secondaries on or off) and Urban or shell soft-loss straggling, each with or without coupled BremsLib radiation ([#797](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/797)). Shell hard events are sampled and written within their step, so nothing is in flight at the yield; the hard stream counter and optical depth are in the record. Straggling draws are keyed by (flight, substep), and its per-history accumulator is reset only when `step == 0`. The shell schema adds the hard-event pair (10 bytes per row) and, with secondaries, the `(n, 3)` direction column (24 bytes); budget accounting uses the allocated columns, so these count automatically. A secondary cascade runs its own continuation passes. The transport LUT ([#800](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/800)) and every host core raise. Opting in adds no physics, precision, sampling or default-output change. Measured costs are recorded in the [equal-precision pilot](../../validation/beam-transport/equal-precision-pilot.md#bounded-continuation-scheduling-2026-10-29).

**Schedule selection.** `PerElectronTransportConfig.schedule` ([#799](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/799)) is `"full-history"`, `"continuation"` or `"auto"`; `None` keeps the earlier rule, continuation exactly when `continuation` is set. Production runs reach it through `PYRITE_MC_TRANSPORT_SCHEDULE`, read once at import into the default config that the runner, sweeps and `pyrite remote` jobs use. In-process Python calls can instead set `KernelTuning.transport` (`TransportKernelTuning(schedule=..., continuation_rows=...)`) ([#776](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/776)); `runner.transport_options.transport_tuning` lowers them onto a copy of that default config through a context variable, so case payloads are unchanged. Each transport pass resolves it in `continuation.resolve_schedule`. An explicit `"continuation"` uses the given or default `TransportContinuationConfig`, and unsupported combinations raise as above. `"auto"` continues on every exact-CUDA pass that is not a GDF beam or a launched pass (cascade generation or channeling handoff), whatever its size. Anywhere else it keeps full-history and logs the reason at INFO. The threshold and the defaults (`rows_per_round=256`, all budget-fitting histories active, host staging at the time; bounded device staging from #842 until #825 restored host staging) come from the #789 sweeps. At 5000 histories, one full-history launch then held Si and continuation's host-staged rounds made it slower. [#825](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/825) lowered the threshold to 1000 (jobs 2982/2983, device staging): a full-history batch is sized by the longest track, so W 1 MeV launched only 256 histories at a time, and continuation was faster in every Si 300 keV and W 1 MeV pair from 1000 to 8000 histories (1.4–5.3× request time, identical output). Above that, every scene gained, and Q256 was the best memory/time point. [#889](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/889) removed the floor: passes under 1000 histories, including benchmark and streamed tail blocks, still launched full-history batches sized by the longest track, about 108 histories per launch at W 1 MeV, at 334 vs 13 ns of kernel time per row (#875 job 3579). The SM count is not an input: populations above the 84-SM resident-thread ceiling still gained from refill, so the scratch budget is the binding limit. The schedule is kernel tuning only. It is not a case field, so case keys and checkpoint identities do not see it, and the runner regression requires identical spectra under both schedules. Job 2675 qualified 14 completed production schedule pairs with identical trajectory arrays and exit tallies; measured transport speedups were 2.03–3.92× at 5000–20000 histories, 5.58× for Si at 100000 and 6.70× for PbWO4 at 100000. On that evidence and the explicit user decision, the process default is now `auto`. The 100000-history W pair exceeded host memory and remains unqualified. This promotion bounds live device scratch only; retained output can still exhaust host RAM (host admission/containment: #807; streaming: #611). See the [production qualification](../../validation/beam-transport/equal-precision-pilot.md#production-continuation-qualification-799) for scope, spread and memory evidence.

## Where the time actually goes

Ne=16000, capacity 2048 so nothing replays, six launches per run. Kernel and device-compaction totals are from `nsys`; wall times are from unprofiled runs because `nsys` inflates host-side time several-fold. The kernel's own median launch duration is the same under both capacities (37.8 vs 37.9 ms for MoSe2), which is the cross-check that the device-side numbers are trustworthy.

| | hopg | MoSe2 |
| --- | --- | --- |
| wall | 0.188 s | 0.927 s |
| transport kernel | 0.050 s (27%) | 0.224 s (24%) |
| device-side compaction | 0.005 s (3%) | 0.072 s (8%) |
| device-to-host payload | 0.061 s (32%) | 0.146 s (16%) |
| unattributed host-side remainder | 0.072 s (38%) | 0.485 s (52%) |

The payload is 82 bytes per recorded segment — 0.46 GB for hopg, 1.15 GB for MoSe2 — moved over pageable memory at a measured 7.6–7.9 GB/s. The remainder is scratch allocation, mask construction, and NumPy assembly of the output arrays; it has not been attributed further. Round 5 added the NVTX ranges that split it (`pyrite.transport.scratch` / `.compact` / `.join` and siblings); a capture using them is what closes this out.

**The kernel is no longer the bottleneck; the driver around it is.**

## Occupancy and divergence

`ncu` could not be used — GPU performance counters are admin-only on the box (`ERR_NVGPUCTRPERM`) and there is no passwordless sudo. Both quantities were obtained without counters instead.

Occupancy comes from the CUDA occupancy API against the compiled module. The kernel uses **92 registers per thread** and 40 B of local memory, so it is register-limited to 640 of the 1536 threads an SM can hold — **41.7% theoretical occupancy**, flat for `nthreads` ≤ 128 and falling to 33.3% above it. Wall time tracks that exactly (MoSe2, medians: 1.08 s at 64, 1.12 s at 128, 1.27 s at 256, 1.64 s at 512), so the default of 128 looked optimal. #875 later re-timed it on the FP64 and mixed kernels and moved the default to 32 (see below).

Divergence is computed exactly rather than sampled. Thread `i` owns electron `e_start + i`, so a warp covers a contiguous block of 32 electrons and costs its longest-lived member; the wasted fraction is `1 - sum(n) / (32 * max(n))` over each block, where `n` is the per-electron segment count the run already reports.

| material | mean segments | p99 | max | warp-wasted | ceiling if removed |
| --- | --- | --- | --- | --- | --- |
| hopg | 353 | 406 | 448 | 11.9% | 1.14x |
| MoSe2 | 879 | 1220 | 1274 | 27.4% | 1.38x |

**This closes the persistent-thread work queue as an option.** Perfect load balancing would recover 27.4% of a kernel that is 24% of MoSe2's wall time — about 7% end to end, and half that for hopg — in exchange for a work-queue scheme that would break the index-addressed output slots the reproducibility guarantees rest on. Electron lifetimes are simply not spread widely enough: transport at fixed energy into a stopping slab has a fairly tight range distribution, so 32 neighbouring electrons finish at similar times.

The ranking that followed was: keep segments on the device, which attacks the D2H, most of the host remainder, and the spectrum phase's re-upload at once. That is done — see the handoff section. Pinned host memory was *not* the cheap partial it looked like, and is now moot.

## Launch shape and compile options retune (#875)

Harness `checks/launch_retune_875.py`, RTX 5080 (CuPy 14.2.0). Arms alternate in rotating order per seed after an untimed warmup; wall is warm `simulate_trajectories`, kernel is the sum of CUDA-event times of the transport launches; every returned array is hashed against the same-precision `kernel_threads=128` arm. Schedules are explicit `full-history` and `continuation`. Jobs 3036 (attributes), 3040 (sweep, seeds 1-3), 3580 (continuation repeat, seeds 11-17), 3579 (caps, seeds 1-3).

**Resources** (cupyx.jit kernel; blocks per SM at 32/64/128 threads): FP64 186 registers, 40 B local, 8/4/2. Mixed 140 registers, 32 B local, 12/6/3. `__ldg` leaves mixed at 140; `--maxrregcount=168` gives 138; r128 gives 114 registers, 16/8/4.

**`kernel_threads`**, median speedup of 128 over the arm (t128 time / arm time), kernel time, seeds 1-3 (continuation: 10 seeds):

| scene | schedule | fp64 t64 | fp64 t32 | mixed t64 | mixed t32 |
| --- | --- | --- | --- | --- | --- |
| Si 300 keV | full-history | 1.23 | 1.29 | 1.14 | 1.19 |
| W 300 keV | full-history | 1.18 | 1.22 | 1.12 | 1.15 |
| Si 1 MeV | full-history | 1.26 | 1.34 | 1.22 | 1.31 |
| W 1 MeV | full-history | 1.24 | 1.31 | 1.16 | 1.22 |
| Si 300 keV | continuation | 1.17 | 1.26 | 1.14 | 1.24 |
| W 300 keV | continuation | 1.01-1.15 | 1.16-1.24 | 0.90-0.96 | 1.14-1.23 |
| Si 1 MeV | continuation | 1.09-1.21 | 1.24-1.26 | 1.11-1.27 | 1.23-1.24 |
| W 1 MeV | continuation | 1.17-1.20 | 1.22-1.23 | 1.13-1.17 | 1.20-1.21 |

Warm wall time follows: full-history t32 is 1.09-1.23x faster than t128 on Si and W for both precisions. Under continuation the runs are 0.4-4 s and host-bound, so wall medians are 1.0-1.2x with wide seed-to-seed spread (some single seeds below 1); two 3-seed continuation wall medians at t32 were below 1 (Si 300 keV mixed 0.94, W 300 keV FP64 0.89) and both reverse in the 7-seed repeat (1.07, 1.07), where no t32 wall median is below 1. Every launch shape was bitwise identical to t128 at each precision in every scene, schedule and seed. The default `kernel_threads` is now 32.

**Register cap and `__ldg` on the mixed kernel** at t32: kernel time is within about 1% of plain mixed t32 for `ro`, r168, `ro`+r168 and r128 on all full-history scenes (1.00x), all bitwise identical to mixed t128 (unlike the FP64 kernel's r128 in #796). r128 raises occupancy to 16 blocks per SM without a measurable gain. Negative result: no cap or read-only build is adopted. The LUT core shares the launch default; it was not measured.

## ELSEPA guide-table panel search (#826)

ELSEPA polar sampling inverts the angular CDF on both bracketing energy rows of every elastic event. Each inversion used to bisect the whole 606-node row (about 10 dependent loads). [#826](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/826) packs a guide with the tables (`scattering.elsepa_guide_table`): `M = 1024` equal-probability cells per row, and `guide[j]` is the last panel whose CDF is at most `j/M`. A draw in cell `floor(xi*M)` bisects only the window `[guide[j], guide[j+1]]`, which averages 1.6 panels. `M` is a power of two, so `xi*M` and every cell bound are exact. The predicate `cdf[k] <= xi` is monotone in `k` (packing rejects a decreasing CDF), so the window bisection returns the same panel as the full bisection. That includes zero-mass panels, ties and draws exactly on a CDF node. The quadratic-root inversion is unchanged, so every sampled `mu` is bitwise unchanged. The device twin is `_jit_device._elsepa_quantile_mu`. The guide costs `4(M+1)` bytes per energy row, about 4 KiB, next to 9.7 KiB of `cdf` and `pdf`.

**Exactness.** The CPU oracle is the pre-#826 bisection, kept in `tests/montecarlo/test_elsepa_elastic.py`. It agrees bitwise with the guided search on every row of all 110 released electron and positron tables (6190 rows). The draws are a dyadic grid that hits every cell bound, uniforms, every CDF node value with both float neighbours, and the 0/1 edges. Synthetic rows with zero-mass panels, and out-of-range and NaN draws, also agree. The CUDA test `test_cuda_guided_inversion_is_bitwise_the_device_bisection` repeats the comparison on the device against the old device bisection for the same table set. On RTX 5080 (CuPy 14.2.0, driver 610.47, revision `3b6232cd`), whole CUDA trajectory results were compared old checkout (`8b508b7b`) against new. Every result array matched by SHA-256 for Si 300 keV/100 µm, W 300 keV/10 µm, Si 30 keV/2 µm, W 100 keV/5 µm with Urban straggling, and Au 1 MeV/50 µm with Kawrakow deflection. Each case ran 4000 histories at seeds 1 and 2 (SLURM 2820/2829). They also matched at 20000 histories for seeds 1–3 on both anchors (2830). The `-m hardware` regression files (ELSEPA, per-electron, continuation, positron and photon-production CUDA) gave 71 passed and 4 skipped (slow).

**Cost: no resolvable end-to-end gain.** The #821 anchors were run with matched physics and no Geant4 scoring: ELSEPA, Kawrakow deflection, SBETHE, BremsLib soft/hard at 1 keV, a 10 keV cutoff, `auto` schedule and 20000 histories. Each run is one warm `simulate_trajectories` call with CUDA synchronized at both ends. Paired fresh processes in alternating order gave 18 calls per arm (seeds 1–3, three rounds, two processes):

| scene | rows/run | base (s) | guide M=1024 (s) | base/guide |
| --- | --- | --- | --- | --- |
| Si 300 keV/100 µm | 20.7 M | 3.28 [1.92, 3.81] | 2.93 [2.32, 3.67] | 1.12 |
| W 300 keV/10 µm | 39.5 M | 5.30 [4.12, 8.31] | 5.19 [3.98, 6.86] | 1.02 |

Medians [min, max]. The same-process interleaved sweep (6 calls per `M`) has no ordering: Si gave 3.10/3.31/2.92/3.39/2.93 s for `M` = 1 (whole-row window)/256/512/1024/2048, and W gave 5.35/5.62/5.65/5.27/5.25 s. The ±20 % call-to-call spread is larger than any difference, so the 1.12 Si ratio is not a resolved speedup. The transport kernel uses 208 registers and 40 B local memory with the guide, against 204 and 40 B before. Both allocate 6656 registers per warp, so theoretical occupancy is unchanged.

An isolated device microbenchmark (SLURM 2836; 2^22 random draws over the 61 production rows per element, each draw doing two inversions and the `ln E` mix, CUDA events, 15 interleaved repeats) shows how little there is to win. The full bisection takes 0.95 ms (Si, W) and 0.99 ms (Au), about 0.23 ns per draw. The guide takes 0.88/0.89/0.91 ms at `M=1024` and 0.86–0.90 ms at `M=4096`, which is 7–9 %. The same code with a whole-row window (`M=1`) takes 1.05 ms: the guide loads cost about 10 % when they do not narrow the search. At about 0.2 ns per draw, a 20.7 M-event Si run spends about 5 ms of throughput time on polar inversion out of 3 s. The bisection was not a material share of the kernel. `M=1024` is kept as the knee of the microbenchmark: `M=4096` gains about 2 % more for four times the guide memory. Whether to land an exact but unmeasurable change is left to #825.

(kernel-specialization)=
## Kernel specialization

The exact kernel used to take its feature switches as runtime arguments, so one compiled kernel carried every path and its register allocation covered their union: 204 registers per thread and 16.7% theoretical occupancy at 128 threads on the RTX 5080 ([#796](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/796)). [#827](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/827) compiles it once per physics configuration instead.

**Switches.** Nine runtime values gate whole code paths: `elastic_model_code` (SR, Mott, ELSEPA rate and angle paths), `energy_model_code` (frozen versus midpoint rows, including the five midpoint row columns), `finite_footprint` (prism side faces), `sbethe_on`, `straggle_on` (Urban or, with shell mode, the two-moment soft sampler), `inelastic_on` (hard optical depth, channel and transfer sampling, hard row columns), `sec_on` (secondary directions), `rad_on` (soft BremsLib stopping, hard photons, photon row columns) and `cont_on` (resume and yield). All but `sec_on` are the fields of `TransportKernelSwitches`; `sec_on` stays runtime (see below). Values that do not select a path stay runtime arguments: `n_layers` (the one-layer shortcut is a cheap compare), `max_dE_frac > 0`, table contents such as `mott_has_table`, and the exact-versus-table radiative moment test inside `_rad_transport_moment`, which removing locally saves only 6 registers on the benchmark variant and would have meant editing a shared device helper.

**Mechanism.** Two routes were compared. A `cupy.RawModule` over generated source with `#if`/`constexpr` guards would need guard markers in, or text surgery on, the transpiled CUDA; `cupyx.jit.rawkernel` accepts no compile options, and keeping guards aligned with the Python body would be a second source of truth. The chosen route uses the transpiler itself: `cupyx.jit` resolves module globals as compile-time constants and folds an `if` whose test is constant, emitting only the taken branch. `_jit_kernel._transport_kernel` therefore reads the switches as module globals rather than parameters (the parameters are renamed `*_arg`). `_jit_kernel_variants.transport_kernel_variant` builds a function object over the same code object with a copy of the module globals in which each switch is an `np.int32` constant, and wraps it in `jit.rawkernel`; `functools.cache` keys it by the normalized tuple, so each configuration is transpiled once per process and compiled lazily on its first launch. For the reference, `_jit_kernel` binds each switch global to a `cupyx.jit` expression naming its `*_arg` parameter (`_cupy_jit.kernel_parameter`, which uses CuPy's private `_internal_types.Data`), and `monolithic_transport_kernel` wraps the template as is. Its transpiled source is identical to the pre-#827 kernel's apart from the parameter names; the launcher signature, argument tuple and placeholder arrays are unchanged, and `_jit_launch._SPECIALIZE_TRANSPORT_KERNEL = False` selects the monolithic kernel for bisection. The LUT kernel is not specialized: it shares no code object with the exact kernel and has fewer switches.

**Exactness.** A variant executes the same source statements in the same order as the monolithic kernel for its configuration; only branches whose test is constant false disappear. `tests/montecarlo/test_transport_kernel_variants_cuda.py` compares every returned array bitwise between the two kernels for SR (layered prism, frozen and midpoint, Urban straggling), coupled radiation (with and without straggling), shell soft/hard (plain, secondaries, straggling, all features), ELSEPA (frozen, midpoint, with SBETHE and coupled BremsLib), and Mott, each under the full-history and continuation schedules.

`sec_on` is the one switch that stays a runtime argument in every variant. Specialized, it failed that comparison on the GPU: the last bit of `hard_secondary_v_hat` changed. With the secondary branch unconditional the compiler shares the product `2π·u_phi` between the primary and secondary azimuths, and the secondary azimuth `2π·u_phi + π` is then a separate multiply and add rather than the `fma.rn.f64` the monolithic kernel emits (confirmed in the PTX of both). NVRTC contracts multiply-adds by default, and whether it does depends on the surrounding code, so a constant switch can change a rounding without changing the source arithmetic. Keeping `sec_on` runtime restores the monolithic code shape there, and costs at most 2 registers. The bitwise regressions are therefore the guarantee, not the construction. A variant outside the tested set could hit the same effect, so new switch combinations that production uses belong in that test.

**Registers and occupancy.** RTX 5080 (driver 610.47, CuPy 14.2.0), from the CUDA function attributes and `cuOccupancyMaxActiveBlocksPerMultiprocessor` (`checks/transport_kernel_variants.py attributes`, job 2838 at `3e2def41d`). Every kernel has 40 B of local memory and no spills. All variants listed use ELSEPA, midpoint and SBETHE. NVRTC compile times are offline (laptop CPU, NVRTC 13.4, `sm_120`).

| kernel | registers | occupancy at 32 / 64 / 128 / 256 threads | NVRTC compile |
| --- | --- | --- | --- |
| monolithic (every configuration) | 204 | 16.7 / 16.7 / 16.7 / 16.7% | 8.4 s |
| #821 benchmark: coupled BremsLib, 10 mm prism | 190 | 16.7 / 16.7 / 16.7 / 16.7% | 6.2 s |
| #821 benchmark + Urban straggling | 186 | 16.7 / 16.7 / 16.7 / 16.7% | 5.4 s |
| shell soft/hard | 134 | 25.0 / 25.0 / 25.0 / 16.7% | 2.6 s |
| shell soft/hard + straggling | 136 | 25.0 / 25.0 / 25.0 / 16.7% | 2.5 s |
| shell soft/hard + secondaries | 134 | 25.0 / 25.0 / 25.0 / 16.7% | 2.6 s |
| all features, continuation | 214 | 16.7 / 16.7 / 16.7 / 16.7% | 6.8 s |

The register file is split across four SM sub-partitions of 16384 registers, so resident warps step at 136 (three warps per sub-partition) and 168 registers rather than following the 65536-register total. Coupled radiation is the largest contributor: without it the shell variants need 134–136 registers and gain a 50% occupancy step. The #821 benchmark variant saves 14 registers but stays above 168, so its theoretical occupancy does not change. The all-features variant uses *more* registers than the monolithic kernel. Nothing guarantees a smaller allocation once the switches are constants. (At 512 threads every kernel above 128 registers exceeds the per-block register limit and cannot launch; `TransportKernelConfig` defaults to 128.)

**Paired warm timing.** The #821 benchmark physics (ELSEPA single scattering with Kawrakow deflection, midpoint SBETHE stopping without straggling, coupled BremsLib at 1 keV with table moments, 10 keV cutoff, 10 × 10 mm slab), 10000 histories per request, seeds 1–3. After an untimed warm-up of both kernels, the two arms run back to back for each seed, in alternating order. The time is `simulate_trajectories` wall time with the device synchronized. Speedup is the median of the per-seed ratios (job 2839 at `3e2def41d`).

| anchor | schedule | monolithic | specialized | speedup |
| --- | --- | --- | --- | --- |
| Si 300 keV / 100 µm | full-history | 2.229 s [2.210, 2.504] | 1.815 s [1.712, 1.906] | 1.29× [1.17, 1.38] |
| Si 300 keV / 100 µm | continuation | 1.367 s [1.250, 1.583] | 1.223 s [1.138, 1.244] | 1.20× [1.01, 1.30] |
| W 300 keV / 10 µm | full-history | 8.840 s [8.804, 9.477] | 7.049 s [7.000, 7.057] | 1.26× [1.25, 1.34] |
| W 300 keV / 10 µm | continuation | 2.610 s [2.420, 2.785] | 2.425 s [2.405, 2.483] | 1.09× [1.00, 1.12] |

Medians with [min, max] over the three seeds. The gain does not come from occupancy, which is unchanged for this variant, but from executing fewer instructions: no switch tests, no dead branches and a smaller instruction footprint. Continuation gains less because its rounds also pay host-side compaction and refill, which specialization does not touch. In all twelve (anchor, schedule, seed) pairs the two arms returned identical arrays (sha256 over every returned array). The digests also equal those of an unmodified `main` checkout at `8b508b7b` run with the same script (jobs 2831 and 2837). The specialized default therefore reproduces the pre-#827 output.

**Compile and cache cost.** Transpiling a variant takes 0.12–0.15 s of Python per process. In a process with an empty `CUPY_CACHE_DIR`, a variant's first 2000-history request takes 1.6–3.5 s against about 0.1–0.2 s warm, so the first-launch cost is the NVRTC compile. With the cache populated, a new process's first request is within 0.1 s of warm. A run compiles each configuration it uses once per cache directory. A sweep usually holds the physics fixed, so it normally compiles one variant (two with a secondary cascade of different switches). 384 distinct switch tuples exist; all of them transpile offline, and none reads a specialized `*_arg` parameter.

**Checks.** The CUDA comparison file (28 tests) and the surrounding CUDA transport tests (continuation, per-electron, ELSEPA, shell soft/hard, secondaries, straggling, hard radiative, SBETHE, kernel tuning, radiative moments: 155 passed, 8 slow skipped) pass on the RTX 5080 with specialization on (job 2838).

(register-cap-readonly)=
## Register cap and read-only loads (#796)

[#796](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/796) asked whether the exact kernel is held back by its registers or by its table loads. At this revision the full-history #821 benchmark variant uses 186 registers per thread (16.7% theoretical occupancy at every block size). That is the `bench821` configuration of `checks/transport_kernel_variants.py`; the continuation variant was not attributed separately. It also reads its tables through ordinary `LDG`, because `cupyx.jit` passes arrays as `CArray` structs and the compiler cannot prove that they do not alias the output arrays. Neither change helped. No build is adopted, and production still launches the `cupyx.jit` variant unchanged.

**Mechanism.** `cupyx.jit.rawkernel` takes no compile options. `_jit_rawmodule.RawModuleKernel` reuses the variant's generated C++ source, rewrites it if a build asks for that, and compiles it through `cupy._core.core.compile_with_cache` with the JIT's options plus the build's. That is the JIT's own compile entry point. `cupy.RawModule` itself cannot compile this source: it leaves out the CuPy kernel header that defines `CUDART_INF`. A `KernelBuild` has three fields:

- `maxrregcount` sets NVRTC `--maxrregcount`;
- `launch_bounds` emits `__launch_bounds__(threads, blocks)`;
- `readonly` lists array parameters (`TRANSPORT_READONLY`: tables, stream keys and per-history inputs).

Read-only parameters are transpiled as `CArray<const T, ...>`. `cupyx.jit` compares array types by their C++ spelling, so device helpers that receive them are instantiated separately, and a write to one of them fails to compile. A prepended partial specialization of `CArray<const T, ...>` with the same layout reads each element through `__ldg`. Launch geometry, argument conversion and stream are the JIT's. With the default build the source and options equal the JIT's, so the CuPy cache key is the same and the JIT's cubin is reloaded; job 2856 compiled no new cubin. The path reads private CuPy state (`_JitRawKernel._func`, `_mode`, `_cached_codes`, `cupyx.jit._compile.transpile`) and is used only by `checks/transport_kernel_builds.py` and `tests/montecarlo/test_transport_kernel_builds_cuda.py`.

**Resources.** RTX 5080, driver 610.47, CuPy 14.2.0, NVRTC 13.4. The jobs ran revisions `0d7a7a1e` (attributes, tests) and `52d6c4fc` (timing). These are pre-rebase SHAs of `a7fe08b0` and `3c36cbe2`, with the same transport sources, and the two differ only in the check script.

- Registers, local memory and occupancy come from the function attributes and the occupancy API (job 2856; `r96` from job 2855).
- Spill bytes come from `--ptxas-options=-v` (job 2861).
- SASS counts come from `cuobjdump -sass` of the cached cubins. The toolkit's CUDA 13.2 `cuobjdump` rejects NVRTC 13.4 cubins, so the 13.4 `nvidia-cuda-cuobjdump` wheel was used. CuPy's cache entry also ends one byte short of the ELF program headers, so the check pads it before disassembling.

| build | registers | stack / spill store / spill load (B) | occupancy at 128 threads | SASS LDG (constant) | LDL / STL |
| --- | --- | --- | --- | --- | --- |
| `jit` (production) | 186 | 40 / 0 / 0 | 16.7% | 477 (7) | 3 / 2 |
| `ro` (`__ldg` reads) | 186 | 40 / 0 / 0 | 16.7% | 478 (460) | 3 / 2 |
| `r168` | 168 | 56 / 20 / 36 | 25.0% | 477 (7) | 9 / 6 |
| `ro+r168` | 168 | 64 / 28 / 44 | 25.0% | 478 (460) | 11 / 8 |
| `r128` | 128 | 160 / 220 / 316 | 33.3% | 477 (7) | 53 / 42 |
| `ro+r128` | 128 | 152 / 198 / 288 | 33.3% | 478 (460) | 50 / 40 |
| `r96` | 96 | 280 / 678 / 804 | 41.7% | — | — |

`__launch_bounds__(128, 3)` gives the same allocation as `r168` (168 registers, 56 B). The read-only build changes 460 of the 478 global loads to `LDG.E.*.CONSTANT` (`ld.global.nc` in the PTX), and it does so without costing a register.

**Paired warm timing.** This uses the #821 benchmark physics on its four anchors (job 2859, `checks/transport_kernel_builds.py timing`):

- histories: 10000 for Si 300 keV/100 µm and W 300 keV/10 µm, 1250 for W 1 MeV/100 µm and 1000 for Si 3 MeV/1 mm (5.4–19.8 M rows);
- seeds 1–3, with the full-history and continuation schedules;
- each build is warmed first, then one full-size untimed call grows the memory pool, then all seven arms run back to back in an order rotated per seed.

An earlier ordering without the full-size warmup (job 2857, cancelled) charged the first arm of each seed for pool growth.

The table gives the speedup over `jit` as the median [min, max] of the per-seed ratios. A value above 1 is faster.

| anchor | schedule | `jit` s | `ro` | `r168` | `ro+r168` | `r128` | `r96` |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Si 300 keV | full-history | 1.767 | 0.98 [0.95, 1.05] | 1.01 [0.99, 1.01] | 0.99 [0.95, 1.00] | 0.97 [0.96, 0.99] | 0.97 [0.93, 1.00] |
| Si 300 keV | continuation | 1.176 | 1.00 [1.00, 1.01] | 1.01 [1.00, 1.06] | 1.01 [1.00, 1.06] | 0.95 [0.94, 1.05] | 0.99 [0.96, 1.02] |
| W 300 keV | full-history | 6.866 | 1.00 [0.99, 1.01] | 1.00 [0.99, 1.00] | 0.99 [0.99, 1.00] | 0.97 [0.96, 0.99] | 0.96 [0.95, 0.97] |
| W 300 keV | continuation | 1.936 | 0.94 [0.91, 1.10] | 0.93 [0.92, 0.95] | 0.94 [0.90, 1.01] | 0.96 [0.90, 1.03] | 0.95 [0.92, 1.08] |
| W 1 MeV | full-history | 9.701 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.01] | 1.00 [1.00, 1.00] | 0.98 [0.98, 0.98] | 0.96 [0.95, 0.96] |
| W 1 MeV | continuation | 2.765 | 0.94 [0.92, 0.99] | 0.99 [0.97, 1.00] | 0.98 [0.97, 0.99] | 0.97 [0.96, 0.98] | 0.95 [0.90, 0.96] |
| Si 3 MeV | full-history | 4.506 | 1.00 [0.99, 1.00] | 1.00 [1.00, 1.00] | 1.00 [0.99, 1.00] | 0.98 [0.98, 0.98] | 0.95 [0.95, 0.96] |
| Si 3 MeV | continuation | 1.432 | 1.00 [0.94, 1.00] | 0.95 [0.94, 1.00] | 0.95 [0.94, 1.00] | 0.98 [0.98, 0.99] | 0.96 [0.91, 0.97] |

The full-history rows, which are the steadier ones, put `ro`, `r168` and `ro+r168` within 1% of `jit`. Every cap at 128 registers or below is 2–5% slower, and `ro+r128` (not shown) tracks `r128`. The continuation rows spread by about ±5%, and none is a resolved gain. Raising theoretical occupancy from 16.7% to 41.7% bought nothing. Neither did turning almost every table load into a non-coherent cached load. The kernel is therefore not limited by resident warps or by its load path at these launch sizes. The FP64 pipe (FP32:FP64 = 64 on this device) and the long dependent chains per step are the remaining suspects. Without counters this cannot be confirmed. Nsight Compute is still refused (`ERR_NVGPUCTRPERM`, job 2855), so no achieved-occupancy or stall claim is made.

**Exactness.** Results by build:

- `ro`, `r168`, `ro+r168`: every returned array was bitwise equal to `jit` in all 24 (anchor, schedule, seed) runs.
- `raw` (default build) and `ro`: equal to the JIT kernel in `test_transport_kernel_builds_cuda.py`, 52 tests: every case of the #827 variant matrix under both schedules (job 2856).
- `r128`, `ro+r128`, `r96`: differ from `jit` in 1, 4 and 4 of the 24 runs, always on W (300 keV both schedules, 1 MeV continuation).

The cause of those differences is FMA contraction in ptxas. The NVRTC PTX has 812 `mul.f64`, 572 `add.f64` and 696 `sub.f64` without a rounding modifier, next to 2523 `fma.rn.f64`. PTX allows the code generator to fuse unrounded multiplies and adds. Under a tight register budget ptxas schedules and fuses differently: the SASS has 3892 `DFMA`/1003 `DMUL` for `jit` and `r168`, 3900/1064 for `r128` and 3914/1111 for `r96`. The rounding of a fused product then changes, and transport amplifies it, as with `sec_on` in #827. The read-only rewrite changes no arithmetic instruction (identical FP64 PTX and SASS counts). A register cap can change results, so any future adoption would have to put the compile options in the case identity and would need its own bitwise or statistical review. The `r168` agreement shows only that this kernel was not affected at that cap.

**Next.** The resident-warp and load-path levers are closed for this kernel on this device. What remains is the arithmetic itself: #788's tabulated radiative moments and #831's precision probe. A counter-backed profile would also help once the host allows one.

## Mixed precision (#831)

[#831](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/831) measured what FP64 costs this kernel on the RTX 5080 (FP32:FP64 = 64). It is a timing experiment only. No precision option, default or model identity changed, and no variant below is validated physics.

**Mechanism.** `checks/fp32_speed_of_light/precision_variants.py` wraps `_jit_launch.transport_kernel_variant`, rewrites the #827 specialized variant's generated source as text, and compiles it through `compile_with_cache`, the same entry point as the #796 builds. CUDA events around each launch give kernel-only time. The #821 harness (`checks/geant4_performance/pyrite_bench.py`) then runs unchanged with `--kernel specialized`, `--rows-per-round 1024` and device staging. There are four variants:

- `exact`: the generated source unchanged, so it is the production kernel.
- `fp32`: the upper bound. Every `double` and floating literal becomes `float`, and read-only tables are passed as float32 copies. The state and row-output arrays stay float64 in memory, with one conversion per load and store, so the host pipeline is unchanged. The physics is wrong by construction.
- `mixed`: the device functions for energy-grid lookup and interpolation (ELSEPA grid, SBETHE stopping, Mott screening, BremsLib moments and photon sampling) and ELSEPA polar-angle inversion run in FP32, and so do their tables. Position, direction, energy, path and clock accumulation, boundary distances and rotation stay FP64.
- `mixed_math`: `mixed`, plus every `log`/`exp`/`sqrt`/`sin`/`cos`/`pow` left in the kernel body evaluated in FP32. These are the inline ELSEPA rate `exp`, the free-path `log`, the azimuth `sincos` and the rotation `sqrt`. Body arithmetic and accumulation stay FP64.

**Run.** Job 2862, at revision `13f4e57b` (the harness is byte-identical to the committed one, and `pyrite_bench.py` is #821's `5c046888`). The four #821 anchors were run at 36000 (Si 300 keV/100 µm), 20000 (W 300 keV/10 µm), 5000 (W 1 MeV/100 µm) and 7000 (Si 3 MeV/1 mm) histories, 37–76 M rows, with seeds 1–3. The variant order was rotated per seed. The 300 keV anchors resolved to the continuation schedule (204 registers for `exact`), the others to full-history (186). Job 2863 repeated the 300 keV anchors at 2000 histories, all full-history. Evidence: `qlmc:~/pyrite-831b/evidence/fp32sol831/{2862,2863}`; summarize with `analyze.py`.

Speedups over `exact` are the median [min, max] of the per-seed ratios. "Kernel" is CUDA-event time. "Warm" is the harness's warm transport time, which includes host staging, compaction and scoring. The second column is the share of warm time spent in the kernel for `exact`.

| anchor | `exact` kernel / warm s (kernel share) | `fp32` kernel | `fp32` warm | `mixed` kernel | `mixed` warm | `mixed_math` kernel | `mixed_math` warm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Si 300 keV | 0.40 / 2.34 (17%) | 2.68 [2.67, 2.68] | 1.15 [1.15, 1.30] | 1.45 [1.45, 1.45] | 1.07 [0.99, 1.08] | 1.72 [1.72, 1.72] | 1.12 [1.05, 1.14] |
| W 300 keV | 0.70 / 3.18 (22%) | 2.70 [2.68, 2.72] | 1.26 [1.14, 1.31] | 1.46 [1.45, 1.46] | 1.06 [1.06, 1.18] | 1.72 [1.71, 1.73] | 1.15 [1.01, 1.20] |
| W 1 MeV | 7.86 / 12.52 (63%) | 3.11 [3.08, 3.12] | 1.99 [1.98, 1.99] | 1.49 [1.49, 1.49] | 1.27 [1.26, 1.30] | 1.81 [1.80, 1.81] | 1.44 [1.41, 1.44] |
| Si 3 MeV | 7.14 / 10.03 (71%) | 2.52 [2.33, 2.57] | 1.92 [1.76, 1.93] | 1.42 [1.27, 1.49] | 1.19 [0.99, 1.30] | 1.70 [1.51, 1.77] | 1.46 [1.28, 1.52] |

Kernel time per retained row matches the kernel column to within 0.01. In job 2863 (full-history, 2000 histories) the 300 keV kernel gains were larger: `fp32` 3.28–3.32×, `mixed` 1.53–1.54× and `mixed_math` 1.89×.

**Resources.** `fp32` uses 106–112 registers (4 blocks of 128 per SM) and 32 B of stack. `mixed` uses 140–164 registers and `mixed_math` 140–162 (3 blocks per SM), against 186–204 (2 blocks) for `exact`. This means the variants also raise occupancy. #796 showed that occupancy alone buys this kernel nothing: a 168-register cap was within 1% and caps of 128 registers or below were slower. So the gain is attributed to the FP32 arithmetic and transcendental pipes, not to residency. That attribution is still an inference, because Nsight Compute is refused on the host.

**Deviation preview.** This compares paired per-primary observables of the #821 harness against `exact` at the same seed, pooled over three seeds. It previews what the #510 gates would face and is not a validation.

- No variant leaves any primary's path length bitwise unchanged, so every trajectory diverges.
- Categorical outcomes are still mostly identical. Transmitted, backscattered and stopped agree for 99.7–99.9% of primaries at 300 keV and for 91–99% on W 1 MeV.
- No mean observable (transmitted, backscattered and stopped fractions, transmitted energy, path, rows, 4π ≥10 keV photon energy, cone photons) resolves a shift beyond |z| = 2.2 at these history counts. The largest relative shifts with |z| > 1.5 are `fp32` Si 3 MeV path +0.23% (z = +2.2), `fp32` W 1 MeV transmitted energy −1.2% (z = −1.7) and `mixed_math` W 300 keV backscatter +0.06% (z = +1.8).
- At these counts the paired standard errors are about 0.1% on path and energy. A real FP32 bias smaller than that is not excluded.

**Decision.** The all-FP32 upper bound is 2.5–3.1× on the kernel, and 1.15–2.0× on warm transport. Both are above the issue's 1.3× threshold, so mixed precision stays open on #825 as a gated implementation issue rather than closing. `mixed_math` keeps every accumulation in FP64 and recovers 1.7–1.8× of the kernel, which is 1.12–1.46× warm. That makes it the first candidate, with most of its gain on the MeV anchors, where the kernel dominates. At 300 keV the host remainder (78–83% of warm time; #807, #611) caps any precision gain near 1.2×. Adoption requires:

- the #510 FP32–FP64 refinement gates at resolved precision;
- ledger claims for every FP32 table and transcendental;
- a model-identity bump, because the results change at rounding scale and above.

## Exact CPU lookup shortcuts (#841)

[#841](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/841) targeted the two most expensive per-step lookups in the CPU exact core (`make_cpu_transport_core`). #803 attributed them at roughly 40–50% of kernel time. Each change had to leave every output bit unchanged. No model identity, packed-buffer layout or `moment_table_sha256` changed.

**Lazy midpoint cutoff.** The unstraggled midpoint row evaluates `cutoff_rate` at `(E + E_cut)/2` only to decide whether the step truncates at `E_cut`. `_jit_cutoff_bound.py` builds a per-layer bound on `|cutoff_rate|` when the core is constructed. The bound is the node maximum of the table the core interpolates (SBETHE log-log stopping or the LUT row, plus the table soft-moment nodes), widened by `CUTOFF_BOUND_MARGIN = 1e-6`. The row then skips the evaluation when `delta = E - E_cut > 0` and `delta / bound > step`. Float rounding is monotone, so the core's own distance `fl(delta / |cutoff_rate|)` would also exceed `step`, and neither the truncation branch nor the tie branch can be taken. The module docstring holds the full argument, including the SBETHE rounding (below `1e-12` relative, against the `1e-6` margin). The bound is refused, so the cutoff is evaluated on every row as before, in these cases:

- the spliced Joy-Luo/Berger-Seltzer stopping and the exact radiative quadrature, which have no node maximum;
- straggled transport;
- a sign failure or excessive LUT dynamic range;
- moment coverage that does not reach the least electron cutoff.

`lazy_cutoff=False` on the factory restores the unconditional evaluation, which the equality test uses as its reference. On the #803 scenes the evaluated rate never exceeded 0.21–0.31 of the bound (sweep jobs 2891, 2906).

**Moment lookup: the guide table is a negative result.** A guide table over `log E` was implemented first. It fixed the starting node, then a walk of mean 0.25 and at most 2 nodes reached the same bracket as the bisection. It matched the bisection bitwise on 400k-point sweeps of every layer, plus nodes, ±1 ulp and midpoints (jobs 2891, 2906). It was not faster. On 243–310-node layers the bisection takes about 8 iterations from cached data, and the guide's extra loads cost as much as they saved: the guided core ran at 0.96–0.99× (per-scene medians) the speed of the same core with the inline bisection (jobs 2907, 2911). It was removed.

**Moment lookup: inlining is the gain.** The measured cost was the call, not the search. `transport_layer_moments` takes thirteen arguments and keeps the exact-quadrature branch in the same function. Calling it out of line, or through a wrapper that keeps both branches beside each other, slowed the core 1.2–2×. The core now branches at each of the three call sites. Table buffers go to `table_moments` (`inline="always"`), which is main's bisection verbatim. Quadrature buffers go to an out-of-line `exact_moments`. `transport_layer_moments` keeps its signature and behaviour for other callers.

**Equality.**

- `tests/montecarlo/test_exact_cpu_lookups.py` compares `table_moments` and `exact_moments` against a frozen copy of main's lookup on dense sweeps.
- The same file checks that each bound dominates every evaluated rate.
- A slow test runs nine transport configurations with and without both shortcuts and compares every result array bitwise. The configurations cover per-electron and lockstep; continuous and shell soft/hard; coupled and uncoupled radiative; single layer and W/Si; midpoint and frozen; and LUT.
- That test failed under a mutation that shrinks the bound 100×, in 8 of the 9 cases.
- In every timing sample below, `trajectory_sha256` equals main's.

**Timing.** The runs used the #803 harness (`baseline_803.py --core per-electron --elastic elsepa --radiative bremslib-soft-hard --lut off --moments table --schedule full-history --histories 2048 --repeats 5`) on qlmc-queen, CPU partition, one thread, seeds 80301–80303. Each sample ran in a fresh process with a fresh Numba cache, and the arm order was reversed on alternate seeds. Job 2915 timed the shipped revision `710a39c1` (`both`) and the same revision with `lazy_cutoff=False` (`inline`) against main `d5bcd957`. Speedup is main's warm median over the arm's, as the median [min, max] over seeds. In all 72 samples, `trajectory_sha256` equals main's.

| scene | main warm s (continuous / shell) | `inline` continuous | `both` continuous | `inline` shell | `both` shell |
| --- | --- | --- | --- | --- | --- |
| Si 300 keV / 100 µm | 2.12 / 2.91 | 1.49 [1.39, 1.51] | 1.63 [1.51, 1.71] | 1.40 [1.36, 1.41] | 1.47 [1.46, 1.50] |
| W 300 keV / 10 µm | 3.57 / 4.64 | 1.52 [1.42, 1.52] | 1.67 [1.66, 1.71] | 1.40 [1.33, 1.40] | 1.51 [1.49, 1.51] |
| SiO2 800 keV | 8.05 / 9.46 | 1.49 [1.41, 1.51] | 1.63 [1.62, 1.66] | 1.40 [1.39, 1.43] | 1.51 [1.50, 1.51] |
| W 2 µm on Si 100 µm, 800 keV | 1.32 / 1.84 | 1.40 [1.27, 1.45] | 1.47 [1.43, 1.62] | 1.34 [1.26, 1.35] | 1.40 [1.38, 1.44] |

Per seed, the lazy cutoff adds 1.08–1.12× continuous and 1.07–1.08× shell on top of the inline lookup (`inline` time over `both` time). The attribution depends on code layout. Job 2911 timed text-edited variants of `ca68aabe` that kept the unused guide code in place. Against main they gave 1.40–1.46× continuous and 1.30–1.34× shell for both changes. The inline lookup alone gave 1.22× continuous and 0.99–1.08× shell, and the lazy cutoff 1.14–1.34×. The shipped revision is faster than those variants, and the gain now comes mostly from the inline lookup.

Evidence: `qlmc:~/pyrite-841-evidence/` (`timing3/` for job 2915, `timing2/` for 2911, `sweep-*.json`, `diag841-2921.log`); summarize with `analyze_841.py`. Job 2921 also checked the lookup on x86: the final `transport_layer_moments` and `table_moments` are bitwise identical to main's compiled lookup on 2.0M queries. On x86, a pure-NumPy copy of main's lookup differs from main's own compiled lookup on 23–46 queries per scene, because the `np.log` ufunc loop and libm `log` can differ by an ulp. The test oracle is therefore compiled with Numba.

## Selecting the core

`transport_core="auto"` is the default, and `resolve_transport_core` decides: the CUDA core when the process has a CUDA device, the run is ungrooved, and `Ne > CUDA_TRANSPORT_MIN_ELECTRONS` (1000); the lockstep CPU core otherwise. The count is the transport's own, `max(Ne, Ne_brem)` — one transport serves both populations. Every explicit `transport_core=` is honored verbatim, so a caller that names a core gets that core or an error, never a substitution; `PYRITE_MC_TRANSPORT_CORE` pins the choice for a whole process, which is how a run reproduces a pre-threshold result or bisects a host/device difference.

**Why 1000.** The measured crossover is `Ne ≈ 500–1000` for both a light and a heavy material (see "Measured throughput"): below it the launch and staging overhead outweighs the kernel — at `Ne=250` MoSe2 the device core is *0.46×* the CPU one — and above it the device pulls away monotonically, 4.1× at `Ne=4000` and 4.9–15.9× at `Ne=16000`. 1000 is the conservative end of that band, so no run that would have been faster on the CPU core is moved off it. The `--ne-line` default of 450 stays on the CPU core; a 20000-electron run does not.

**Where it runs.** A device-transported run gives up the `gpu-pipeline` engine: its premise is CPU transport hidden behind GPU work, and there is neither anything left to hide nor room for a second CUDA context on the card this process is already driving. `run_cases` therefore keeps such a run in the driver process, serially, and every worker process is kept off the device in `_worker_init` — the pin covers both pools and every call site inside a worker. `_worker_init` redirects only `auto` and `cuda`; it used to overwrite the variable unconditionally, which made `PYRITE_MC_TRANSPORT_CORE=per-electron` a no-op in any pooled run. The switch is all-or-nothing across a run's cases, since a mixed run would strand its CPU-core cases in the driver with nothing overlapping them; sweeps hold `Ne` fixed across the grid, so mixed runs are the exception.

Serial is also simply the faster arrangement once transport is on the device: the case pays one kernel instead of `n_seg`-proportional pipe traffic, and the segments then stay resident for the spectrum kernels (`keep_segments_on_device`, set on this branch alone — device arrays cannot be pickled out of a worker). If the device cannot hold the resident payload, `_transport_case` catches the OOM and replays the same seed with the segments downloaded: counter-addressed streams make that replay exact, so the fallback costs bus time, not the result.

**What it changes.** The realization, not the distribution — see `Validation: gpu-transport-core` in the ledger, and "What is and is not bit-for-bit" above. A pinned spectrum taken at `Ne > 1000` on a CUDA box is not reproduced by the lockstep core and must be regenerated or compared statistically.

At sweep scale this is now measured rather than argued. Over `coh_test`/hopg — 72 cases at `Ne=20000`, 144 (case, energy) records — the CUDA core and the CPU `per-electron` core agree to **0.000% on every recorded field**, and the difference against lockstep is *exactly* the difference the CPU per-electron core already produces. The port is not merely statistically indistinguishable from its specification; at production scale it reproduces it. The stream change itself is not small: `line.spec` median 0.43% / max 2.31%, `brem` median 4.01% / max 25.74% at `Ne_brem=150`, `spec_coherent` median 8.20% / max 37.91% — all tracking `1/√N` for their own population. See "Verification: whole-sweep A/B on remote-host" in [compute-performance optimization history](compute-performance-optimization.md).

## Not done

- **A capture that uses the transport NVTX ranges.** Round 5 added them (`pyrite.transport.tables` / `.sample` / `.alloc` / `.core` / `.output`, with `.upload` / `.scratch` / `.launch` / `.capsync` / `.compact` / `.exitcodes` / `.join` nested inside the CUDA driver), replacing the `pyrite.transport.line` / `pyrite.transport.brem` names Round 1 invented and never shipped. Nothing has been profiled with them yet; that run is what attributes the host-side remainder above.
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
pyrite-dev test \
  tests/montecarlo/test_transport_per_electron.py \
  tests/montecarlo/test_transport_core_default.py \
  tests/montecarlo/test_segment_staging.py
```

The kernel's own properties are in `test_transport_per_electron.py` (51 tests: 36 run on the CPU, 15 skip without a device). The selection policy and the routing that follows from it are in `test_transport_core_default.py`, which patches the device probe and needs no GPU at all — what is under test there is the decision, not the kernel.
