# Remaining work — compute performance optimization

Written 2026-08-09 against `main` (`c0114c4`), when Rounds 1-4 were fully
merged and nothing was in-flight. Round 5 then closed W1, W2, and W3 item 1.

**Completed 2026-08-14.** The authorized `qlmc` run closed the outstanding
primary-only Nsight acceptance check and the remaining W4 attribution. At
production `hopg_hbn` scale the pipeline was not
transport-starved; spectrum dispatch/synchronization dominated instead. The
measured boolean-compaction fix in `montecarlo/spectrum/lines.py` preserves the
selected rows byte-for-byte while reducing blocking D-to-H synchronizations
from 46 to 19 per case and CUDA launches from 471 to 366 per case. The separate
checkpoint shard-write scaling issue is recorded below but remains outside
this task.

## Round 5 status (2026-08-09, on the task branch, committed: `82c09a8`, `7b78d26`,
`cd938b4`, `3cc0fb0`, `11d030e`)

The user's directives were: W1 do it; W2 re-check whether the win still holds;
W3 add the flags back; W4 write the NVTX ranges plus a plan, defer
`numba.prange`; W5 stays deferred.

| item | state |
| --- | --- |
| W1 | **Done.** Four ledger rows; `line-absorption-tabulation` came back a `discrepancy`, see below |
| W2 | **Closed: deleted.** No flip (~1.7% → ~2.7–4.8% of case wall, still low single digits). Option (a) taken 2026-08-09: `_USE_JIT_LINE_PROLOGUE` and `montecarlo/line_prologue_jit_kernel.py` removed; lint/typecheck/`test-suite core` (1168 passed, 57 skipped) all green. Recommendation was documented in this file and acted on under Auto Mode's bias-to-act — reversible via git history if the user wants it back |
| W3 | **Done for item 1** (flags on `cxr run -R`). Item 2 (one real remote job) still needs authorization |
| W4 | **NVTX done**, plan written into the round doc's "Round 5". The pipeline host-memory sizing was then taken out of sequence and **fixed** (it is the one correctness-adjacent lever). The remaining levers need a CUDA box; `numba.prange` deferred by decision |
| W5 | **Deferred**, as directed |

Two things that need a human, neither invented by this round:

- `line-absorption-tabulation` is a **`discrepancy`**. Tabulated `μ` is off by
  `2.72e-01` at hopg's C K-edge where `χ_g` on the same grid is off `5.82e-05`;
  `exp(−μL_esc)` is off `5.23e-01` at `L_esc = 1e4 Å`. Edge-localized, median
  `1.6e-6`. The in-code claim that it was "at least as accurate as the chi/U
  interpolation" was false and has been corrected in the comment; the numerics
  are untouched. Three resolutions are listed in the ledger row.
- Orphan marker outside W1's named scope: `relativistic-ceiling` in
  `src/cxr_mc/sweep.py:61,567`, self-labelled "(placeholder)". Reported, not
  fixed.

Open questions 1-3 below are answered (`ALEX-DESKTOP` is a local desktop, not an
ssh target; do not flip the prologue; port the flags). Question 4's remote-gated
levers are still open.

Companion reading, not repeated here: [`README.md`](README.md) (the full
per-item measurement record) and
[`docs/compute-performance-optimization.md`](../../../../docs/compute-performance-optimization.md)
Rounds 1-4 plus its "Still open" list (the durable engineering backlog).

## Closed, do not reopen

Rounds 1-4 landed. Part B (the MoSe2 `--ne-line=20000` "stall") is answered and
closed: reproduced, attributed to compute-bound transport plus a one-time
pipeline-fill transient, scaling linearly in `n_seg` with no discontinuity and
no material-specific slow path; OOM retry, chunk resizing, checkpoint I/O and
`njit` compile all ruled out by direct telemetry. Decision was
document-as-expected, no default change. Round 4 then made the CUDA transport
core the default above 1000 electrons with `Validation: gpu-transport-core` and
a ledger row, measured at 1.55x case-loop and 12.7 GB -> 0.76 GB host RSS on
`qlmc`. Two of Part B's four follow-ups are done:
`docs/performance-profile-analysis.md` was corrected (`0da9e44`) and the fill
transient is documented there.

## W1. Ledger the four orphan line-path `Validation:` markers

**Blocks W2. No GPU needed. Smallest, highest-leverage item here.**

Verified still true on `main` 2026-08-09: the incoherent line path carries four
`Validation:` markers with **no ledger row**, and no ledger row's `code` column
names any of the five `*_jit_kernel.py` modules.

| marker | site |
| --- | --- |
| `line-amplitude-fusion` | `montecarlo/spectrum.py:102`, `:856` |
| `line-gemv-elementwise` | `montecarlo/spectrum.py:137` |
| `line-hkl-batch` | `montecarlo/spectrum.py:204`, `:1142` |
| `line-absorption-tabulation` | `montecarlo/spectrum.py:747` |

Two of the four carry in-code notes saying ledger + regen + human sign-off are
required, and `line-absorption-tabulation` self-describes as "a physics-METHOD
change, not a reassociation", so it is not automatically the same low-risk class
as the other three.

Work: one ledger row each (source equation or reassociation argument,
assumptions, limiting case, checks), classed against the existing
`coherent-line-hkl-batch` precedent. Fresh context verifies; only a human marks
`signed-off`.

This is the same work as `TODO.md` Active item 3 and should be dispatched there
rather than here — it is listed in this plan only because W2 cannot be decided
before it lands.

## W2. Decide the fate of `_USE_JIT_LINE_PROLOGUE` — CLOSED 2026-08-09, deleted

**Needs: W1, a user decision, and (for one option only) a CUDA box.**

Resolved: option (a) below, taken. `spectrum.py`'s `_USE_JIT_LINE_PROLOGUE`
flag and gated call site, plus `montecarlo/line_prologue_jit_kernel.py`
(347 lines), are deleted. `WM`/`G`/`ES`/`EP`/etc. stay — they still feed the
coherent-stream and eager line paths. No test referenced the flag or module
(confirmed before deletion), so no test changes were needed. Historical
measurements below and in `docs/compute-performance-optimization.md` Rounds
3 and 5 are left as the record; `docs/gpu-transport-rawkernel.md`'s references
to `line_prologue_jit_kernel` as a house-style precedent are also left (past
tense, not misleading about current state).

State: `spectrum.py:39` `_USE_JIT_LINE_PROLOGUE = False`, gating a 347-line
`line_prologue_jit_kernel.py` that nothing else reaches. Zero tests reference
the flag or the module. The numerics were verified comprehensively on `qlmc`
(exact keep-mask agreement over 17.2 M pairs, bit-for-bit `E_r`/`aw`,
significant-bin error ~170x inside the repo's own accepted cross-path bound,
bitwise run-to-run determinism) — the numerics are *not* the blocker. `w` is
not bit-for-bit (max 7.97e-6 relative, 124 ULP) and the mechanism is identified:
the eager gather kernel blocks FMA contraction with explicit
`__fadd_rn`/`__fmul_rn`/`__fsub_rn`, the prologue's `_interp_row`/`_interp_shared`
do not.

What actually blocks a flip is value, not correctness: the win is 4.2-8.9% of
the *line phase*, and the line phase is ~1-3% of case wall time on `qlmc` where
transport is 2.4-3.4x the whole GPU phase. Round 4's transport work has since
moved the bottleneck again.

Three options, pick one:

- **(a) Delete the module and the flag.** Reclaims 347 lines of dead,
  untested, unreachable CUDA code and removes a permanently-off branch from
  `spectrum.py`. Cheapest, local-only, no physics claim. The measurements are
  preserved in `README.md` and the round doc if it is ever wanted back.
- **(b) Flip it.** Requires `Validation: line-prologue-fusion` anchored on
  `line_prologue_jit_kernel.py::run_line_prologue_kernel`, status `filtered`,
  same class as `coherent-line-hkl-batch`; a ledger row carrying the item-2/3
  numbers; golden regen; and a real regression test, since none exists. Also
  wants the eager kernel's rounding discipline (`__f*_rn` intrinsics or
  `-fmad=false`) to remove the `w` divergence — note that still would not make
  the end-to-end spectrum bit-for-bit, because the prologue feeds the reduction
  kernel all `n_seg * N_g` fixed-order pairs against the eager path's compacted
  survivors. Tolerance, not bit-for-bit, is the right frame. Needs a CUDA box to
  revalidate after the rounding change.
- **(c) Leave gated.** Status quo; carries the dead code and the open question
  forward indefinitely.

**Recommendation: (a).** 1-3% of wall time does not justify a fifth physics
claim on a path that already owes four (W1), and the branch has been off since
Round 3 with nothing depending on it.

Gate correction (answered 2026-08-09): the original checklist required the
interleaved A/B on **both** `qlmc` and `ALEX-DESKTOP`. `ALEX-DESKTOP` is not an
ssh target and never will be — it is the user's home desktop (RTX 3060 Ti), so
its arm runs **locally, when working at that machine**, not through
`cxr remote`. It is not absent from the gate, just not remotely reachable; an
agent on `qlmc` cannot close that arm and should not treat its absence as a
blocker.

## W3. The `--cpu` / `--cpu-only` profiler — completed

**Completed 2026-08-14.** The flags were ported to the supported remote run
surface, and the authorized `qlmc` job exercised the profiler. Nsight job
`hopg_hbn-8` / SLURM `1640` ran only the primary scan when invoked with
`--nsys`, satisfying the remaining acceptance check without starting the CPU
phase.

The CPU-profiling phase this task built (explicit `-c/--cpu` and `--cpu-only`,
first-class queue phase, phase-aware dashboard, artifact lifecycle) originally
lived only on the deprecated `cxr remote run`. Resolution:

1. **Port before 0.3.0 — done.** Both flags moved to the supported remote run
   surface and the generated CLI artifacts were updated in the earlier slice.
2. **Exercise the accepted remote path — done.** `--nsys` alone ran only the
   primary scan in job `1640`. The resulting ~318 MB Nsight trace provided the
   W4 attribution below. The combined `--cpu` and `--cpu-only` arms retain their
   existing unit/script evidence; this resumed slice did not rerun them.

Skills: `cli-ui-ux`, `remote-gpu-jobs`, `regression-testing`.

## W4. Deferred performance levers

All measured, all deliberately not taken. Sizes are from Round 4 on `qlmc`
(RTX 5080); the durable list is the round doc's "Still open".

| lever | size | why deferred | needs GPU |
| --- | --- | --- | --- |
| NVTX ranges in `transport.py` | diagnostic only | **Done and exercised.** The ranges separated hidden transport work from the spectrum bottleneck in the `qlmc` production trace | yes to use |
| Host-side remainder of the transport driver (scratch alloc, mask construction, output assembly) | **Closed 2026-08-14.** At `hopg_hbn` scale transport was 17.70 s cumulative across four workers (~4.4 s wall, hidden), and driver transport wait was 1.45 s | not the production bottleneck; no transport change | yes |
| Spectrum boolean-mask compaction | `cxr.lines` was 20.86 ms/case; 46 blocking scalar readbacks and 471 launches per case | **Done 2026-08-14.** Resolve the survivor index once and reuse it across segment arrays: 19 readbacks and 366 launches per case, byte-identical selected rows | yes |
| Compact resident segments to `REAL` at the join | ~halves 509 MB held / 1176 MB peak pool | **Not justified for throughput by the local trace:** `.join` was 2.3% / 0.5% of transport-core wall for hopg / MoSe2. Memory-only motivation remains, with the documented dtype/API cost | yes |
| `gpu-pipeline` memory sizing | pipeline arm drove `qlmc` to 50.3 GB RSS + 12.9 GB swap on a 45 GB box | **Confirmed 2026-08-14.** `--cpus-per-task=8` and `SLURM_CPUS_PER_MATERIAL=8` resolved to four workers even though process affinity exposed all 32 cores. Feed wait was 1.1%; more workers would not improve this workload | confirmed |
| `numba.prange` over the per-electron core | starts from a 0.27-1.05x deficit vs lockstep | needs more than two cores just to break even | no |
| Grooved transport on the CUDA core | grooved runs stay on lockstep | scope | yes |

The pipeline memory item was the only one with a correctness-adjacent edge (a
box can be driven into swap), so it was taken first; the rest are throughput.

## W5. Optional: the missing MoSe2 pipeline arm

Round 4's whole-sweep A/B has a device arm for `promising`/mose2 but no pipeline
counterpart — that run was `SIGTERM`ed at case 319/432 after driving the shared
box into swap and making it unreachable over ssh for ~15 minutes. It did return
the one number that mattered (`gpu_feed_wait_fraction` median 0.569 vs hopg's
0.103, and peak VRAM 9405 MiB against the resident arm's 6873). Completing it
would only refine a comparison already decided. **Recommend: do not rerun**
unless W4's pipeline-sizing item is taken, and then only with an explicit
memory cap. W4's sizing fix landed 2026-08-09, so this arm is now the natural
workload for confirming it — the model predicts 4 workers and 6 in flight on a
`--cpus-per-task=8` allocation, ~15 GB instead of 50.3 GB.

## Dispatch

| item | worker | skills | authority needed |
| --- | --- | --- | --- |
| W1 | `implement-task` under Active item 3 | `physics-validation`, `physics-review` | none beyond task-local commits — **done** |
| W2 (a) | `implement-task-lite` | `monte-carlo`, `regression-testing` | user decision first — **done 2026-08-09** |
| W2 (b) | n/a | n/a | superseded — (a) was taken instead |
| W3 | `implement-task` | `cli-ui-ux`, `remote-gpu-jobs`, `regression-testing` | authorized remote job |
| W4 | `lead-task`, one lever per slice | `performance`, `remote-gpu-jobs` | remote GPU, shared-box care |

## Open questions for the user

1. ~~Is `ALEX-DESKTOP` still a target box?~~ **Answered:** it is the user's home
   desktop, not an ssh target. Its arm is run locally from that machine; `qlmc`
   is the only remote box.
2. ~~`_USE_JIT_LINE_PROLOGUE`: delete, flip, or leave?~~ **Closed 2026-08-09:
   deleted.** Flag and module removed; see W2 above.
3. ~~Do `--cpu`/`--cpu-only` move to `cxr run -R`?~~ **Answered: yes**, and they
   have. The authorized primary-only Nsight check ran on `qlmc`; the two CPU
   arms retain their earlier unit/script coverage.
4. ~~Is any of W4 wanted now?~~ **Closed 2026-08-14:** the authorized `qlmc`
   trace found and removed the material spectrum synchronization bottleneck.
   The other measured levers remain unjustified or out of scope.

## Final `qlmc` evidence (2026-08-14)

Production job `hopg_hbn-7` / SLURM `1639` ran 5,508 cases at `Ne=450` in
184--187 s (~33.4 ms/case). Terminal attribution was spectrum 132.07 s (70.5%),
checkpoint 46.15 s (24.6%), with CPU transport hidden behind spectrum work.
Median GPU utilization was 12% and median CPU utilization 3.7%, but the low
utilization was dispatch/synchronization bound rather than worker starvation:
GPU feed wait was only 1.1%.

Nsight job `1640` attributed 20.86 ms/case to `cxr.lines` and 4.06 ms/case to
`cxr.brem`, against only ~2.4 ms/case of actual GPU kernel work. The trace made
253,368 D-to-H copies across 5,508 cases: 46 tiny copies per case, each paired
with one blocking `cudaStreamSynchronize`. A four-case causal
reproduction matched the production signature exactly: 184 synchronizations,
1,884 launches, and 340 copies.

The source was repeated CuPy boolean gathering over `_SEG_ARRAYS` in
`_clip_segments_to_cutoff`. Resolving `xp.nonzero(keep)[0]` once and reusing
the integer index reduced blocking synchronizations and D-to-H copies from 46
to 19 per case, and launches from 471 to 366. Selected H-to-D payloads were
byte-identical. Across three warmed cases, the hot-line timing changed from
approximately 20.9/20.9/30.9 ms to 17.9/17.6/18.9 ms. Focused path tests:
192 passed, 2 skipped; the final local cutoff/spectrum subset passed 46 tests.

### Follow-up mask reuse after the production rerun

The user's next production run, `hopg_hbn-9` / SLURM `1641`, completed hopg in
175 s and h-BN in 186 s but still showed roughly 10--15% GPU utilization. Do
not compare its wall directly with the earlier `Ne=450` evidence: the current
profile now uses `Ne=300`. Jobs `hopg_hbn-10` / `1642` (baseline) and
`hopg_hbn-11` / `1643` (candidate) are the matched comparison: same 5,508
uncached hopg cases, parameter digest `4135cde714d5`, `Ne=300`, `Ne_brem=150`,
four lockstep transport workers, RTX 5080 CUDA backend, and full Nsight trace.

The remaining trace signature was again repeated mask compaction: the batched
incoherent line path tested a mask and then gathered three arrays with it;
bremsstrahlung gathered four arrays with one mask. Both now resolve the mask to
an integer index once and reuse it. Measured result:

| metric | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| blocking synchronizations / D-to-H copies per case | 19 | 13 | -31.6% |
| CUDA launches per case | 366 | 345 | -5.7% |
| case-loop wall | 188 s | 181 s | -3.7% |
| performance-session elapsed | 190.898 s | 184.274 s | -3.5% |
| spectrum cumulative | 130.706 s | 125.012 s | -4.4% |
| `cxr.lines` mean | 19.202 ms/case | 18.670 ms/case | -2.8% |
| `cxr.brem` mean | 3.277 ms/case | 2.797 ms/case | -14.7% |

The complete candidate `line.pkl` and `brem.pkl` artifacts are byte-identical
to the baseline (matching SHA-256 for each). Local focused cutoff, spectrum,
bremsstrahlung, and coherent-emission checks passed: 59 tests. The 13 remaining
readbacks are no longer repeated gathers over the two masks fixed here; further
reduction needs a broader result-transfer or kernel-interface change.

Checkpoint time is a separate lifecycle issue, not part of this optimization.
The job had 1,836 configurations and called `_save_part` once per config;
final consolidation took only 2.68 s, leaving about 43.5 s in shard writes.
`_manifest_save(..., _material_subset(), ...)` appears to rescan accumulated
results repeatedly, consistent with O(N^2) sweep behavior. Route that diagnosis
to the chunked-checkpoint lifecycle task; do not fix it here.

## Local ALEX-DESKTOP evidence (2026-08-10)

The first W4 gate is now exercised on the user's home desktop: Ryzen 9 5900X,
22.9 GiB RAM, RTX 3060 Ti 8 GiB, NVIDIA driver 610.43.02 / CUDA runtime 13.2,
CuPy 14.1.1, Nsight Systems 2026.4.1. One warmed, uncached case per material used
the `promising_low_ne` grids with `Ne=8000`, `Ne_brem=150`, seed 75001, 30 keV,
1e4 A, polar tilt 45 degrees, azimuth 140 degrees, CUDA transport, resident
segments, and `REAL=float32`. The arms differed only by material.

| material | segments | unprofiled wall, five reps | `.core` | `.capsync` | `.compact` | `.join` | pool high-water |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| hopg | 373,949 | median 64.33 ms (63.09-89.19) | 58.23 ms | 28.51 ms | 12.76 ms | 1.33 ms | 375 MiB |
| MoSe2 | 3,495,865 | median 740.46 ms (711.59-748.68) | 290.75 ms | 240.64 ms | 19.64 ms | 1.46 ms | 1,203 MiB |

Nsight adds overhead, so stage numbers explain the trace and are not compared
to the unprofiled wall as a speed ratio. `.join` is only 2.3% of hopg and 0.5%
of MoSe2 transport-core wall; the gate for dtype-changing `REAL` compaction is
not met on throughput. `.compact` is visible but is not the join-time dtype
conversion proposed by that lever. No OOM occurred. Raw local artifacts are in
`/tmp/pyrite-local-bench-20260810/` and intentionally untracked.
