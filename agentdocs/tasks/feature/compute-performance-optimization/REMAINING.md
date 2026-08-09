# Remaining work — compute performance optimization

Written 2026-08-09 against `main` (`c0114c4`), when Rounds 1-4 were fully
merged and nothing was in-flight. Round 5 has since landed five more commits
on this branch (`82c09a8`..`11d030e`, pushed to origin, not yet merged to
`main`) closing W1, W2, and W3 item 1. This doc is the plan for what is
*left*, which now needs only a CLI/remote-authorization decision (W3 item 2)
and a CUDA box (W4).

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

## W3. The `--cpu` / `--cpu-only` profiler is stranded

**Needs a CLI decision plus one authorized remote job.**

The CPU-profiling phase this task built (explicit `-c/--cpu` and `--cpu-only`,
first-class queue phase, phase-aware dashboard, artifact lifecycle) lives only
on `cxr remote run`, which is now **deprecated and hidden, with removal in
0.3.0**. Supported submission is `cxr run PROFILE -R/--remote`, and that command
has no `--cpu` or `--cpu-only`.

Two open items:

1. **Port or drop before 0.3.0.** Either move both flags onto `cxr run -R` (and
   regenerate `docs/cli-reference.md` + `tests/data/cli_contract.json`), or
   decide the feature retires with the command. Doing nothing deletes the
   feature silently at 0.3.0.
2. **Exercise it once for real.** The last unchecked box on the original
   stepwise checklist: the slow profiler has never been run through an
   authorized remote quick-profile job. Everything about it is verified by unit
   and script tests only. Acceptance is already written in
   [`README.md`](README.md) "Acceptance checks" — `--nsys` alone runs no CPU
   pass, `--cpu` runs primary then CPU and reports both while attached,
   `--cpu-only` starts no GPU phase and produces `.cpu.prof` + `.cpu.txt`, CPU
   failure fails the job while retaining primary artifacts.

Skills: `cli-ui-ux`, `remote-gpu-jobs`, `regression-testing`.

## W4. Deferred performance levers

All measured, all deliberately not taken. Sizes are from Round 4 on `qlmc`
(RTX 5080); the durable list is the round doc's "Still open".

| lever | size | why deferred | needs GPU |
| --- | --- | --- | --- |
| NVTX ranges in `transport.py` | diagnostic only | none — it is just unwritten; the round-1 method doc *claims* `cxr.transport.line`/`.brem` exist and they do not | no to write, yes to use |
| Host-side remainder of the transport driver (scratch alloc, mask construction, output assembly) | 38% of hopg wall, 52% of MoSe2 | unattributed — this is exactly what the NVTX ranges above would resolve, so it is gated on them | yes |
| Compact resident segments to `REAL` at the join | ~halves 509 MB held / 1176 MB peak pool | changes the dtype the function documents | yes |
| `gpu-pipeline` memory sizing | pipeline arm drove `qlmc` to 50.3 GB RSS + 12.9 GB swap on a 45 GB box | **Done 2026-08-09** (taken first, out of sequence): `_usable_cpus()` reads the affinity mask / `SLURM_CPUS_PER_TASK` / cgroup quota instead of `os.cpu_count()`, and sizing now budgets *slots* — worker payloads and in-flight driver payloads alike — so `2*nw + 2` residencies fit the host budget. Confirmation on hardware still owed | to confirm |
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
   have. The one authorized remote job that exercises them is still owed.
4. Is any of W4 wanted now, or does the task close with Round 4 and leave the
   levers in the round doc's "Still open"? The plan is written (round doc,
   "Round 5"); levers 1-4 all need `qlmc`.
