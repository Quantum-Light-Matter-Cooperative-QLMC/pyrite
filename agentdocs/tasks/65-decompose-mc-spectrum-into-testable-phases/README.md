# 65 — Decompose `mc_spectrum` into testable phases

Issue: [#65](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/65)
(parent [#64](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/64)).
Branch: `65-decompose-mc-spectrum-into-testable-phases`.

Pure restructuring of `src/pyrite/montecarlo/spectrum/lines.py`. No physics
change, no number change, no public-signature change.

## Measured starting shape

`lines.py` is 2 760 lines. `mc_spectrum` spans `:779–2672` (1 894 lines, 26
parameters), of which `:806–1092` is the docstring. The body decomposes as:

| Span | Role |
|---|---|
| `:1094–1136` | argument validation, crystal info, orientation, `n_hat` |
| `:1138–1262` | segment staging, flight grouping, tabulations, kinematics hoists |
| `:1264–1334` | coherent/grouped phase precompute |
| `:1337–1502` | inter-electron decoherence setup + 3 nested closures, mosaic quadrature |
| `:1504–1897` | `_accumulate` closure (394 lines), per-(reflection, orientation) |
| `:1899–1980` | route choice + per-hkl compatibility stacking prologue and row loop |
| `:1983–2668` | batched `(n_seg, N_g)` path — its own table cache, block loop, coherent stream, reduction |
| `:2669–2672` | component assembly and `/Ne` normalisation |

The issue's three-phase model maps onto this as prepare / accumulate /
finalise, with one correction worth recording: **there is no single "final
reduction"**. Accumulation has two mutually exclusive routes (the per-hkl
compatibility loop and the batched path), each carrying its own reduction, and
the batched route is the larger half at ~685 lines. It therefore has to be
split further to satisfy the "no function over ~400 lines" criterion.

## Approach

`_accumulate` and the three decoherence closures capture ~35 enclosing locals
between them. Rather than rewrite the bodies to reach through a state object —
which would touch nearly every line and put bit-exactness at risk — each lifted
phase opens with a short unpacking prologue that rebinds the names it needs from
the setup object, and the body below it moves **verbatim**. The diff stays
reviewable and the numbers stay pinned by construction.

Two dataclasses carry the state the closures used to capture:

- `SpectrumRequest` — frozen, binds the 26 public parameters. `eq=False`, so it
  keeps identity hashing (see the deviation note below).
- `_SpectrumSetup` — mutable, the output of phase 1: resolved geometry, staged
  device arrays, tabulations, grouping decision, coherent/decoherence
  precompute, and the three accumulation buffers.

## Golden protocol

`mc_spectrum` output is pinned before any edit by a 41-case matrix covering both
routes and every CPU-reachable branch: incoherent, coherent, components,
flight-grouped, mosaic, layered, grooved, finite-footprint, sinc-windowed,
decoherence-active, orientation controls, and degenerate shapes. Each case is
stored as raw float64 bytes plus a SHA-256 digest and re-checked bit-for-bit
after every step. GPU bit-exactness is explicitly **not** a criterion (issue
decision); the CPU goldens are the contract.

The harness lives in the session scratchpad — it is a migration instrument, not
a durable artifact. What survives it is the direct per-phase tests added under
`tests/montecarlo/`.

## Landed shape

| Function | Lines | Role |
|---|---|---|
| `mc_spectrum` | 150 | public signature, docstring, request construction (20 executable) |
| `_prepare_spectrum` | 364 | phase 1 |
| `_needs_per_hkl_route` | 20 | route choice |
| `_accumulate_per_hkl` | 101 | phase 2, compatibility route |
| `_accumulate_reflection` (+ `_accumulate_reflection_coherent`) | 283 + 171 | one (reflection, orientation) row |
| `_accumulate_batched` | 253 | phase 2, batched route |
| `_batched_reflection_tables` / `_batched_tables` / `_batched_block` | 105 / 89 / 103 | batched sub-phases |
| `_batched_coherent_block` / `_batched_incoherent_block` | 112 / 109 | per-block steps 5-7 |
| `_batched_coherent_finalize` | 186 | coherent reduction |
| `_finalize_spectrum` | 15 | phase 3 |
| `_mc_spectrum` | 8 | prepare -> route -> finalise |

Largest function in `lines.py` is now 364 lines, down from 1 894. The file grew
from 2 760 to ~3 330 lines: the phase signatures, carrier dataclasses and
unpacking prologues cost more lines than the block comments removed, which is
the trade the issue asked for.

Carrier types beyond the two the issue named: `_BatchedTables` and
`_BatchedBlock` hand per-case and per-segment-block quantities between the
batched sub-phases, and `_LineBatch` replaces a closure over `nonlocal`
counters -- that state was the specific reason the batched incoherent step
could not previously be its own function.

## Deviations from the issue plan (and why)

- **`SpectrumRequest` is not hashed for `_table_cache`.** The issue anticipated
  keying the cache off the frozen request. The batched path's existing
  `table_key` is an explicit tuple containing `id(E_grid_eV)`, which is not a
  value the dataclass can reproduce; switching to a value hash would change
  cache hit/miss behaviour, and `segments`/`hkl_list` are unhashable anyway. The
  key is left byte-identical and merely sourced from the request. `eq=False`
  gives the dataclass working identity hashing without inviting the change.
- **No caller changes.** Keeping the public signature stable means
  `mc_spectrum_solid_angle`, the `*_jit_kernel` modules, and the four
  `montecarlo/runner/__init__.py` entry points need no edit at all.
- **One optional-state seam.** Precompute that simply went unbound on routes
  that never read it (the coherent and decoherence terms) now needs an explicit
  `None` as a dataclass field. Nothing reads one where the closure would have
  raised `UnboundLocalError`.
- **The docstring moved too.** The issue scoped step 7 to the 287 lines of
  *block comments*, but `mc_spectrum` cannot reach "under 150 lines" while
  carrying a 287-line docstring. Its per-option physics narrative became links
  to the `docs/physics/` pages that already own that material; every
  `Validation:` marker and the whole numpydoc contract stayed.

## Not covered by the goldens

The CUDA streaming and fused-reduction paths (`_LineBatch.flush`, the stream
branch of `_batched_coherent_finalize`, `_batched_coherent_block`'s JIT branch)
are unreachable on the NumPy backend, so no CPU golden touches them. They were
moved verbatim, but `_LineBatch` is a genuine restructuring of `nonlocal`
closure state and its lazy kernel import now happens per flush rather than once.
Worth a GPU smoke run before merge.

`mc_spectrum(..., use_henke=False)` remains broken on `main` and here; see
below.

## Known pre-existing defect (out of scope, not introduced here)

`mc_spectrum(..., use_henke=False)` raises `IndexError: too many indices for
array` from `_interp_gather2d` — `chi_g`/`U_g` return a differently shaped
tabulation on that branch, so the batched `(N_g, N_tab)` stacking degenerates to
1-D. Reproduced on `main` at `2aed7009` before any change in this task. Not
covered by the goldens for that reason; worth its own issue.
