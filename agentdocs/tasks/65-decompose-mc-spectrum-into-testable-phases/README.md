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

## Known pre-existing defect (out of scope, not introduced here)

`mc_spectrum(..., use_henke=False)` raises `IndexError: too many indices for
array` from `_interp_gather2d` — `chi_g`/`U_g` return a differently shaped
tabulation on that branch, so the batched `(N_g, N_tab)` stacking degenerates to
1-D. Reproduced on `main` at `2aed7009` before any change in this task. Not
covered by the goldens for that reason; worth its own issue.
