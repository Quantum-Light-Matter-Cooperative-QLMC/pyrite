# issue-181-pxr-segment-absorption — handoff

Issue: #181. Branch base: issue-176-segment-escape @ ee1adcbb.

## Slices

1. **Incoherent line route** (committed): segment-mean escape weight;
   Validation `segment-escape-average` (extended).
2. **Phased-field routes** (committed): coherent route and flight-grouped
   reduction take the exact per-piece complex formation integral under
   absorption and the in-medium escape leg. Validation
   `coherent-formation-absorption` (new, `unverified`); derivation record
   `docs/validation/radiation-physics/coherent-formation-absorption.md`.
   Marker `LINE_ESCAPE_MODEL = "segment-mean-v2-coherent-formation"`.

## Physics decision (slice 2)

Per piece the field factor is `t_L F`, `F = exp(-tau_c/2) sinh(w)/w`,
`w = i v - q`, `v = a_vac (E - E_vac) - delta omega dL/2` on the VACUUM sinc
centre/width. Writing `v` on the bulk in-medium denominator would count the
medium twice inside a piece and break split invariance. For a flat exit face
the resulting coherent line centre is the Snell-refracted root
(`delta/(-n_z)` per depth); the bulk root of `xray-in-medium-resonance` uses
the unrefracted `k = Re n omega n_hat` (`delta (-n_z)`). The incoherent route
and all amplitudes keep the bulk root, so coherent and incoherent line
centres differ at O(delta) (0.21 eV of 1600 eV at hopg 002 / 100 keV / 119°).
Taken on the lead's recommendation when the user said to continue; flagged in
the ledger notes of `xray-in-medium-resonance` as an open question.

## Verification done (slice 2)

- CPU: `tests/montecarlo/test_coherent_formation_absorption.py` (quadrature,
  Parseval, split invariance under absorption+refraction on three routes,
  coherent = incoherent integrated yield); two-half split residual now
  1.8e-15 of peak; escape-path root test; substep ladder ~30x tighter.
- CUDA: all seven coherent kernels transpile (cupyx.jit) and compile to PTX
  (NVRTC, compute_80) on a GPU-less host. Device execution NOT run.
  `coherent_grouped_jit_kernel.py` split out of `coherent_stream_jit_kernel.py`
  for the 1250-line module budget.
- Full suite green except pre-existing env failures (SYCL/dpctl, mp_api
  external DB, 2 test_docs errors); lint, typecheck, docs build,
  `validation-ledger --check` clean.

## Hardware verification (2026-09-26, lab box RTX 5080)

Committed trees shipped to `~/scratch/pyrite-181` (#181) and
`~/scratch/pyrite-181-base` (#176 tip `ee1adcbb`); SLURM jobs 160-165.

- CUDA: the three CUDA-gated files plus
  `test_coherent_formation_absorption.py` under `PYRITE_MC_BACKEND=cuda
  PYRITE_TEST_BACKEND=cuda`: 79 passed, 1 skipped (child-session guard).
  The first run failed `test_coherent_spectrum_is_invariant_under_segment_splitting`
  (all three cases) by 1.6e-3 of peak in 3 bins at the line centre against
  a hard-coded `eps_multiple=1e4`. That is float32 phase rounding (piece ages
  of ~7e4 A); the device bound now derives from the flight phase
  `k0 L / beta` (~7e4 rad). fp64 host floor unchanged at 1e-9.
- `checks/segment_escape_split_ladder.py` (full: hopg, 100 keV, 2 um, 30 deg,
  20000 e x 3 seeds). The #181 run first crashed: the transparent control
  patched the removed `_per_hkl._segment_escape_distance`; it now zeroes the
  `expand_escape_pieces` escape ends. Control rows are identical before and
  after. `spec_escape_only` ratio k=32 / k=1:

  | metric | before (#176) | after (#181) |
  |---|---|---|
  | mott total | 1.0201 | 1.0097 +- 0.0012 |
  | mott 1754 eV | 1.0236 | 1.0101 +- 0.0012 |
  | elsepa total | 1.0276 | 1.0025 +- 0.0003 |
  | elsepa 1754 eV | 1.0351 | 1.0024 +- 0.0003 |
  | 3509 eV (mott / elsepa) | 0.9486 / 0.9696 | 0.9483 / 0.9689 |

  The 3509 eV row does not move with #181: raw `spec` there is flat (+0.2 %)
  while the transparent control grows +5.7 %, so it is formation-time
  broadening shedding window yield, not escape bias. The residual ~1 % Mott
  drift (~8 se) is accepted by the user. Characteristic and brem flat;
  ELSEPA - Mott C K -0.02 % at every k.

## Remaining acceptance

- Fresh-context physics validation of `coherent-formation-absorption` and the
  #181 extension of `segment-escape-average`, if not already covered by
  `aebedcc7`.

## Open risks

- Kernel cost: two extra sin/cos per (line, energy) in formation mode, plus
  more rows (segments split at escape pieces) on the coherent route.
- `expand_escape_pieces` advances piece ages with float64 beta while device
  `t_L` uses REAL beta (rounding-level clock mismatch on float32). Measured
  on hardware: within the phase-scaled float32 bound (1.6e-3 of peak on a
  40 um track).
