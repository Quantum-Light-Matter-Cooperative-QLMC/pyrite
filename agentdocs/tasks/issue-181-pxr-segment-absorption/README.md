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

## Remaining acceptance

- User runs CUDA-gated tests on hardware:
  `tests/montecarlo/test_coherent_formation_absorption_cuda.py`,
  `tests/montecarlo/test_xray_dispersion_cuda.py`,
  `tests/montecarlo/test_spectrum_cuda_cheap_hoists.py`, plus the CUDA-backend
  run of the CPU formation tests.
- Fresh-context physics validation of `coherent-formation-absorption` and the
  #181 extension of `segment-escape-average`.
- Remote before/after run of `checks/segment_escape_split_ladder.py`.

## Open risks

- Kernel cost: two extra sin/cos per (line, energy) in formation mode, plus
  more rows (segments split at escape pieces) on the coherent route.
- `expand_escape_pieces` advances piece ages with float64 beta while device
  `t_L` uses REAL beta (rounding-level clock mismatch on float32).
