# Reference elastic scattering data

## Problem and scope

The default `mott` transport samples collision distances from the Browning
empirical total elastic cross section while using NIST SRD 64 transport data
only to tune the first angular moment of a screened-Rutherford surrogate.
Browning's stated range ends at 30 keV, but cxr-mc intends roughly 1--300 keV.
The surrogate does not reproduce the full differential cross section,
higher moments, or backscatter tail, and silent ratio clipping/fallbacks can
invalidate the claimed match.

Source: [`docs/electron_transport_physics_recommendations.docx`](../../../../docs/electron_transport_physics_recommendations.docx), Stage 2 elastic work.

This task owns offline, packaged reference elastic data and its runtime model.
It does not own energy integration, stopping power, straggling, or channeling.

## Implementation path and likely owners

Current owners are `src/cxr_mc/data/mott_transport_cross_sections/`,
`materials/_transport_data.py`, `montecarlo/transport.py`, and the mirrored
CUDA device implementation in `transport_jit_kernel.py`. Catalog validation
also checks transport data availability.

- Establish provenance, redistribution terms, versions, units, interpolation,
  and generation/reproduction procedure for NIST SRD 64 total and differential
  elastic cross sections over supported elements and 50 eV--300 keV.
- Replace Browning total collision rates in `mott` mode with tabulated NIST
  totals; retain an explicit analytic fallback only if policy requires it.
- Prefer sampling tabulated DCS CDFs. If package size/runtime prevents that,
  validate a surrogate against at least the first two angular moments and
  backscatter behavior over an energy/Z grid.
- Count/report endpoint clamps, moment-ratio clipping, missing data, and every
  fallback in run metadata. Missing reference data errors by default.

## Checklist

- [ ] A -- Resolve source/version/licensing/redistribution and write a
      reproducible acquisition or generation record. No opaque copied tables.
- [ ] B -- Define versioned offline schemas for total, transport, and DCS/CDF
      data with unit and monotonicity validation.
- [ ] C -- Compare current Browning totals with NIST totals across supported
      elements and 1--300 keV; publish error surfaces and boundary behavior.
- [ ] D -- Integrate NIST total elastic rates into CPU paths and expose selected
      model/fallback/clamp metadata.
- [ ] E -- Prototype direct DCS-CDF sampling versus a compact higher-moment
      surrogate; decide using accuracy, package size, CPU/CUDA cost, and
      backscatter benchmarks.
- [ ] F -- Implement the accepted angular model in lockstep, per-electron, and
      CUDA cores with shared validated tables/schema.
- [ ] G -- Validate first two moments, angular tails/backscatter, interpolation,
      endpoint policy, CPU/GPU parity, and case-level transport/spectra.
- [ ] H -- Update package data manifests, public science/provenance docs,
      validation ledger, and golden data.

## Decisions and open questions

- **Decided:** default `mott` mode may not extrapolate the Browning total above
  its validity range.
- **Decided:** missing data and fallbacks must be explicit and visible; silent
  clipping/fallback is not acceptable.
- **Open/blocking:** NIST redistribution and reproducible data-acquisition
  terms. Do not commit derived tables until provenance/licensing is resolved.
- **Open:** full CDFs versus a validated compact surrogate; neither is
  `one-shot` until size/performance/accuracy evidence closes the choice.
- **Open:** behavior below/above table endpoints and for catalog elements with
  incomplete reference data.
- **Open:** metadata/checkpoint identity impact when elastic model version
  changes.

## Delegation slices and required skills

- A--C require `lead-task`, `repo-orientation`, `monte-carlo`,
  `scientific-library`, and `documentation-maintenance`; not `one-shot` while
  provenance and model form remain open.
- D--F require `lead-task`, `monte-carlo`, `performance`, and
  `remote-gpu-jobs`; model selection is a physics decision, not a mechanical
  table swap.
- G uses `regression-testing`, fresh-context `physics-validation`, and
  `run-cxr-mc`/`remote-gpu-jobs` for GPU evidence.
- H uses `regen-golden` and `documentation-maintenance`.

## Acceptance checks

```bash
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test-suite core
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev test-suite packaging
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev lint
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev typecheck
UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev verify
```

- Every packaged table is versioned, reproducible, source-attributed,
  unit-validated, and legally redistributable.
- Default `mott` total collision rates use reference data throughout the
  supported range; no Browning extrapolation remains above 30 keV.
- Accepted angular sampling matches documented total/transport cross sections,
  first two moments, and backscatter/tail tolerances across the benchmark grid.
- Missing data, clamps, clipping, and fallback use are errors or explicit
  metadata according to the approved policy.
- CPU and CUDA implementations share the physics model and pass documented
  statistical/parity checks; changed outputs receive fresh validation/goldens.
