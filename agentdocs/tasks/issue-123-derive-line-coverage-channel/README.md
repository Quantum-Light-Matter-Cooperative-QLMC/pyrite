# Issue #123: derive line coverage measures the PXR/CBS channel

Issue: https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/123
Branch: `issue-123-derive-line-coverage-channel`
Worktree: `/home/alex/dev/wt/pyrite/issue-123-derive-line-coverage-channel`

Base: stacked on `issue-101-line-windows` at `4da0a8ea`. Land after #101, or
rebase onto `main` once #101 merges.

## Problem

`energy_grid/derive.py::_candidate_from_result` computed line
`coverage_energy` from `result["spec"]`, and the runner folded characteristic
emission into `spec` (and `spec_coherent`). A low-energy characteristic spike
(C K ~277 eV for hopg) owned the cumulative integral. #109 measured a 300 eV
line `stop` at 30 keV vs catalog 2600 eV.

## Decision: separate emission arrays (user, 2026-09-17)

Scope widened from the derive-only fix. `spec`, `spec_coherent`,
`spec_characteristic`, and `brem`/`brem_wide` are separate arrays everywhere;
none includes another. Combination happens only at consumers, through
`src/pyrite/_spectral_components.py` (`line_spectrum`, `incident_spectrum`)
or `Result.line_total()` / spatial `"line_total"`/`"coherent_total"`.

- Public API is exclusive (breaking): `Result.spectrum`,
  `Result.coherent_spectrum`, spatial `"line"`/`"coherent"` exclude
  characteristic.
- Legacy data: HDF5 containers carry `emission_components = "separate"`.
  Containers without it, and all pickles, are separated once in
  `_checkpoint_io.load` (subtract co-located `spec_characteristic`). Marker is
  container-level so slim/merge/basket key projections cannot drop it.
- Behavior preserved: every former reader of the total `spec` (plots,
  detectors, tables, metrics, azimuth selection, `E_pk`) now calls
  `line_spectrum(r)`, which includes characteristic by default.
- Analysis toggles (`apply_characteristic`, `_characteristic_view`) drop the
  component instead of subtracting it.

## Brem audit (static)

`brem_wide` never carried characteristic emission. `_brem_grid_for_rows`
takes the max over all row energies (job 1742 spanned 30-300 keV); 140 keV
matches diamond's 140.7 keV override (also carbon), so the 40 keV hopg
override looks stale. Documented in `line-grid-sinc-convergence.md`. Confirm
per-row brem stops from the job 1742 JSON during the remote check.

## Plan

- [x] Derive measures line coverage on PXR/CBS `spec` only; regression test.
- [x] Runner, reline, and `_lines_for_case` stop summing characteristic.
- [x] Checkpoint store attaches, never adds; legacy containers separated once.
- [x] Consumers combine through `_spectral_components`.
- [x] API `Result.line_total()`; spatial `*_total` components.
- [x] Docs: storage schema, results guide, API, physics page, validation
      write-up/ledger wording, repo map.
- [x] Record the decisions in issue #123.
- [x] Remote check (job 1768 / `20260917-105036-f96fc59f`, 1 mm, 5 deg,
      95 deg): hopg 30 keV raw 95% PXR/CBS coverage is 2470.3 eV and rounds
      to a 2600 eV `stop`, exactly the catalog row rather than the old 300 eV.
      Its brem raw/rounded stops are 13.55/14.3 keV, inside the 40 keV hopg
      override. Reviewed every hopg and wse2 catalog energy; see the validation
      write-up. No catalog regeneration performed.
- [ ] Fresh-context physics review of the `characteristic-radiation` ledger
      wording change (combination point moved; equation unchanged).

## Known pre-existing failures (base `4da0a8ea`)

`tests/dev/test_validation_check_records.py` (#101's unmapped
`checks/line_window_backend_agreement.py`); ty `convergence_case.py:466`.
Environment-only: SYCL, live external DB, sphinx not installed.

## Acceptance

Per issue #123. Focused checks:

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/energy-grid/test_derive.py tests/energy-grid/test_bounds.py
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev lint
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev typecheck
```

## Constraints

- No kinematics, resolution-policy, or kernel changes; #101's
  bandwidth/resolution split stands.
- No catalog or golden regeneration.
- Heavy derive runs go through `remote-gpu-jobs`.
- Skills: `scientific-library`, `regression-testing`, `remote-gpu-jobs` for
  the acceptance run; `physics-review` if the channel definition changes a
  ledgered claim.
