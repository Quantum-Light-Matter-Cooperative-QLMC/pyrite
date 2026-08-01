# Analysis Compare loading and quality selection

Branch: `fix/analysis-compare-loading-quality`
TODO scope: direct user report from 2026-08-01.

## Problem

Opening the **Compare** tab in `cxr app analysis` takes roughly one minute and
prints repeated loads of every available material checkpoint. The tab renders
three cross-material plots, and the current notebook calls `cached_analysis`
once per material *per selection mode* (`quality_peak`, `peak`, and
`line_brem_ratio`). Those distinct cache keys permit three cold unpickles of
each large checkpoint before the three persistent artifacts exist.

The third plot, selected by local line-to-bremsstrahlung ratio, also drops
nearly every material except HOPG while reporting that no line cleared quality
`0.5`. This conflicts with the first two populated plots. Current code applies
the same quality floor to all three modes, then adds a finite-ratio gate only
for `line_brem_ratio`; its single `"dropped"` result and message do not
distinguish quality rejection from missing/invalid ratio. Whether the ratio
metric itself is wrong or only its candidate selection/reporting is wrong must
be established against the user's checkpoints before changing semantics.

## Scope and owners

- `notebooks/analysis_app.py`: lazy Compare-tab orchestration and cache-key
  shape; keep the marimo app thin.
- `src/cxr_mc/plots/spectra.py`: reusable cross-material summary, selection,
  gate reasons, and renderer-independent data contract.
- `src/cxr_mc/run.py` / `src/cxr_mc/analyze.py`: persistent analysis-cache
  plumbing only if the shared-summary design exposes an invalidation or
  concurrency defect.
- `tests/test_material_comparison.py`, `tests/test_run.py`, and
  `tests/test_analysis_app.py`: numerical selection, cache behavior, and app
  structure/regression coverage.

Out of scope: changing checkpoint data, rerunning material sweeps, lowering the
documented `0.5` quality threshold without evidence, or redesigning unrelated
analysis tabs.

## Implementation checklist

- [ ] Reproduce with the current checkpoint set; capture per-material load
      counts and elapsed Compare-tab time for cold persistent cache and warm
      cache cases.
- [ ] Inspect quality and local-ratio metrics for every candidate in at least
      one retained material and one incorrectly dropped material. Record
      whether rejection comes from quality, non-finite ratio, or selection
      logic.
- [ ] Factor one reusable per-material analysis summary that computes the
      metrics/candidate information needed by all three plots from one loaded
      checkpoint.
- [ ] Cache that shared summary once per checkpoint identity and every setting
      that affects the computed metrics; derive each selection mode from the
      small summary without reloading checkpoint data.
- [ ] Correct the ratio plot's metric/gating logic if reproduction proves it
      wrong. Preserve the common quality `>= 0.5` eligibility rule unless the
      evidence establishes a different intended contract.
- [ ] Represent and report drop reasons separately (quality floor, unavailable
      beam energy, undefined/non-finite ratio) so the UI never labels a ratio
      failure as a quality failure.
- [ ] Add focused regressions covering one cold load per material across all
      three plots, warm/restart cache hits, cache invalidation, consistent
      quality eligibility, and valid ratio selection across multiple
      materials.
- [ ] Run notebook structural and real no-browser smoke checks plus focused
      tests, lint, and scoped diff review.

## Decisions and open questions

- Treat performance and third-plot correctness as one task: both originate in
  the same Compare-tab per-material analysis pass and should share one summary
  contract.
- Keep persistent cache under `checkpoints/.analysis-cache/`; do not solve the
  issue by enlarging the raw checkpoint `lru_cache` and retaining multiple
  140–225 MB stores in memory.
- Preserve all three plot meanings and the explicit `0.5` quality floor.
- Open: are most ratios non-finite because of stored checkpoint content,
  `_metrics_map`/local-background calculation, or an overly strict candidate
  gate? Reproduction decides the owning fix.
- Open: should a mathematically undefined ratio omit only that selection point
  or render a diagnostic marker? Prefer omission with an exact reason unless
  measured data shows a more useful representation.

## Delegation

Normal `implement-task` slice after reproduction; likely small cross-module
work but performance evidence and numerical selection must stay coupled.
Required skills: `notebook-workflow`, `performance`, `regression-testing`, and
`scientific-library`. Use `run-cxr-mc` for the real app smoke path. No heavy
Monte Carlo or remote sweep is required.

## Acceptance checks

- Cold Compare-tab render loads each unchanged material checkpoint at most
  once while producing all three plots; warm rerender and app restart with
  valid persistent artifacts load none.
- Before/after elapsed time and load counts are recorded using the user's
  representative checkpoint set.
- All three modes start from the same candidates satisfying line quality
  `>= 0.5`; ratio mode applies only its documented additional validity rule.
- A material with qualifying finite-ratio candidates appears in the third
  plot; every omitted material reports the actual exclusion reason.
- Checkpoint changes invalidate the shared summary; stale cached metrics are
  not reused.
- Focused app/material-comparison/cache tests pass.
- `marimo check notebooks/analysis_app.py` and `cxr app analysis --smoke`
  succeed through the project runner.
- Relevant lint/type checks and `git diff --check` pass.
