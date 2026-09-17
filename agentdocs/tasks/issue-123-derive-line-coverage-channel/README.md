# Issue #123: derive line coverage measures the PXR/CBS channel

Issue: https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/123
Branch: `issue-123-derive-line-coverage-channel`
Worktree: `/home/alex/dev/wt/pyrite/issue-123-derive-line-coverage-channel`

Base: stacked on `issue-101-line-windows` at `4da0a8ea`. Land after #101, or
rebase onto `main` once #101 merges.

## Problem

`energy_grid/derive.py::_candidate_from_result` computes line
`coverage_energy` from `result["spec"]`. The runner adds characteristic
emission to `spec` (and `spec_coherent`) before returning
(`montecarlo/runner/__init__.py`, `spec = spec + spec_characteristic`), so a
low-energy characteristic spike (C K ~277 eV for hopg) owns the cumulative
integral. #109 measured a 300 eV line `stop` at 30 keV vs catalog 2600 eV.

`spec_characteristic` is returned separately, so the PXR/CBS channel is
`spec - spec_characteristic` with no runner change.

`brem_wide` comes only from `_brem_wide_from_segments`; it carries no
characteristic term. The 140 keV brem stop therefore has another cause
(coverage of the wide brem grid, ceiling/override interaction, or truncation
handling) and needs a separate explanation.

## Owners

- `src/pyrite/energy_grid/derive.py`: channel selection in
  `_candidate_from_result` (`coverage_energy_eV`, `total_intensity`); docstring
  stating which channel the bandwidth policy owns.
- `src/pyrite/energy_grid/bounds.py`: `coverage_energy` only if the brem audit
  needs it.
- Tests: `tests/energy-grid/test_derive.py`, `tests/energy-grid/test_bounds.py`.
- Docs: bandwidth-policy wording wherever derive's coverage channel is described
  (`docs/research/beam-transport/energy-grid-recommendations.md`, validation
  pages, `docs/repo-design/` energy-grid pages); grep before editing.

## Plan

- [ ] Subtract `spec_characteristic` from `spec` for line coverage and total
      intensity; tolerate results without the key (older checkpoints) by
      treating it as zero. Document the channel in `derive.py`.
- [ ] Decide whether the derived grid must still span characteristic lines.
      Default: bandwidth policy does not measure them; spanning and window
      semantics stay with #88/#101 seeds. Record the decision here and in the
      issue.
- [ ] Audit the incoherent path: reproduce the 140 keV hopg brem stop from the
      #109 data or a small local case, explain it, fix or document.
- [ ] Regression test: synthetic result with a strong low-energy characteristic
      spike yields the PXR/CBS coverage energy, not the spike.
- [ ] Remote check (via `pyrite remote`, never local): hopg 30 keV line `stop`
      within margin of measured 95% PXR/CBS coverage (catalog 2600 eV); review
      hopg and wse2 against catalog rows. No catalog regeneration in this task.

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
