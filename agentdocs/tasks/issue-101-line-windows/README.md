# issue-101-line-windows

Local line windows and feature seeds on a nonuniform line axis.
GitHub issue: #101. Branch: `issue-101-line-windows`.

## Landed on this branch

| Commit | Slice |
|---|---|
| `5b275635` | piecewise line-axis window plans (`_line_windows.py`, `validate_backend_coordinates`) |
| `ab3d9c48` | deterministic seed providers (`montecarlo/spectrum/line_seeds.py`), ledger row `line-window-seeding` |
| `cfb0f988` | opt-in `windows` policy block, schema 2, runner wiring, windowed cache keys |
| `0ef0860b` | window-refinement ladder, `reference_grid`, `convergence_job start-windows` |
| `886ef66c` | error-budget doc; `--max-points` ladder budget |

## Campaign record

Window-refinement campaign, 2026-09-16, remote box `qlmc`:

- Job `20260916-191238-87ccd502` (SLURM 1756) — **failed**. hopg 300 keV at 32
  samples per feature asks 1 260 300 bins; an 11.14 GiB device budget cannot
  chunk that, and the `BackendResourceError` killed the campaign after four
  converged cases. Fixed by `--max-points` in `886ef66c`.
- Job `20260916-192355-e6d2086b` (SLURM 1757) — **complete**, all twelve cases,
  ~11 minutes wall. Checkpoint `line_window_convergence_101b.json` on the box
  under the job directory; re-pull with
  `python -m pyrite.energy_grid.convergence_job pull --json-out <name>.json`.
  The four cases job 1756 had finished reproduced bit-for-bit, which is the
  determinism evidence for the ladder across submissions.

Results and their reading are in
`docs/validation/beam-transport/line-spectrum-error-budget.md`; the raw JSON is
deliberately untracked, matching the #109 campaign.

## Open checklist (from the issue)

- [x] Resolve PXR/CBS interference fringes inside windows separately from
      backbone continuum spacing. Answered by measurement, not code: window
      spacing (0.03-0.9 eV) is already separate from the fixed 3 eV backbone,
      and the sidelobe period equals the main-lobe width, so the same eight
      nodes per feature resolve the fringes. A finer fringe-only quantile is not
      warranted -- see the width-tail table in the error-budget doc. Coherent
      route excluded; that is #117.
- [x] CUDA and fallback line routes agree on the windowed axis (float32 and
      FP64). Measured by `checks/line_window_backend_agreement.py` on hopg
      30 keV and wse2 300 keV: CUDA FP64 vs CPU FP64 yield 2.2e-16 / 2.2e-16,
      centroid and FWHM bit-identical; CUDA float32 vs CPU FP64 yield 1.67e-7 /
      1.71e-7, FWHM 3.1e-5 / 8.3e-7, pointwise 1.3e-4 / 9.8e-6.
- [x] Windows reaching the >= 20 keV float32 binade checked against FP64.
      Diamond at 300 keV is the first catalog case with line yield in
      [16384, 32768) eV (bandwidth to 18900 eV, line at 17084 eV). Reused
      `precision_ladder`, not a fork. Yield and centroid stay inside their
      shares; dominant-line FWHM does not (1.0-1.8e-3 against a 1e-3 share) --
      see the error-budget doc. One case, Ne=60, weak in-window line.
- [x] Seed interface tested with a synthetic non-PXR component standing in for
      #103. Already covered:
      `test_a_registered_non_pxr_component_contributes_windows` registers a
      `transition-radiation` provider and asserts its three fringe windows reach
      the plan.
- [x] Window-plan identity: payload changes move case/checkpoint/cache identity,
      and existing automatic payloads without windows stay bit-for-bit. Covered
      by `test_policy_without_windows_keeps_the_historical_payload`,
      `test_window_policy_changes_case_identity` (four distinct policies, four
      distinct `case_content_key`s), the cache-key tests in
      `test_line_grid_policy.py`, and `test_a_stale_cached_plan_is_recomputed`.
      `case_content_key` is the single identity function checkpoint persistence
      uses, so case and checkpoint identity are the same test.
- [ ] Ledger row `line-window-seeding` is `unverified`; fresh-context
      verification and human sign-off pending.

## Findings worth carrying forward

- Float32 lineshape distortion in the 20 keV binade breaches the FWHM share
  while yield and centroid hold. The 8-ulp collapse floor does not see it (zero
  collapsed intervals at every rung), and the deviation stays flat in
  `spacing/ulp`, so no spacing floor would help either -- #109's conclusion
  extends to this binade. A window reaching it needs `PYRITE_FP64=1` if
  dominant-line FWHM is gated. Whether the automatic policy should warn on such
  a plan is undecided.
- Local GPU (RTX 3060 Ti, 4 GiB admitted of 8) cannot evaluate a 326k-point
  diamond window plan in FP64; the lab box handled 400k in float32. Use narrow
  `precision_ladder` windows locally, or `pyrite remote` for full plans.
- Running GPU work in this worktree needs `uv sync --extra nvidia` once (cupy is
  a large download, several minutes) and then `PYRITE_MC_BACKEND=cuda`;
  `PYRITE_FP64=1` selects FP64. Without the extra the backend probe fails closed
  rather than falling back, so CPU work needs an explicit `PYRITE_MC_BACKEND=cpu`.

- The sinc width distribution is bounded below, not heavy tailed: the longest
  single flight floors `t_L` and flight lengths have no long tail, so the
  intrinsic-source quantile width `w(1e-3)` equals the narrowest feature width
  to within 11% (1.00/1.11/1.00/1.00 over hopg 30 and 100 keV, wse2 30 keV, at
  Ne=60 and 240). That is why `samples_per_feature` nodes really do land across
  the narrowest shape-bearing feature even though the spacing is derived from a
  quantile. `kinematic_line_seeds` now reports `narrowest_feature_width_eV` and
  `samples_at_narrowest` so a case that breaks the property is visible.
  Reproduce with the scratch script pattern: build a `CaseLadder`, recompute
  `2*pi*HBARC_EV_ANG/(denominator*t_L)` per segment, and compare its minimum
  against `sinc_feature_spacing`. Run CPU-only with `PYRITE_MC_BACKEND=cpu` --
  the worktree venv has no cupy, and the backend probe otherwise fails closed.

- The cost argument for windows is energy dependent. Against a uniform grid at
  the finest window's own spacing: 10.4x/13.4x saving at hopg 30 keV, 5.1x/5.6x
  at 100 keV, 2.5x/2.9x at 300 keV; wse2 5.6x down to 1.9x. The kinematic seed
  spans the weighted quantile range of `E_res` over the case's segments, and
  multiple scattering broadens that population with energy until windows merge.
  At hopg 100 keV tilt 5 they cover the axis outright (plan max spacing 0.01 eV
  against a 3 eV backbone). Per-reflection rather than per-case windows is the
  obvious lever; not attempted.
- wse2 100 keV tilt 85 accepts no spacing under the Richardson gate because its
  line/background ratio never stabilises, while agreeing with the dense
  reference to 7.1e-4. Same shape-class limitation #109 found under uniform
  refinement.
- Absorption-edge windows follow the Chantler node density and do not refine
  with `samples_per_feature`, so the ladder gate measures kinematic and
  characteristic convergence only; edge error shows up in the reference
  comparison.
