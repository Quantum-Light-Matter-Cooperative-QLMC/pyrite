# 3D penetration plot: exit paths, regen/electron/FWHM controls, real-time playback

Date: 2026-07-22
Status: draft (pending user review)
Scope: `src/cxr_mc/plots/plotly_trajectories.py`, `notebooks/analysis_app.py`
(penetration tab only). No change to `src/cxr_mc/plots/trajectories.py`'s 2D
matplotlib/Altair views, `plot_trajectory_grid`, or `trajectory_chart`.

## Motivation

The interactive 3D cutaway (`trajectory_volume_figure`) currently truncates
every electron track exactly at the crystal boundary. Backscattered and
transmitted electrons vanish at the entrance/exit face, giving no sense of
where they actually go. The figure is also non-interactive beyond the
`realistic` checkbox and a fixed `Ne=40`: no way to see a different random
draw, control track density, or set the beam spot size directly. Finally,
the cascade currently renders fully-formed; there is no way to watch
electrons propagate through the slab over time.

## 1. Exit paths

**What draws:** for each electron whose terminal segment ends at a slab
face (top: `z ≈ 0`, "backscattered"; bottom: `z ≈ thick`, "transmitted"),
draw a short, dim, dashed straight-line continuation from the exit point
along the exit direction (the terminal segment's own direction — no further
scattering happens in vacuum). Side-exits (realistic mode, finite
footprint, exit through a transverse face) are **not drawn** — out of
scope, rare, and require extra face-boundary classification.

**Classification:** per electron, find the terminal segment (max `t_fs`
among that electron's segments — segments are appended in chronological
order per electron, so the last occurrence is terminal). Classify by its
`end_xyz` z-coordinate against 0 and `thick` within a small tolerance.
Electrons that stopped (energy below cutoff, interior termination) or
side-exited get no exit-path line.

**Style:** dashed, low-opacity, fixed display length (same scale as the
existing detector-direction arrow, `0.28 * span`), a muted single color
(not the turbo energy colorscale) so it reads as "left the crystal,
ballistic" rather than competing with in-crystal tracks. One legend entry
("exit path"), not one per electron.

**Data plumbing:** `_trajectory_data` (trajectories.py) does not currently
return `t_fs`/`v_hat` in a form directly indexable per-electron in original
segment order for this purpose — actually `t_fs`, `elec_id`, `start_xyz`,
`end_xyz` are all already returned in original per-segment order, so the
terminal-segment lookup (groupby elec_id, argmax t_fs) can be done entirely
in `plotly_trajectories.py` from the existing `data` dict. No change needed
to `_trajectory_data`'s return shape.

## 2. Regenerate / electron-count / beam-FWHM controls

Three new marimo controls in the penetration tab, next to the existing
`penetration_realistic_ui` checkbox:

- **Regenerate button** (`mo.ui.button`): click count feeds the `seed`
  passed to `trajectory_volume_figure`/`_trajectory_data`. Every click
  produces a fresh random draw; unrelated controls (electron count, FWHM,
  tilt, energy, realistic toggle) re-render with the *same* seed until
  Regenerate is clicked again.
- **Electron-count slider** (`mo.ui.slider(start=50, stop=500, value=250,
  step=10)`):
  value IS the number of electrons actually transported and drawn — no
  hidden multiplier. This repurposes/removes `_NE_SCALE` (currently `x4`)
  for this call site; the beam-bundle visual density that `_NE_SCALE` was
  compensating for is now the user's direct choice via the slider.
- **Beam-FWHM number box** (`mo.ui.number`, mm, default 1.0): only takes
  effect when `penetration_realistic_ui.value` is True, overriding the
  current `case.get("beam_fwhm_mm") or 1.0` default passed to
  `trajectory_volume_figure(..., realistic=True)`. The zoomed (default)
  view keeps its fixed `_ZOOM_BEAM_FWHM_MM` visualization-only spot,
  unaffected by this box.

`trajectory_volume_figure` signature changes from
`(rec_or_case, *, Ne=40, seed=0, realistic=False)` to
`(rec_or_case, *, Ne=40, seed=0, realistic=False, beam_fwhm_mm=None,
reveal_until_fs=None)` — `beam_fwhm_mm=None` preserves today's internal
default-selection logic (`case.get(...) or 1.0` in realistic mode); passing
a value overrides it. `reveal_until_fs` is the playback hook (section 3).

## 3. Real-time playback

**Time source:** the cascade's own physical clock, `t_fs` (already computed
per segment as `t_ang / C_ANG_PER_FS`, the segment's *start* age). This is
a genuine physical-time scrub, not an electron-count or segment-count
reveal.

**Frame model:** discretize `[0, T_max]` into a fixed `N_FRAMES = 60`
cutoffs, `T_max = max(t_fs + segment duration)` over the current dataset
(segment duration ≈ `L_ang / (beta * C_ANG_PER_FS)`, or approximate via
consecutive segment start-age deltas — pin down exact formula in the plan).
At frame cutoff `T`, a segment is drawn in full if its start age `<= T`,
otherwise omitted entirely (**whole-segment reveal, no sub-segment
interpolation** — segments are already short and numerous per track, so
stepping through 60 frames should read as continuous growth without needing
to interpolate a partial line within one segment). An electron's exit-path
dash appears once its terminal segment's start age `<= T` (instant
appearance, not grown further — it's a fixed decorative continuation, not
additional transport).

**Controls:**
- **Play/Pause** (`mo.ui.switch`).
- **Speed** (`mo.ui.slider`, e.g. 0.25x–4x, default 1x): scales how fast
  the frame index advances per wall-clock tick.
- **Repeat** (`mo.ui.switch`): when the frontier reaches frame `N_FRAMES-1`,
  either loop back to frame 0 (Repeat on) or stop and flip Play to Pause
  (Repeat off).

**Mechanism:** `mo.ui.refresh(default_interval=...)` drives a periodic
cell re-run while Play is on (marimo's built-in polling/timer primitive —
no custom JS/websocket work needed). A persistent frame-index counter
(`mo.state` get/set pair) advances by `speed`-scaled steps each refresh
tick, wrapping or clamping per Repeat. The current frame index maps to a
`reveal_until_fs` cutoff (`frame_index / (N_FRAMES-1) * T_max`) passed into
`trajectory_volume_figure`. When Play is off, no refresh timer runs and the
figure renders at whatever frame index it was left on (or fully revealed if
never played) — scrubbing a slider by hand is out of scope, matching the
button-triggered ask.

**Data plumbing:** `track_vertices_3d` gains an optional cutoff-aware
sibling or parameter to filter by a `t_fs`-per-segment array before
building the NaN-separated polyline; `_trajectory_data`'s return dict needs
`t_fs` exposed at natural segment order (it already is, as
`segs["t_ang"] / C_ANG_PER_FS`) for this filter to zip against
`start_xyz`/`end_xyz`/`E`/`elec_id`. `reveal_until_fs=None` in
`trajectory_volume_figure` means "no filter, current full-reveal behavior"
— all non-playback call sites (tests, non-penetration-tab usage) are
unaffected by default.

## Out of scope

- Side-exit continuation paths (realistic mode, transverse face).
- Sub-segment interpolation during playback (a segment "growing" mid-flight
  rather than popping in whole).
- Scrubbing playback by hand (a draggable frame slider) — only
  Play/Pause + Speed + Repeat, per the ask.
- Any change to the 2D matplotlib/Altair penetration views, the dense grid,
  or non-penetration-tab consumers of `trajectory_volume_figure`.

## Testing

- Extend `tests/test_plotly_trajectories.py`: exit-path trace present for a
  case with both backscattered and transmitted electrons (existing test
  case at 30 keV / hopg thin slab should exhibit both); trace named/dashed
  distinctly; `reveal_until_fs` filters tracks (fewer segments than
  unfiltered, all filtered segments' implied age `<=` cutoff);
  `beam_fwhm_mm` override changes the drawn beam-footprint outline in
  realistic mode.
- New: a frame-index/`reveal_until_fs` mapping helper (pure function, no
  marimo dependency) gets a direct unit test — the marimo wiring itself
  (refresh timer, state) is exercised via `notebook-workflow` skill
  conventions / `scripts/dev.py test` marimo app smoke checks, not unit
  tests of marimo internals.
