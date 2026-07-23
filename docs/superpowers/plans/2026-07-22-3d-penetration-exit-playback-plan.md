# Plan: 3D penetration exit-paths, controls, real-time playback

Derived from `docs/superpowers/specs/2026-07-22-3d-penetration-exit-playback-design.md`.
Execute via subagent-driven-development. Branch: `feat/3d-penetration-playback`.

## Global Constraints (bind every task)

- **Scope files:** only `src/cxr_mc/plots/plotly_trajectories.py` and
  `notebooks/analysis_app.py` (penetration tab). **No change** to
  `src/cxr_mc/plots/trajectories.py` 2D matplotlib/Altair views,
  `_trajectory_data`'s return shape, `plot_trajectory_grid`, or
  `trajectory_chart`. `_trajectory_data` already returns `t_fs`, `elec_id`,
  `start_xyz`, `end_xyz`, `E`, `L` in natural per-segment order — use them,
  do not modify that function.
- **Final `trajectory_volume_figure` signature (reached across Tasks 1 & 2):**
  `trajectory_volume_figure(rec_or_case, *, Ne=40, seed=0, realistic=False, beam_fwhm_mm=None, reveal_until_fs=None)`.
  Task 1 adds `beam_fwhm_mm`; Task 2 adds `reveal_until_fs`. Both default
  `None` → today's behavior, so every existing caller/test is unchanged by
  default.
- **`beam_fwhm_mm` semantics:** `None` preserves current realistic-mode
  default (`case.get("beam_fwhm_mm") or 1.0`); a passed value overrides it.
  Only affects `realistic=True`. Zoom (default) view keeps the fixed
  `_ZOOM_BEAM_FWHM_MM` visualization spot, unaffected.
- **Electron count is literal:** the `Ne` passed by the penetration tab IS
  the number transported and drawn — no hidden multiplier. `_NE_SCALE`
  (currently `x4`) is removed from the `trajectory_volume_figure` call site.
- **Exit paths:** only top face (`z ≈ 0`, backscattered) and bottom face
  (`z ≈ thick`, transmitted). Side exits NOT drawn. Terminal segment per
  electron = max `t_fs` among that electron's segments. Classify its
  `end_xyz` z against `0` and `thick` within a small tolerance. Stopped /
  interior-terminated / side-exited electrons get no line. Style: dashed,
  low-opacity, fixed length `0.28 * span`, one muted single color (NOT the
  turbo colorscale), exactly ONE legend entry named `"exit path"` for all
  electrons.
- **Playback frame model:** `N_FRAMES = 60`. `T_max = max(t_fs)` over finite
  segment start-ages in the current dataset (whole-segment reveal is by
  START age, so `max(t_fs)` is exactly sufficient for full reveal at the
  final frame — no segment-duration term needed). Frame-index → cutoff:
  `reveal_until_fs = frame_index / (N_FRAMES - 1) * T_max`. A segment is
  drawn in full iff its start age `<= reveal_until_fs`, else omitted whole
  (no sub-segment interpolation). An electron's exit-path dash appears once
  its terminal segment's start age `<= reveal_until_fs`.
- **`reveal_until_fs=None` means no filter** (current full-reveal behavior).
- **marimo controls** (Task 3), next to `penetration_realistic_ui`:
  Regenerate button (`mo.ui.button`; click count feeds `seed`),
  electron-count slider (`mo.ui.slider(start=50, stop=500, value=250, step=10)`),
  beam-FWHM number (`mo.ui.number`, mm, default 1.0), Play/Pause
  (`mo.ui.switch`), Speed (`mo.ui.slider`, 0.25x–4x, default 1x), Repeat
  (`mo.ui.switch`). `mo.ui.refresh` drives periodic re-run while Play on;
  `mo.state` get/set holds frame index. Hand-scrubbing a slider is OUT of
  scope.
- Every new/changed physics-free helper is fine, but any docstring touching
  transport must stay accurate. Run repo lint/typecheck/tests via
  `rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py ...`.

## Out of scope

Side-exit continuation, sub-segment interpolation, hand-scrubbing slider,
any 2D/dense-grid/non-penetration-tab change.

---

## Task 1 — Exit paths + `beam_fwhm_mm` override + drop `_NE_SCALE`

**File:** `src/cxr_mc/plots/plotly_trajectories.py`, tests in
`tests/test_plotly_trajectories.py`.

Implements spec §1 and the `beam_fwhm_mm` half of §2.

1. **Signature:** change `trajectory_volume_figure(rec_or_case, *, Ne=40,
   seed=0, realistic=False)` → add `beam_fwhm_mm=None` (do NOT add
   `reveal_until_fs` — that is Task 2). Update the docstring.
2. **Drop `_NE_SCALE` at this call site:** remove the `Ne = Ne * _NE_SCALE`
   line so the passed `Ne` is transported literally. Remove the `_NE_SCALE`
   constant and its comment if no longer referenced (grep first). The bundle
   density that `_NE_SCALE` compensated for becomes the caller's choice.
3. **`beam_fwhm_mm` plumbing:** in `realistic=True`, use
   `beam_fwhm_mm if beam_fwhm_mm is not None else (case.get("beam_fwhm_mm") or 1.0)`
   for BOTH the `_trajectory_data(..., beam_fwhm_mm=...)` transport call and
   the `_beam_footprint_outline` overlay, so the drawn footprint matches the
   transported spot. Zoom branch unchanged (`_ZOOM_BEAM_FWHM_MM`).
   Note `_beam_footprint_outline` reads `case.get("beam_fwhm_mm")` today —
   make it honor the override (pass the resolved value in, e.g. via a small
   param, rather than mutating `case`).
4. **Exit paths:** add a helper that, from the existing `data` dict
   (`start_xyz`, `end_xyz`, `elec_id`, `t_fs`, `thick`), for each electron
   finds the terminal segment (argmax `t_fs` within that electron), and if
   its `end_xyz` z is within tolerance of `0` (backscattered) or `thick`
   (transmitted), emits a dashed straight continuation from the exit point
   along the terminal segment's own direction (`end_xyz - start_xyz`,
   normalized), of display length `0.28 * span`. Collect all such dashes
   into ONE `go.Scatter3d` (NaN-separated, one legend entry `"exit path"`,
   muted single color, `dash="dash"`, low opacity). Tolerance: pick a small
   fraction of `thick` (e.g. `1e-6 * thick` or a few display-unit epsilons)
   — document the choice. Add the trace to the figure only when at least one
   exit path exists. Reuse existing muted palette constant or add one.

**Tests (extend `tests/test_plotly_trajectories.py`):**
- Exit-path trace present, named `"exit path"`, `dash="dash"`, for the
  30 keV / hopg thin-slab case that exhibits both backscattered and
  transmitted electrons. Assert exactly one such trace (one legend entry).
- `beam_fwhm_mm` override changes the drawn beam-footprint outline in
  realistic mode (compare footprint trace coordinates for two different
  `beam_fwhm_mm` values → different extents).
- Existing tests still pass (default-arg behavior unchanged; note `Ne` is
  now literal — the existing `Ne=6`/`Ne=4` tests should still produce a
  valid figure).
- `fig.to_json()` stays serializable.

## Task 2 — Playback data plumbing: `reveal_until_fs` filter + frame helpers

**File:** `src/cxr_mc/plots/plotly_trajectories.py`, tests in
`tests/test_plotly_trajectories.py`. Depends on Task 1 (same file/signature).

Implements spec §3 library side.

1. **`track_vertices_3d` cutoff filter:** add an optional parameter (e.g.
   `t_fs=None`, `reveal_until_fs=None`) so that when both are provided, only
   segments whose start age `t_fs <= reveal_until_fs` contribute
   start/end/NaN triples. `None` → today's full behavior (all segments).
   Keep the existing shape/length validation. Zip the cutoff against the
   same per-segment order as `start_xyz`/`end_xyz`/`E`/`elec_id`.
2. **`trajectory_volume_figure` `reveal_until_fs=None` param** (final piece
   of the target signature): thread it into `track_vertices_3d` for the
   in-crystal tracks AND gate the exit-path dashes — an electron's dash is
   included only if its terminal segment's start age `<= reveal_until_fs`
   (when the cutoff is not None). `None` → full reveal (unchanged).
   `data["t_fs"]` supplies per-segment start ages.
3. **Pure frame helpers (no marimo import), unit-tested:**
   - `frame_reveal_fs(frame_index, t_max, n_frames=N_FRAMES)` →
     `frame_index / (n_frames - 1) * t_max`. Define `N_FRAMES = 60` as a
     module constant.
   - `dataset_t_max(data)` → `max` of finite `data["t_fs"]` (0.0 if empty).
   - `advance_frame(frame_index, speed, repeat, n_frames=N_FRAMES)` →
     `(next_index, still_playing)`: advance by `speed`-scaled step
     (round/int), wrap to 0 when reaching `n_frames-1` if `repeat` else
     clamp at `n_frames-1` and return `still_playing=False`. This encodes
     the Repeat loop/stop logic so Task 3's marimo cell stays thin.

**Tests:**
- `reveal_until_fs` filters tracks: a mid-range cutoff yields strictly fewer
  segment vertices than the unfiltered figure, and every revealed segment's
  implied start age `<= cutoff`.
- `frame_reveal_fs`: endpoints (`0 → 0`, `n_frames-1 → t_max`) and a
  midpoint.
- `advance_frame`: wrap when repeat, clamp + `still_playing=False` when not,
  speed scaling.
- `dataset_t_max` on a small synthetic `data` dict.

## Task 3 — marimo controls + playback wiring in penetration tab

**File:** `notebooks/analysis_app.py` (penetration tab only). Depends on
Tasks 1 & 2 (uses final signature + helpers). Use the `notebook-workflow`
skill conventions; keep notebook output-free.

Implements spec §2 (controls) and §3 (playback wiring).

1. **New controls** in the cell that currently defines
   `penetration_realistic_ui` (or adjacent cells): Regenerate `mo.ui.button`,
   electron `mo.ui.slider(start=50, stop=500, value=250, step=10)`,
   beam-FWHM `mo.ui.number(value=1.0, ...)`, Play/Pause `mo.ui.switch`,
   Speed `mo.ui.slider` (0.25–4, value 1), Repeat `mo.ui.switch`. Return them
   from their cell(s) and add to the consuming penetration-tab cell's args.
2. **Frame state:** `get_frame, set_frame = mo.state(0)` in a setup cell.
3. **Refresh timer:** `mo.ui.refresh` whose interval is active only while
   Play switch is on (per marimo idiom — e.g. gate by passing
   `default_interval` and reading `.value` to trigger re-run; if Play off,
   don't advance). On each tick while playing, call `advance_frame(get_frame(),
   speed, repeat)`; `set_frame(next_index)`; if `not still_playing`, flip the
   Play switch to off (Repeat-off stop behavior).
4. **Wire the figure:** compute `T_max` via `dataset_t_max` (needs the data;
   simplest: let `trajectory_volume_figure` accept the frame cutoff — pass
   `reveal_until_fs = frame_reveal_fs(get_frame(), T_max)` when Play has been
   used, else `None` for full reveal). Because `T_max` needs the dataset,
   either (a) expose a tiny helper to compute `T_max` for the case, or (b)
   compute the figure with `reveal_until_fs=None` first when idle. Keep it
   simple: when the frame index is 0 and Play was never on, render full
   (`reveal_until_fs=None`); once playing/advanced, pass the mapped cutoff.
   Pass `Ne=<slider value>`, `seed=<regenerate button click count>`,
   `beam_fwhm_mm=<number value>` (effective only in realistic mode),
   `realistic=penetration_realistic_ui.value`.
5. Add all new controls to the tab's control `vstack` next to
   `penetration_realistic_ui`.

**Verification (no unit test of marimo internals):** run
`rtk env UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run python scripts/dev.py test`
for the plotly tests, `scripts/dev.py lint` + `typecheck`, and the marimo app
smoke check per `notebook-workflow`. Confirm the notebook is output-free
(`scripts/dev.py nbstrip` clean). The pure helpers already have unit tests
from Task 2 — Task 3 adds no new unit tests, only wiring + smoke verification.
