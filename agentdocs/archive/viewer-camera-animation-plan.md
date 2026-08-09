# Plan: `cxr viewer` camera fit + animation prerender

Archived historical plan. Shipped code and tests, not this file, define current
behavior.

Target: `notebooks/trace_app.py` penetration tab, backed by
`src/cxr_mc/plots/plotly_trajectories.py`. Two independent workstreams.

## 1. Camera: fit default (zoomed) view onto data

### Problem

Default (`realistic=False`) view is too zoomed out. Causes, in
`trajectory_volume_figure_from_data`:

1. `scene.aspectmode = "data"` + Plotly auto-range fits the bounding box of
   **every trace**, not the cascade region. Decorations inflate that box:
   - incident beam stubs extend `0.28 * span` upstream (`_incident_beam_lines`)
   - detector-direction arrow extends `0.28 * span` (`_direction_arrow`)
   - exit paths extend `0.28 * span` past exit faces (`_exit_paths_3d`)
   - vacuum legs can run further still
   Net: crystal + tracks occupy maybe half of each axis range, less on z.
2. Fixed camera eye `{"x": -0.8, "y": 1.5, "z": -0.4}` (norm ≈ 1.75, in
   Plotly's normalized scene units where default is 1.25·√3 ≈ 2.17). Distance
   is relative to the *inflated* scene cube, compounding cause 1.

`realistic=True` (true-scale) view is explicitly out of scope — keep as is.

### Approach

All changes gated on `realistic=False`; realistic branch keeps current
auto-range behavior.

**Step 1 — explicit axis ranges around the cascade region.**
In `trajectory_volume_figure_from_data`, when `realistic=False`, set
`scene.{x,y,z}axis.range` explicitly from `_display_extent` output (already
the fitted window around tracks):

- x, y: `[lo - m, hi + m]` with margin `m = 0.08 * span` (tune 5–12%).
- z: `[-0.15 * span, thick + 0.15 * span]` — small upstream sliver so beam
  stub entry points read, small downstream sliver for exit-face context.
- Ranges must be computed on **rotated** (lab-frame) extents: the crystal box
  goes through `R = _case_R(case)`, so derive ranges from the rotated corner
  set of the `_display_extent` box (min/max per axis of
  `_rotate(corners, R)`), padded as above. Otherwise a tilted slab clips.
- Decorations that poke past the ranges get clipped by Plotly — acceptable
  and intended. Optionally shorten stub/arrow length factor from `0.28` to
  `~0.18 * span` so less is clipped; visual call during implementation.
- Switch `aspectmode` to `"manual"` with `aspectratio` proportional to the
  three range widths (normalized so max = 1). This reproduces `"data"`
  proportions while respecting explicit ranges (Plotly's `"data"` mode +
  explicit ranges interact poorly; manual is the reliable path).

**Step 2 — bring camera in.**
Keep eye *direction* (current view angle is deliberate), rescale norm:
`eye = current_eye / 1.75 * D` with `D ≈ 1.25` as starting point. Expose as
module constant `_CAMERA_EYE` with comment; tune by eyeball against 2–3
cases (thin slab, thick slab, tilted).

**Step 3 — preserve user orbit.**
`uirevision="penetration-volume"` already keeps camera across param reruns.
Ranges changing per case would reset axes under `uirevision` only if the
revision key changes — keep key stable; verify scrub/orbit still persists
after range changes (manual check in `cxr viewer --smoke` + live).

### Tests

- `test_plotly_trajectories.py`: zoomed figure has explicit finite ranges
  covering the rotated crystal box + margin; realistic figure has no explicit
  ranges (unchanged); aspectratio proportional to range widths; camera eye
  norm equals `_CAMERA_EYE` norm.
- Tilted-case regression: rotated box corners all inside ranges.

## 2. Animation: faster + more frames via prerender

### Problem

`N_FRAMES = 8` is a **payload** ceiling, not compute: Plotly frames replace
trace arrays wholesale, so total payload grows ~linearly in frame count
(60 frames ≈ 518 MB, 8 ≈ 36 MB at Ne=250). No delta/append mechanism exists
client-side without custom JS. 8 frames over ~60 s reads choppy.

### Approach: prerendered video REPLACES the interactive animation

(Revised 2026-07-26: interactive Plotly frame animation is dropped, not
kept.) `trajectory_volume_animation`, `N_FRAMES`, the frame-payload
machinery, its tests, and the trace_app Speed slider are all removed. The
penetration tab keeps a static full-reveal figure (orbit/hover intact, via
`trajectory_volume_figure_from_data`) plus the **Render** button producing a
smooth fixed-camera looping video. `frame_reveal_fs` / `dataset_t_max` /
`trajectory_volume_data` / `trajectory_volume_figure_from_data` survive —
the renderer uses them.

**Step 1 — library: offscreen frame renderer.**
New module `src/cxr_mc/plots/render_trajectories.py`:

```python
def render_reveal_animation(
    rec_or_case, data, out_path, *,
    realistic=False, beam_fwhm_mm=None,
    n_frames=60, fps=12, width=960, height=560,
    camera=None, progress_cb=None,
) -> Path
```

- Loop `k in range(n_frames)`: `cutoff = frame_reveal_fs(k, t_max, n_frames)`,
  build figure via existing `trajectory_volume_figure_from_data(...,
  reveal_until_fs=cutoff)`, strip `updatemenus`/`sliders`, pin camera +
  ranges (reuse workstream 1 fit), export PNG via kaleido
  (`fig.to_image`).
- Stitch PNGs to `.mp4` (h264 via `imageio-ffmpeg`) — falls back to `.gif`
  if ffmpeg unavailable. Write to a cache dir (see Step 3).
- `progress_cb(k, n_frames)` hook for UI progress.
- Pure function, no marimo import — unit-testable with tiny Ne and
  `n_frames=3`.
- Reuses transported `data` dict — **no new Monte Carlo transport**, no new
  physics, no ledger entry needed.

Deps: `kaleido` + `imageio[ffmpeg]` as a new optional extra
(`viz-render`) in `pyproject.toml`; renderer raises a clear
`ImportError`-derived message naming the extra if missing.

**Step 2 — trace_app wiring: Render button.**
In penetration tab:

- `mo.ui.run_button(label="Render smooth animation")` + frame-count slider
  (24–120, default 60) + fps slider optional (or fix fps=12).
- On click: cell calls `render_reveal_animation` with cached `_data` dict,
  wrapped in `mo.status.progress_bar` driven by `progress_cb`. Marimo cells
  are blocking — a 60-frame kaleido render is ~1–2 min; progress bar makes
  that acceptable. (If too slow in practice, follow-up: thread + `mo.state`
  poll; do not build that speculatively.)
- Result displayed via `mo.video(src=path, controls=True, loop=True)` below
  (or replacing, behind a toggle) the interactive figure. Video loops —
  fixes the no-repeat caveat of Plotly's Play button for free.

**Step 3 — cache.**
Key: hash of `(material, case params, Ne, seed, realistic, beam_fwhm_mm,
n_frames, fps, camera, module version salt)`. Store under
`~/.cache/cxr-mc/viewer-renders/<key>.mp4`. Button short-circuits to cached
file when present; slider/param change invalidates via key change. Small LRU
sweep (keep last ~20 files) to bound disk.

**Step 4 — remove interactive animation path.**
- Delete `trajectory_volume_animation`, `N_FRAMES`, and the
  `test_trajectory_volume_animation_*` tests.
- trace_app: replace animation figure with static full-reveal
  `trajectory_volume_figure_from_data` figure; drop the Speed slider; keep
  the Ne slider and data-dict cache.

### Tests

- `render_reveal_animation`: produces file, frame count honored (probe via
  imageio metadata), `n_frames=3`/tiny-Ne smoke, progress_cb called n times,
  missing-deps error message names the extra. Mark kaleido-dependent tests
  skip-if-unavailable.
- Cache: same key hits cache (mtime unchanged), param change misses.
- trace_app: `uvx marimo check` clean; `cxr viewer --smoke` passes.

### Sequencing

1. Workstream 1 (camera) first — small, independent, and the render path
   pins the same camera/ranges, so fit must land first.
2. Renderer library + tests.
3. Deps extra + cache.
4. Remove interactive animation path (library + tests), then trace_app
   wiring: static figure + Render button + video (`notebook-workflow` for
   the app edit; no CLI surface change).
