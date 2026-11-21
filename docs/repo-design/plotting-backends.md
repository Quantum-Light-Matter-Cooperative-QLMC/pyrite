# Plotting backends

`pyrite.plots` prepares plot data independently of the renderer and supports
two rendering backends, Plotly and Matplotlib. Display code never changes source arrays, physical
normalization (`Phs/s/eV/uA`), units, peak treatment, checkpoint or result
schemas. This page records the current backend split and the browser evidence
behind it (#622).

## Current split

| Use | Backend | Entry points |
| --- | --- | --- |
| Interactive 2D analysis: spectra, overlays, case baskets, temporal profiles, material comparisons, metric trends, scan and pixel heatmaps, Timepix3/Eagle XO response spectra, 2D penetration tracks and survival, saved-trajectory summaries | Plotly | `pyrite.plots.plotly.spectra`, `.temporal`, `.sweeps`, `.pixels`, `.detectors`, `.penetration`, `.trajectory_summary` |
| Physics, channeling diagnostics, validation and publication figures | Matplotlib | `pyrite.plots.mpl` |
| Dense captured trajectories | Datashader raster (Matplotlib); existing Plotly 3D track views | `pyrite.plots.mpl.trajectories`, `pyrite.plots.plotly.trajectories` |
| Click-selectable azimuth × polar-tilt Map | Plotly | `pyrite.plots.plotly.map_select` |

Numerical frames come from `pyrite.plots._spectrum_frames`,
`._temporal_frames`, `._frames`, `._trajectory_frames`, `._pixel_frames` and
`._detector_frames`. Plotly builders return a `Figure`, or `None` for empty
results; pixel views take prepared frames. Static Plotly export (Kaleido) is
optional. The former Altair renderer (`pyrite.plots.altair`) was removed in
#622; its frame builders moved to the neutral modules above.

Spectra are decimated **before** figure construction by the peak-aware
paired-component sampler (`max_points`, default 5000 per chart, 2500 with both
emissions). Pixel images are dense `Heatmap` arrays with masked missing cells,
raw hover values and row zero at the top; the logarithmic count view uses a
signed `log1p` transform labelled in physical counts.

## SVG versus WebGL for spectrum traces

A spectrum trace switches to `Scattergl` only above `_WEBGL_MIN_POINTS =
100_000` points (`pyrite.plots.plotly.spectra`); decimated app spectra always
render as SVG. `checks/plotly_browser_bench_622.py` measured the production
trace builder on synthetic line + continuum spectra in headless Firefox 160 and
Chrome 158 with GPU WebGL (Mesa Intel Raptor Lake-P iGPU, Plotly 7.1.0 bundled
plotly.js). Medians of five repeats after one warm-up; times include two
animation frames, so they are quantized to about 16.7 ms. Full matrix:
`checks/plotly_browser_bench_622_results.json`.

| Points × traces | Firefox render SVG / WebGL (ms) | Firefox zoom SVG / WebGL (ms) | Chrome render SVG / WebGL (ms) | Chrome zoom SVG / WebGL (ms) |
| --- | ---: | ---: | ---: | ---: |
| 5,000 × 4 | 39 / 143 | 33 / 34 | 33 / 83 | 34 / 34 |
| 6,000 × 12 | 67 / 159 | 34 / 51 | 50 / 117 | 33 / 33 |
| 25,000 × 12 | 133 / 217 | 71 / 50 | 100 / 183 | 34 / 33 |
| 100,000 × 4 | 151 / 367 | 84 / 35 | 117 / 334 | 50 / 34 |
| 100,000 × 12 | 401 / 1112 | 218 / 96 | 417 / 683 | 117 / 34 |

SVG renders faster at every measured size. WebGL only shortens zoom and pan
redraws, and only clearly at about 100k points per trace. Each WebGL trace also
uses one of the browser's few WebGL contexts per page, a real limit in a marimo
app that shows many charts. The previous 6,000-point switch therefore cost
render time with no interaction benefit. Headless timing on one integrated
GPU does not cover hover latency, discrete GPUs or other operating systems.

## Map cell selection

The analysis Map draws a single-panel Plotly heatmap from the same `heatmap_frame`
values, selection score, flux gate, energy panel and units, with a nearly
transparent `Scatter` click target over each *finite* cell. Missing, gated and
NaN cells have no target. The marimo `on_change` handler resolves the clicked
x/y against the displayed frame (not browser-supplied metadata), rejects
ambiguous multi-cell events, and stores the geometry with its scope
(material, face, checkpoint, beam energy, thickness). Pan and zoom are enabled.

`checks/plotly_map_pilot_622.py` writes a synthetic azimuth × tilt checkpoint,
launches `analysis_app.py` and drives the gate with real pointer events in
headless Firefox 160 and Chrome 158 (marimo 0.25.2, Plotly 7.1.0). Results for
8 × 8 and 25 × 25 grids: `checks/plotly_map_pilot_622_results.json`.

| Gate step | Firefox | Chrome |
| --- | --- | --- |
| Single click at a finite cell centre selects it (about 1 s to spectra) | Pass | Pass |
| Four further cells including corners, 8 × 8 and 25 × 25 | Pass | Pass |
| Pan drag starting on another cell keeps the selection | Pass | Pass |
| Click on a flux-gated blank cell selects nothing | Pass after fix | Pass after fix |
| Map quantity change keeps a still-visible selection | Pass | Pass |
| Light/dark theme change keeps the selection | Pass | Pass |
| Beam energy or thickness change hides the selection | Pass | Pass |
| Keyboard focus and cell selection | Fail | Fail |
| Map zoom redraw | 50–90 ms | 30–360 ms (first) |

The pilot found two defects, both fixed in `selectable_map_figure`: Plotly
snaps a click to the nearest point within `hoverdistance` (default 20 px) of a
marker, so a click on a blank cell selected its neighbour; and the horizontal
colorbar title squeezed the x axis to about 135 px, below the cell size the
target markers assumed. The colorbar title is now vertical and
`hoverdistance=1` accepts only clicks on a target marker.

The stored choice is scope-bound rather than discarded: after an energy change
hides it, returning to the original energy shows it again. Keyboard cell
selection is not available; the former Altair canvas Map lacked it too. An
installed-wheel marimo run has not been tested.

**Decision:** Plotly is the only Map renderer and Altair is removed (#622). The
pointer gate passes in both browsers; keyboard access remains a known gap.
`altair`, `vegafusion` and `vl-convert-python` are no longer PyRITE
dependencies (`marimo[recommended]` still installs `altair` for marimo itself).
