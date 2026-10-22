# Plotting backends (migration issue #622)

Use **Plotly** for interactive spectra, temporal curves, case comparisons, metric scans, sweep heatmaps and pixel images. Use **Matplotlib** for physics figures, channeling diagnostics, validation and print-quality exports. Dense captured trajectories continue to use **Datashader** where appropriate; existing Plotly 3D track views remain unchanged.

The interactive builders are importable as `pyrite.plots.plotly.spectra`, `.temporal`, `.sweeps`, `.pixels`, and `.detectors`. Their numerical inputs come from `pyrite.plots._spectrum_frames`, `._temporal_frames`, `._frames`, `._pixel_frames`, and `._detector_frames` respectively. The legacy `pyrite.plots.altair` API is retained. These chart functions return Plotly `Figure` objects and `None` for empty results, apart from pixel views which expect prepared frames.

The analysis app uses Plotly for regular spectra, overlays, baskets, temporal profiles, material comparisons, metric trends, fallback scan charts, pixel-detector views and Timepix3/Eagle XO detector-response spectra. **The click-selectable azimuth × tilt map remains Altair**: its marimo `.value` contract is not interchangeable with Plotly selection, so it must not be migrated without the interaction-gate tests in #622. Detector response spectra share the exact physics frame functions in `pyrite.plots._detector_frames` across Plotly and legacy Altair. The dual-axis Eagle XO QE plot uses a secondary Plotly y-axis. Headless static exports remain optional (Kaleido).

Spectrum decimation occurs **before** building Plotly figures, using the existing peak-aware energy-grid algorithm and paired-component selection. Original checkpoint data, source normalization (`Phs/s/eV/uA`), broad bremsstrahlung tails, emission components, and output schemas are unchanged. Plotly switches to WebGL only for individual long traces (more than 6,000 points); ordinary charts use SVG. The threshold is provisional, not a measured browser benchmark.

Pixel images are rendered as dense 2D `Heatmap` arrays with masked missing cells, raw hover values and row zero at the top. Display blocks respect pixel range boundaries; a selected block is outlined. The logarithmic appearance for counts uses a signed `log1p` transform with labels in physical counts rather than dropping zeros. Reporting histograms retain exact left/right energy edges and half-open bin semantics.

The migration needs follow-up work: a dedicated Plotly↔marimo click-selection pilot, end-to-end browser benchmarks (Firefox and Chromium), rendered screenshot comparisons, broader headless export/package tests and removal of redundant legacy frame definitions. These have not been completed by the first two incremental slices. Plotly and Altair remain in the `plots` extra while both are used.

## Map selection pilot (issue #622, slice 3)

The **Map** view now has a renderer selector. **Altair (stable)** remains the
initial choice and retains its original marimo point-selection contract. The
**Plotly (experimental)** choice is a single-panel map using the same
`heatmap_frame` values, selection score, flux gating, energy panel and units.
It overlays small `Scatter` click targets over *finite* cells because Plotly's
heatmap selections may require a drag gesture; click near a cell's center.
This intentionally leaves missing, gated and NaN cells unselectable and keeps
zero-valued cells valid. Pan/zoom are supported. The Plotly experiment is
restricted to the Map view; other heatmaps continue to use standard Plotly
`Heatmap` without synthetic point targets.

Marimo supplies a list of point dictionaries to the plot's `on_change` handler.
The handler resolves x/y against the currently displayed frame (not untrusted
browser record metadata), rejects ambiguous multi-cell selections, and stores
just the selected geometry in marimo state. Rendering the selected spectrum
uses that state without rebuilding heatmap metrics; a change of metric or theme
keeps the choice *only if* the same source record remains visible. Checkpoint,
material, face, incident energy or thickness changes invalidate the stored
choice. The default Altair widget does **not** use this state path.

**Pilot validation gate (not yet complete):** In both Firefox and Chromium,
load a saved azimuth × tilt sweep, choose Plotly, then verify a single click
near the center of a finite cell updates its narrow/broad spectra; dragging to
pan does not select another cell; a gated/blank cell does not select anything;
changing map quantity and light/dark theme retains a valid selected geometry;
changing energy or thickness clears stale selections; returning to Altair
recovers the original click behavior. Test keyboard access, dense 25×25 grids,
zoom/pan responsiveness and installed-wheel marimo separately before changing
the default. If Plotly click behavior fails any gate, retain Altair.
