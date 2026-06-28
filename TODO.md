# TODO

**jupyter to marimo.** *in progress on this feature branch (feature/marimo-transfer)*
Here we are exploring the move to marimo from jupyter notebooks,
primarily to improve data visualization and interactivity.
As part of this, we are experimenting with replacing matplotlib with altair
due to the extreme sluggishness and compute-weight of the matplotlib plots

Progress (Altair renderers in `src/cxr_mc/plots/altair_*.py`, marimo apps in `notebooks/*_app.py`):
- [x] Altair spectra (`altair_spectra.py`) — intrinsic by-energy spectra
- [x] Altair detectors (`altair_detectors.py`) — Timepix/Eagle detected, Eagle charge density
- [x] Altair sweeps (`altair_sweeps.py`) — metric-vs, heatmaps, auto-pick scan
- [x] Altair trajectories (`altair_trajectories.py`) — penetration survival + interactive low-Ne tracks
- [x] marimo `scan_app.py` (runner) + `analysis_app.py` (reactive viz, wires the Altair renderers)

Remaining / deferred:
- Stays on matplotlib by design: datashader trajectory grid (segment count too high for Vega-Lite), efficiency curves, best-spectra, charge-map, cross-material comparison.
- Not yet ported: spectra "chunk" (best-azimuth total|CXR) and "full" (log-log measured range) views; the interactive `browse()` click-through (replaced by marimo dropdowns).
- Once marimo apps are trusted, retire the `.ipynb`/`.md` notebook pairs and make the marimo apps canonical (update README/repo_map/skills).

Also covering notebook/repo reorg (PA #3 deferred steps):

3. ~~**Fix `checkpoint_path_for` to anchor to repo root.**~~ Done — `_DEFAULT_CHECKPOINT_DIR` in `src/cxr_mc/run.py` anchors to repo root via `Path(__file__).resolve().parents[2]`.

4. ~~**Move notebooks to `notebooks/`.**~~ Done — `scan.ipynb`, `analysis.ipynb` (+ `.md` pairs) moved to `notebooks/`; `repo_map.md`, `README.md`, `docs/running-on-a-cluster.md`, and the `notebook`/`repo-orientation` skills updated.
