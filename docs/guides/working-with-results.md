# Working with results

Reopen the saved [quickstart run](getting-started.md), separate its radiation
components, and integrate a fixed photon-energy band. Loading and plotting a
checkpoint does not rerun electron transport.

## Plot one saved case

Install `pyrite-mc[plots]`. If you have not run the quickstart, first
[fetch its reference tables](getting-started.md#install-and-fetch-reference-data)
and create the checkpoint:

```bash
pyrite run quickstart
pyrite checkpoint list
```

Download {download}`working_with_results.py <../examples/working_with_results.py>`
and pass the checkpoint directory printed by the run. For the bundled inputs:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
python working_with_results.py pyrite-output/checkpoints/hopg@quickstart-cdf8eed9af74
```

A changed profile or catalog can produce a different stem: use your listed
quickstart directory. The script expects its single saved case: 30 keV electrons,
a 2 µm HOPG target, 45° polar tilt and 90° observation angle. These are campaign
checkpoint records, distinct from the in-memory `Result` returned by `simulate`.
The loading and reduction functions below are supported **advanced-stable** APIs.

```{literalinclude} ../examples/working_with_results.py
:language: python
:start-after: tutorial-start
:end-before: tutorial-end
```

```{eval-rst}
.. plot::
   :alt: Three saved HOPG source components and a comparison of their PXR/CBS and characteristic yields in the 100 to 1000 eV band.
   :caption: Actual quickstart checkpoint output. Grey shading marks the integrated band. The bar chart excludes bremsstrahlung; all curves retain their native saved grids.

   from _tutorials import working_with_results
   working_with_results()
```

The characteristic peak near 277 eV dominates the **100–1,000 eV** integral;
PXR/CBS produces separate narrow structure. The curves show density in
**photons/(incident electron · eV · sr)**. Integrating over energy removes
`/eV`, giving **photons/(incident electron · sr)**. `radiation_band_metrics`
interpolates band endpoints within the saved line grid and integrates its
additive line components; it excludes the continuum. The plotted `brem_wide`
uses its own wider grid, while the checkpoint's `brem` is already interpolated
onto the line grid. Do not add native arrays with different energy axes.

No current, acceptance, exposure, filter or detector-response factor has been
applied. These are simulated source components, not measured counts. The
checkpoint's `scale` and `source_current_na` support explicit flux conversion;
[detector-response recipes](../reference/python-scene-recipes.md#apply-detector-response-explicitly)
cover response folding.

The saved quickstart used seed 1 and stopped after 600 sampled histories under
its adaptive policy. That stopping decision does not establish convergence of
every spectral bin or the carbon-line integral. Check sampling, grid and
reflection convergence before quantitative use; consult the
[model capability matrix](../physics/capabilities.md) and
[validation status](../validation/status-summary.md). The
{download}`stored arrays and provenance <../examples/data/working_with_results.json>`
retain resolved inputs, adaptive diagnostics, dataset identity and input-file
hashes. Normal documentation builds plot those arrays without simulation or
table downloads.

(checkpoint-layout-and-identity)=
(analyze-or-export)=
(stored-components)=
(pixel-detector-observations)=
(preserve-or-reduce-data)=

## Other checkpoint workflows

Use the [checkpoint and app reference](../reference/results-and-apps.md) for
component files, dataset identity, interactive analysis, pixel observations,
comparison, export and archiving. For example, `pyrite app analysis launch hopg`
opens the saved-dataset picker with the `apps` extra installed.

(save-electron-trajectories)=
(find-and-describe-captures)=
(export-segments-for-visualization)=
(score-spectra-from-saved-trajectories)=
(export-scene-geometry)=
(inspect-trajectories-in-paraview)=
(paraview-preset)=
(analyze-saved-trajectory-captures)=

## Saved electron histories

Checkpoints retain spectra. [Saved trajectory workflows](../reference/trajectory-workflows.md)
cover separate capture artifacts, reductions, replay, VTK scenes and ParaView.

(planar-channeling-demo)=
(shaped-entry-wavefunctions-experimental)=

## Experimental quantum views

The [channeling app reference](../reference/results-and-apps.md#planar-channeling-demo)
and [planar channeling guide](experimental-planar-channeling.md) cover solver
visualization and the restrictions on entrance-state selection.
