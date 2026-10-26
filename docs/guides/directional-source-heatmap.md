# Detector-free directional source heatmaps

Map the escaped PXR/CBS spectrum around a silicon film, then compare spectra at
three observation angles. This **experimental** API scores a detector-free
angular slice from one shared electron transport.

Install `pyrite-mc[plots]` and [fetch the reference tables](getting-started.md#install-and-fetch-reference-data).
Download {download}`the complete script <../examples/directional_source_heatmap.py>`
and run it in your installed environment:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
PYRITE_MC_BACKEND=cpu python directional_source_heatmap.py
```

The example uses 4,096 incident electrons, 48 angles and 513 photon energies on
CPU; first use also compiles kernels. Both sampling counts are set to 4,096 because
the directional scorer uses the same transported population for its components.
The target is 1000 Å (100 nm) thick with a finite 5 × 5 mm footprint.

```{literalinclude} ../examples/directional_source_heatmap.py
:language: python
:start-after: tutorial-start
:end-before: tutorial-end
```

```{eval-rst}
.. plot::
   :alt: Simulated silicon energy–angle map and three spectra showing angle-dependent PXR/CBS peaks and strong attenuation near directions parallel to the film faces.
   :caption: Actual seeded CPU simulation, precomputed from the downloadable script.

   from _tutorials import directional_source_heatmap
   directional_source_heatmap()
```

The peak energies change with observation angle. Compare the three spectra to
see why one detector direction does not describe the whole source. The map
includes intra-electron PXR–CBS interference and target self-absorption; it
excludes cross-electron coherent radiation, downstream filters, angular
acceptance and detector response. These are simulated source densities, not
experimental measurements.

## Coordinates and units

The incoming beam follows lab +z: 0° is forward, +90° points along +x and
±180° is backward. This is the **x–z plane slice**, not an azimuth-integrated
angular distribution. Angles and photon energies are point samples;
−180° is duplicated at +180° only when drawing the periodic heatmap.

The arrays have shape `(angle, energy)` and units
**photons/(incident electron · eV · sr)**. No angular-bin width or solid angle
is folded in. `source.rate(10.0)` would rescale to **ph/s/eV/sr** at 10 nA
without rerunning transport. See [spectral observables and normalization](../physics/radiation-physics/spectral-observables.md#detector-free-directional-rates).

## Finite-target escape

The finite footprint includes escape through side faces. Near directions
parallel to the film faces, a long absorbing path can produce a deep physical
minimum. With an infinite slab, face-parallel directions are instead unsupported
and returned as NaN with a support mask. See the
[escape contract](../reference/directional-sources.md#coordinates-and-escape)
and [finite-crystal validation](../validation/geometry/finite-transverse-crystal.md).

(photon-window-and-relativistic-forward-lines)=

## Resolution, statistics and limits

The 100–2000 eV window, 7.5° angular spacing, one reflection family and chosen
history count illustrate the workflow; they do **not** establish convergence or
uncertainty. Increase histories and repeat independent seeds, refine both grids
and check reflection convergence before interpreting fine structure or peak
heights. The finite energy window is not a total yield. A brighter region alone
does not establish statistical significance.

This run emits `LineOpticalDataWarning`: some line contributions encounter
unavailable optical data and may be omitted. The cached output records these
warnings; their pair counts are not a missing-yield estimate. See the
[runtime coverage screen](../physics/radiation-physics/coherent-radiation.md#runtime-kinematic-screen)
before interpreting absolute intensities.

For a different beam or target, check the reported
`source.provenance["line_window_ceiling_eV"]`: a photon stop below that ceiling
can truncate forward lines, especially at relativistic energies. The
[grid contract](../reference/directional-sources.md#photon-grids-and-limits)
explains bounds and restrictions. The {download}`stored output and provenance <../examples/data/directional_source_heatmap.json>`
record this run's seed, grids, resolved models and table identity; source hashes
keep the displayed figure synchronized with the example.

(reusable-source-calculation)=
(run-on-molab)=
(execution-backends)=

## Explore interactively

```bash
pyrite app angular-heatmap launch
```

The app adds target-orientation comparisons, linked cuts, current scaling and
separate radiation-component maps. See [app controls, downloads and molab](../reference/directional-sources.md#interactive-app)
for its contracts, including the distinct internal Cherenkov view. Review
[model capabilities](../physics/capabilities.md) and
[validation status](../validation/status-summary.md) before quantitative use.
