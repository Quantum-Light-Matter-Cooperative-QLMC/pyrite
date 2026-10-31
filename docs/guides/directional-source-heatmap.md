(detector-free-directional-source-heatmaps)=
# Compute a detector-free directional source map

Use [Angular radiation distributions](../notebooks/designing/05-angular-radiation-distributions.ipynb)
for the guided public-API example, complete source and captured map. Directional
scoring is **experimental** and reuses one transported electron population for
all directions.

## Choose and compute the map

1. Define the beam and target. Set the target footprint to the physical sample
   dimensions when side-face escape matters.
2. Choose observation angles and an explicit photon-energy point grid. Set
   equal CXR and bremsstrahlung counts in `FixedSampling` for the shared
   directional population.
3. Call `pyrite.radiation.directional_source` for PXR/CBS, or
   `directional_sources(..., components=...)` for separate supported components.
4. Plot each returned density on its own angle and energy axes. Retain its
   support mask, resolved seed and provenance with the result.

The [directional-source reference](../reference/directional-sources.md)
owns supported component selectors, grids, geometry and API restrictions;
[generated signatures](../reference/python-api/radiation.md#provisional-directional-source-api)
cover the calls. The source includes PXR–CBS interference and target absorption;
it excludes cross-electron coherence, downstream filters, detector acceptance
and detector response. These are simulated source densities, not measured counts.

To reproduce the earlier fixed seeded CPU map, download
{download}`directional_source_heatmap.py <../examples/directional_source_heatmap.py>`
and install `pyrite-mc[plots]` with the
[reference tables](getting-started.md#fetch-reference-data):

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
PYRITE_MC_BACKEND=cpu python directional_source_heatmap.py
```

That script uses 4,096 histories, 48 directions, 513 energies and a 100 nm silicon
film with a finite 5 × 5 mm footprint. Its
{download}`stored output and provenance <../examples/data/directional_source_heatmap.json>`
retain the resolved models and table identity. Continue with the notebook to
explore the map interactively.

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

The stored run was made with the opt-in coverage screen enabled
(`PYRITE_LINE_DATA_COVERAGE=1`) and emits `LineOpticalDataWarning`: some line contributions encounter
unavailable optical data and may be omitted. The stored fixed-run output records these
warnings; their pair counts are not a missing-yield estimate. See the
[runtime coverage screen](../physics/radiation-physics/coherent-radiation.md#runtime-kinematic-screen)
before interpreting absolute intensities.

For a different beam or target, check the reported
`source.provenance["line_window_ceiling_eV"]`: a photon stop below that ceiling
can truncate forward lines, especially at relativistic energies. The
[grid contract](../reference/directional-sources.md#photon-grids-and-limits)
explains bounds and restrictions. The {download}`stored output and provenance <../examples/data/directional_source_heatmap.json>`
record this run's seed, grids, resolved models and table identity; source hashes
pin that fixed example and its output.

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
