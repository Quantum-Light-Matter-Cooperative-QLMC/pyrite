# Checkpoints and analysis apps

For a worked component plot, use [Working with results](../guides/working-with-results.md).
Apps require `pyrite-mc[apps]`; Python plotting requires `pyrite-mc[plots]`.
A checkout's `uv sync` includes both.

## Checkpoint layout and identity

Canonical full runs use `pyrite-output/checkpoints/<material>/`. Other profiles
and modified parameters use identity-qualified dataset stems. Each dataset
records its resolved inputs and hash in component metadata; incompatible inputs
do not silently resume together. List active datasets and archive labels with:

```bash
pyrite checkpoint list
pyrite checkpoint archive list
```

The analysis app offers active datasets under the effective checkpoint root.
See [profile identity](../repo-design/storage/dataset-identity-and-storage.md) and
[checkpoint persistence](../repo-design/storage/checkpoint-case-store.md).
The canonical Python loading entry points are `pyrite.storage.load_checkpoint`
and `checkpoint_manifest`; they are **advanced-stable** APIs.
`pyrite.results.records` flattens a loaded store, and `select_results` selects
records by case fields before plotting a larger sweep.

## Stored components

| File | Content |
| --- | --- |
| `line.h5` | PXR/CBS line components, excluding atomic and channeling radiation |
| `brem.h5` | Bremsstrahlung continuum |
| `characteristic.h5` | `spec_characteristic` on the fine line grid |
| `channeling.h5` | Optional `spec_channeling` on the fine line grid |

Loaded records keep `spec`, optional `spec_coherent`, `spec_characteristic`,
optional `spec_channeling` and `brem` separately; no component includes another.
The coherent and incoherent arrays are alternative PXR/CBS modes. Characteristic
and channeling terms enter a selected total once each. The analysis app shows
characteristic radiation by default; its **Components** selector controls
additive terms independently. A dataset without `characteristic.h5` loads
without that contribution. Channeling-off datasets omit `channeling.h5`.

Raw `spec` and `brem` densities have units **photons/(incident electron · eV · sr)**.
`E_grid` is the line grid; `brem` is interpolated to it, while `brem_wide` uses
`E_grid_brem`. Stored `scale` converts to photons/(s · nA · eV) using acceptance;
`source_current_na` provides the current. A detector response is a separate
operation. Preserve those distinctions when comparing a raw component plot,
a source-flux plot and an acquired pixel image.

For in-memory `simulate` results, use `Result.combined_cxr()` and the
[scene-result recipes](python-scene-recipes.md#interpret-the-result).
`simulate` does not write campaign checkpoints. The [result schema](../repo-design/storage/result-schema.md)
owns field definitions and legacy-read compatibility.

Planar channeling sources include bound lines and tracked free-to-bound capture.
They retain the diagnostic model's constant mean electron energy and laterally
infinite photon-attenuation slab. Measured linewidth reproduction remains
incomplete. See [planar runs](../guides/experimental-planar-channeling.md#add-planar-channeling-to-a-transport-run)
and [refused paths](quantum-api.md#supported-and-refused-paths) for selection, photon-grid requirements, replay and limitations.

## Analysis and comparison apps

```bash
pyrite app analysis launch hopg
pyrite app compare launch hopg
```

The analysis sidebar selects **Material**, **Checkpoint**, **Emission**, beam
energy, polar tilt, azimuth and thickness. The checkpoint picker labels profile
and face, for example `standard · flat` or `standard · blazed`. It opens the
standard flat run when available; otherwise choose a checkpoint explicitly.

| Analysis view | Shared-slice behavior |
| --- | --- |
| **Spectra** | Overlay narrowband and broadband spectra; **Vary** changes energy, tilt or azimuth and pins the other values. |
| **Map** | Azimuth × tilt heatmap at fixed energy/thickness; select a cell for its spectrum, trends and top-20 ranking. A 1-D sweep shows scan plots. |
| **Detectors** | Eagle XO or Timepix3 response at fixed tilt/azimuth/thickness, varying beam energy. |

When a pinned thickness was not reached at a lower energy, the view identifies
its fallback to that energy's thickest computed slab. For static scientific
figures, `pyrite.plots` provides spectral, sweep and detector-response helpers.

The compare app's case basket persists while the material/checkpoint picker
changes. Its cross-material view reads every available catalog checkpoint,
independently of that picker. `pyrite app analysis export`, `pixels export` and
`compare export` produce static HTML. Accepted arguments and output paths are
in the [CLI reference](../repo-design/cli/cli-reference.md).

## Pixel-detector observations

Profiles with a detector acquisition store one counting observation per case
under `pyrite-output/observations/<stem>/`, beside the matching checkpoint.
Multiple detectors have separate ID-qualified stems. Factors hold per-tile
spectra, pixel solid angles and filter paths, rather than a pixel × energy cube.
Reopening never reruns transport. `pyrite.observations.observation_inventory(stem)`
lists observations by case without loading their factors. See [physical
detector construction](python-scene-recipes.md#place-a-finite-filter-over-detector-pixels).

```bash
pyrite app pixels launch hopg
```

Choose a dataset, observation and image: total registered counts, counts within
configured reporting-bin edges, primary filter transmission at a stored
continuum energy, or filter coverage along pixel-centre rays. **Counts** defaults
to deterministic expectations; a Poisson acquisition also offers its labelled,
seeded realization.

Images up to 128 × 128 display at full resolution. Larger images aggregate
counts by sum, transmission by mean and coverage by maximum; tooltips report
pixel ranges. Row/column selectors show one pixel's position, direction, solid
angle, tile and filter paths, true accepted spectrum, and measured reporting-bin
histogram with underflow/overflow/below-cut accounting.

Timepix3 counts represent clustered photons assigned to the incident-ray pixel;
raw neighbouring-pixel triggers are outside the model. Its first response-matrix
build can take tens of seconds. Static export fixes the centre-pixel selection;
interactive selection needs `launch`. Remote observations remain on the remote
host until observation transfer is implemented.

## Channeling app

```bash
pyrite app channeling launch
pyrite app channeling launch --smoke
pyrite app channeling export --stem channeling
```

This experimental app solves independently of campaign checkpoints. Choose Si
or diamond (110), energy, incidence, divergence and zone depth inside the app;
there is no material argument. Export runs its default case headlessly to
`pyrite-output/results/channeling.html`. Selectors require a live app.
See [CLI options](../repo-design/cli/cli-reference.md) for launch and tunnel controls.

The eigenstate selector shows one Bloch-momentum-zero eigenfunction with unit
probability per period; its largest plane-wave coefficient is real and positive.
The spatial heatmap shows ensemble probability density, including within-block
coherences, rather than a unique ensemble wavefunction. Folding into one period
retains surviving incident-electron weight. The conditional view divides by
that weight and shows zero when none survives; survival is reported separately.
Display controls reuse solver output and use each depth's current basis. See
[spatial reconstruction](../validation/channeling/cr-planar-spatial-density-reconstruction.md).

The app also offers an [experimental axial mode](../guides/experimental-axial-channeling.md)
with an explicit Run action, 2-D maps, coherent phase and optional string-centred
OAM. [Quantum API contracts](quantum-api.md) cover shaped entry states.
The [planar entrance-state reference](quantum-api.md#specify-and-inspect-quantum-entrance-states)
distinguishes runtime diagnostic factories from the restricted experimental
`Beam(state=...)` source admitted in production scenes.

## Archive and reduction commands

Use checkpoint commands rather than editing dataset directories:

```bash
pyrite checkpoint --help
pyrite checkpoint archive --help
pyrite checkpoint slim --help
```

Archive supports store/list/restore/merge; slim reduces transfer size. `gc`
removes cases outside current profile resolution; `rm` targets selected datasets.
Review their preview and confirmation options before deletion. Remote checkpoint
submission and transfer are covered in [Running on a cluster](../guides/running-on-a-cluster.md).
