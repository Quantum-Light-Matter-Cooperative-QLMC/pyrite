# Results and analysis

Inspect computed spectra, provenance, acquired counts and analysis reductions. Historical root aliases remain supported while canonical ownership moves into `pyrite.results`.

The entries below use generated docstrings, signatures, and source links. API support and scientific validation are separate contracts.

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :nosignatures:

   pyrite.PixelMetadata
   pyrite.PixelRayMap
   pyrite.SpectralFactors
   pyrite.SpatialResult
   pyrite.Analysis
   pyrite.Result
   pyrite.Provenance
   pyrite.CxrSpectra
   pyrite.SpectralDensity
   pyrite.results.QuantifiedSpectrum
   pyrite.results.QuantifiedDirectionalSpectrum
   pyrite.results.QuantifiedApertureSpectrum
   pyrite.results.QuantifiedSpatialSpectrum
   pyrite.results.QuantifiedSpatialImage
   pyrite.results.QuantifiedAcquisitionBatch
   pyrite.results.Settings
   pyrite.results.records
   pyrite.results.filter_results
   pyrite.results.select_results
   pyrite.results.best_azimuth
   pyrite.results.cxr_metrics
   pyrite.results.radiation.radiation_band_metrics
   pyrite.results.summary_table
   pyrite.results.directional.DirectionalSpectrum
```

## Result analysis

Single simulations return `Result`; resumable campaigns store result mappings
in [component checkpoints](../../repo-design/storage/dataset-identity-and-storage.md).
`Settings` is the compatibility surface for checkpoint analysis. `Analysis`
holds presentation controls and legacy conversions; `simulate` does not
consume it.

## Read pixel geometry with units

`spatial.pixel_metadata(pixels=[(row, column), ...])` returns
`pyrite.results.PixelMetadata` in selection order. Its `local_position`,
`center`, `distance` and `path_length` properties return quantities in mm;
`polar` and `azimuth` return degrees, and `solid_angle` returns sr. For example:

```python
assert result.spatial is not None  # Result from a planar-detector scene.
metadata = result.spatial.pixel_metadata(pixels=[(0, 0)])
position_cm = metadata.center.to("cm")
polar_rad = metadata.polar.to("radian")
acceptance = metadata.solid_angle
```

Each property returns a fresh array through the shared Pint registry. Shapes,
lab/detector frames and filter order match the numeric fields documented on
`PixelMetadata`. Unit conversion does not change acceptance or photon scoring.
Directions, pixel coordinates and tile indices remain numeric; angle conversion
does not create a density Jacobian. Existing unit-suffixed fields and stored
observations remain numeric. Pint loads on quantity access only.

## Convert spectral display units

`SpectralDensity.quantify()` and the experimental
`DirectionalSpectrum.quantify()` return independent quantity views. Call
`.to_energy("keV")` to convert energy nodes and their density together:
per-keV density is 1000 times per-eV density, preserving the integrated yield.
The numeric source fields and legacy tables keep their existing units.

Directional views retain `(angle, photon_energy)` axis order, component identity,
and a copied `supported` mask. `view.angle.to("radian")` converts angular point
coordinates; density remains per steradian. Angles are not solid-angle bins.
These views perform no current, exposure or acceptance normalization and retain
unsupported NaNs. Array read-back returns fresh copies; case and provenance
remain on the numeric source. Reconstruct views after loading numeric results.
See the [output contract](../../repo-design/quantity-output-contract.md) for
measure and normalization details.

## Binned photon-aperture histograms

`PhotonContinuumTally.quantify()` and `PhotonEventTally.quantify()` return an
experimental `pyrite.results.QuantifiedApertureSpectrum`. This is a distinct
**bin-mean** histogram, not a node-sampled spectrum. Its `photon_energy_edges`
contain N+1 boundaries; `bin_widths`, `density`, `direct_density` and
`redistributed_density` have N elements. Convert with `.to_energy("keV")`:
energy edges and widths shrink by 1000 and densities grow by 1000, but
`counts` (the in-range integral per bin) are invariant.

`detected_counts` includes `unbinned_counts` (out-of-range events), which are
never redistributed into reported bins. Counts are dimensionless Pint values
with explicit `count_basis="photons_per_incident_electron"` metadata, **not**
electric charge or a beam-current rate. The view is already
`normalization="aperture-integrated"` and has no per-steradian measure; it does
not reapply detector acceptance. The numeric tally and the existing
`Result.aperture_dataframe(current_nA=...)` are unchanged, as are checkpoint
identities and optional event relaxation-energy budgets.

## Selected pixel spectra

`SpatialResult.quantify_spectra(pixels=[(row, column), ...])` (or `region=...`)
returns an experimental `QuantifiedSpatialSpectrum` with native energy nodes,
`(pixel, photon_energy)` density and an ordered `pixel_coordinates` array.
Unlike source and directional spectra, each pixel spectrum has **already been
multiplied by pixel solid angle** and filter transmission. Its dimensional
measure is therefore `1/energy`, **not** `1/energy/sr`; the semantic metadata
is `normalization="pixel-accepted"` and `count_basis="photons_per_incident_electron"`.

```python
view = result.spatial.quantify_spectra(pixels=[(0, 0)], component="cxr")
converted = view.to_energy("keV")
# Both view.density and converted.density are per incident electron and pixel.
# The per-keV density is 1000 times the corresponding per-eV density.
```

`measured=True` applies the detector response exactly once, when selecting
spectra; `.to_energy()` only changes presentation units. Both the original
numeric spectrum and detached quantity view remain independent. This method
materializes **only the requested pixels**, never a full detector × energy cube.
Integrated `SpatialResult.image()` yields remain numeric and distinct from
registered acquisition counts; both now offer independent quantity views.

## Integrated pixel images

`SpatialResult.quantify_image((low_eV, high_eV), component="cxr")` performs the
same window integration as numeric `image()` and returns a detached
`QuantifiedSpatialImage`. Its `photon_yield` is a `(row, column)` Pint
**dimensionless** array, with `count_basis="photons_per_incident_electron"` and
`normalization="pixel-accepted"`. The per-pixel solid angle and transmission
have already been applied; there is no remaining per-eV or per-sr density.

`photon_energy_window` gives the requested inclusive node-selection bounds,
not interpolated bin edges. Calling
`.to_energy("keV")` converts **only** these bounds: integrated pixel yields,
response application and beam normalization do not change. `measured=True`
applies the existing detector response exactly once during numeric scoring.
This is a **per-primary photon yield** image, not an exposure-normalized
registered-count image. The legacy `image()` array remains unchanged.

## Acquired pixel counts

`AcquisitionBatch.quantify()` returns a detached experimental
`QuantifiedAcquisitionBatch` view with measured-energy **bin edges** and
counts over an **acquisition exposure**, not per-primary yields, rates or
per-energy densities. `expected_counts` and optional `realized_counts` remain
separate; `counts` selects already-realized values if present. `total_counts`
sums reporting bins only. Underflow, overflow and below-threshold channels
remain separate per pixel.

```python
view = result.acquire(pixels=[(0, 0)]).quantify()
kev = view.to_energy("keV")
# Energy edges and widths convert; registered counts remain identical.
```

The snapshot preserves pixel selection, source components, observation digest
and original stochastic draws. Display conversion neither scores nor redraws
photons. The numeric acquisition and persistence schemas remain unchanged.
See the [output contract](../../repo-design/quantity-output-contract.md).

## Pixel count result alias

`PixelCountResult` is the advanced-stable alias for `AcquisitionBatch` returned
by pixel acquisition. Measurement requests remain under
[`pyrite.observations`](observations.md).

```{eval-rst}
.. autosummary::
   :toctree: ../../_autosummary
   :template: autosummary/public-alias.rst
   :nosignatures:

   pyrite.results.PixelCountResult
```
