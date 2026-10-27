# Quantity-output conversion contract (#755)

**Status: node-sampled `SpectralDensity.quantify()`, experimental
`DirectionalSpectrum.quantify()`, experimental binned aperture tallies,
selected-pixel `SpatialResult.quantify_spectra()`, integrated spatial-image
and acquisition count views are implemented; other families remain proposed
for review.** This is the
contract for incremental result views. It
builds on the [physical-quantity migration inventory](quantity-migration-inventory.md)
and [existing numeric result/storage contracts](result-storage-contracts.md).

## Preserve value, measure and context separately

Every view must declare all of the following before any conversion:

| Item | Required description | Example |
| --- | --- | --- |
| Axis | name, unit, ordered nodes **or** bin edges, shape | `photon_energy`, eV, `node`, `(N,)` |
| Value | dimension/scale, array shape, dtype and source | `density`, reciprocal energy/sr, `(N,)` |
| Measure | density with respect to which integration variables | `dE dΩ`, **not** merely `dE` |
| Normalization | named physical denominator that is not inferred from Pint dimensionality | `per incident electron`, `per second`, `accepted detector` |
| Support | masks, out-of-support values, uncertainty if present | directional grazing `supported=False`, density NaN |
| Ownership | copy/read-only semantics, host/device placement, no hidden work | read-only copy/view and no simulation |
| Provenance | identity/source grid/response and binning provenance | `source` vs `observation-average`; acquisition digest |

In particular, `photon` is a counted event and `incident electron` a
normalization **label**. Do not use Pint's electron elementary-charge unit to
represent an incident-particle denominator. A proposed quantity object may
carry the *dimensional* part (e.g. `1/eV/sr`) while the separate, mandatory
semantic metadata retains photon/event and incident-electron meanings. These
labels must survive export as machine-readable metadata.

## Mathematical conversion rules

### Node-sampled source density

If `x` is energy and `f_x = dN/dx`, a monotone change of energy coordinate
`y = y(x)` requires

$$
f_y(y) = f_x(x(y))\left|\frac{dx}{dy}\right|,\qquad
\int f_x\,dx = \int f_y\,dy.
$$

For energy in eV -> keV, `E_keV = E_eV / 1000`, so
`density_per_keV = 1000 * density_per_eV`. Uncertainties in the same density
scale by `1000`; variances by `1000**2`. Coordinate-only conversion must
never silently relabel a per-eV density as per-keV.

A constant node-sampled density `f_eV = 2` on `[1000, 2000] eV` integrates to
`2000` photons per named normalization; in `[1, 2] keV` its value must be
`2000` per keV and yield the **same** integral. A node conversion must not
introduce new sample positions or silently integrate extrapolated values.
Missing/suppressed components and NaN unsupported directions remain missing
or unsupported, not zeroed solely to make Pint conversion succeed.

### Binned histograms

For `N` energy bins, `edges` has length `N+1`, density and count each have
length `N`, `width_i = edges[i+1] - edges[i]`, and `count_i = density_i * width_i`.
Convert both edges and widths with the axis; transform the bin-mean density
by the reciprocal Jacobian. **Counts do not change** when only the displayed
energy unit changes. Never treat an edge histogram as N sampled nodes or
interpolate it to union-grid nodes. Missing/out-of-range/unbinned aperture
observations remain disjoint from reported in-range counts.

For example, edges `[0, 100, 300] eV` and density `[0.5, 1.5]` per eV yield
counts `[50, 300]`. After conversion to `[0, 0.1, 0.3] keV`, the density is
`[500, 1500]` per keV and counts remain `[50, 300]`.

### Coordinate angles and solid-angle measures

Changing `angle_deg` to radians changes the **coordinate**, not the density
measure of `DirectionalSpectrum`: its values remain per sr. Its angular
samples are directions in an angular plane, **not** bins of dΩ. A separately
proposed reparameterization of a *coordinate density* must declare its exact
Jacobian, e.g. `dΩ = sin(theta) dtheta dphi` on a spherical grid. This is not
an implicit Pint `degree -> radian` operation. The `supported` mask and NaNs
must be preserved. Physical aperture / pixel factors have *already* applied
acceptance, so there is no second `sr` multiplier.

### Contextual normalizations are not unit conversions

- `SpectralDensity.normalization == "source"` is response-free directional
  source density. `"observation-average"` is filtered, solid-angle-weighted
  detector average **still per sr**. Multiplying by accepted solid angle is
  a physical operation using the case or pixel ray map, not an axis conversion.
- `Result.spectral_flux(units="electron" | "sr" | "nA" | "detector")`
  uses named beam-current/acceptance scalings. The `units` argument is **not**
  a Pint energy-unit parameter. Its output currently has both eV and keV
  coordinate columns, while density columns remain per **eV** in all modes.
  Keep this legacy behavior unchanged; new quantity-view naming is deferred.
- `DirectionalSpectrum.rate()` uses electron rate from beam current (nA).
  A per-sr source is not an accepted-detector rate; do not assume an angular
  bin width or acceptance in this method.
- `SpatialResult.spectra()` already multiplies each per-sr intrinsic factor
  by its pixel solid angle and attenuation, hence yields per-eV accepted
  density. `average_density()` divides the sum by total pixel acceptance;
  `image()` integrates per-eV density over the requested node window.
- `Result.aperture_dataframe(current_nA=...)` integrates physical-aperture
  observations, so **do not** multiply by another aperture or solid angle.
  Current changes per-primary to per-second measures independently of eV/keV.
- `AcquisitionBatch` values are expected or realized **registered counts**
  after response and exposure. A changed energy-edge display unit must not
  scale counts, alter RNG state, draw new counts or rescore response.
- Axial quantum spatial density (Å⁻²), wavefunctions and current/divergence
  (Å⁻²/Å⁻³) use **different** measures and conditional/incident normalization;
  optical pulse J, V and W must never be interpreted as photon counts.

## Read-back and serialization constraints

Any future user-facing conversion accessor must be **opt-in**, must state its
axis/measure/normalization semantics, return a fresh copy or explicitly
immutable view, and avoid caching mutable Pint objects. Use the shared lazy
`pyrite.units.ureg` for all returned quantities, including inputs originally
constructed in external registries. No hidden simulation, spectrum resampling,
CPU/GPU copying or observation scoring. Source and persisted `float`/NumPy
fields, dataclass equality, CLI numeric defaults and all identity hashes stay
identical. Return unsupported input dimensions/normalizations as a clear
error, not a guessed conversion.

### First implemented slice: scalar component source spectra

`SpectralDensity.quantify()` returns an independent
`pyrite.results.QuantifiedSpectrum` on native energy nodes. It has Pint
`photon_energy` and `density` properties, an immutable conversion operation
`to_energy(unit)`, and explicit `sampling="nodes"`,
`measure="per_energy_per_steradian"`,
`count_basis="photons_per_incident_electron"`, and unchanged
`normalization` (`"source"` or `"observation-average"`).

```python
from pyrite.results import SpectralDensity

component = SpectralDensity([1000, 2000], [2.0, 2.0])
spectrum = component.quantify()
converted = spectrum.to_energy("keV")
assert list(converted.photon_energy.magnitude) == [1.0, 2.0]
assert list(converted.density.magnitude) == [2000.0, 2000.0]
assert spectrum.photon_energy.units != converted.photon_energy.units
assert converted.count_basis == "photons_per_incident_electron"
```

The energy-coordinate change scales density inversely. In a `QuantifiedSpectrum`,
**unit conversion of the spectral axis is coupled to density conversion**;
using Pint's ordinary `.to(...)` on an individual returned quantity is only
for calculations and cannot change the spectrum's paired state.
`photon_energy` and `density` are freshly constructed on property access, so
callers cannot mutate this spectrum's backing arrays through Pint magnitudes.
No source arrays, simulation, current scaling, or solid-angle integration
are altered. A non-energy `to_energy` request raises `ValueError`.

Do not confuse a quantity with its physical semantics: Pint's `steradian` is
dimensionless, so the explicit measure and count-basis labels remain required.
Other families (aperture histograms, spatial and
acquisition data, quantum and optical outputs) need their own reviewed contracts.
No generic result adapter is approved. Existing Polars/CSV fields retain their
unit-suffixed numeric names unless a versioned format is approved.

### Directional source view

`DirectionalSpectrum.quantify()` returns experimental
`pyrite.results.QuantifiedDirectionalSpectrum`. `axis_names` declares
`("angle", "photon_energy")`; `angle` returns degree quantities,
`photon_energy` returns eV quantities, and `density` returns per-eV/sr
quantities. `to_energy(unit)` couples energy and reciprocal density conversion
on each angular row. `supported` returns the original boolean angular mask;
NaNs remain unchanged. Component identity, node sampling, source normalization,
per-sr measure and photon/incident-electron count basis are explicit metadata.

Converting `view.angle.to("radian")` changes only the returned coordinate.
No angular Jacobian, sr-bin width, current, exposure or acceptance is inferred.
The view accepts CPU NumPy source arrays only and snapshots them. Every array
property returns a fresh copy; modifying the source or a returned array cannot
alter another view. Case and provenance remain on the numeric source rather
than being serialized into a registry-bound carrier. Reconstruct the view
from numeric results after loading. Nonstandard `DirectionalSpectrum.units`
are rejected explicitly rather than assumed to describe per-primary density.
No uncertainty array exists on this source type; none is invented by the view.

### Binned aperture histogram view

`PhotonContinuumTally.quantify()` and `PhotonEventTally.quantify()` return an
experimental, detached `pyrite.results.QuantifiedApertureSpectrum`.
`photon_energy_edges` and `bin_widths` are Pint energy quantities, with
N+1 and N values respectively. `density`, `direct_density` and
`redistributed_density` are bin means in reciprocal display energy;
`counts`, `direct_counts` and `redistributed_counts` are dimensionless Pint
quantities with N bin-integrated values. `to_energy(unit)` converts the
axis, widths and densities together, **not** counts.

`detected_counts == sum(counts) + unbinned_counts` within the stored tally's
validation tolerance. Unbinned detections are never assigned to energy bins.
Metadata includes `sampling="bin_means"`, `measure="per_energy"`,
`normalization="aperture-integrated"`, `count_basis="photons_per_incident_electron"`
and `source_component` (`continuum`, `characteristic` or `annihilation`). There
is no per-sr factor, rebinning, current normalization, exposure scaling or
response resimulation. Counts are Pint-dimensionless but semantically photons
per original primary, not electron charge. Event relaxation budgets and
persistence provenance are left on the numeric tally. Invalid source arrays
are rejected, CPU arrays are snapshotted and each property returns a fresh
quantity. Reconstruct after loading; no serialized schema changes.

### Selected-pixel spectral view

`SpatialResult.quantify_spectra(pixels=... | region=..., component=...,
measured=...)` returns an experimental, detached
`pyrite.results.QuantifiedSpatialSpectrum`, carrying ordered integer
`pixel_coordinates` and node-sampled `(pixel, photon_energy)` values. The
underlying bounded `spectra()` call applies per-pixel solid-angle acceptance
and primary filter transmission. Its spectral density has Pint units of
`1/energy`, **not** `1/energy/sr`, and retains the semantic measure
`"per_energy"`, normalization `"pixel-accepted"`, and count basis
`"photons_per_incident_electron"`. The `measured` flag records whether the
existing detector response was used. Energy conversion changes the energy
node and reciprocal density scale, but does not reapply filters/acceptance,
response, exposure or source-current scaling. The brem and line grids stay
separate. Pixel coordinates are index labels, not a physical length axis.

Selected CPU arrays are snapshotted and returned Pint quantities always own
fresh magnitudes. The view does not eagerly construct a full pixel-energy
cube, retain a detector or mutate the numeric result. Integrating over the
selected native nodes yields per-primary pixel photons; `SpatialResult.image()`
already provides window-integrated **numeric** photon yields. Pixel geometry
metadata and measured `AcquisitionBatch` **counts** require separate quantity
contracts; the view does not convert between source, accepted, and registered
normalizations.

### Integrated spatial-image view

`SpatialResult.quantify_image(energy_range_eV, ...)` returns a detached
`pyrite.results.QuantifiedSpatialImage` after calling the existing chunked
`SpatialResult.image()` calculation exactly once. Its `photon_yield` array has
`(pixel_row, pixel_column)` axes, `sampling="integrated_window"`,
`measure="per_pixel"`, `normalization="pixel-accepted"` and
`count_basis="photons_per_incident_electron"`. Integrating over the original
energy nodes has already removed the reciprocal-energy density; per-pixel
solid-angle acceptance has removed the reciprocal-steradian factor. Pint
represents these per-primary photon yields as dimensionless **but their count
basis is not detector counts**. Nothing in this view provides beam current or
acquisition exposure.

`photon_energy_window` is the requested pair of inclusive eV node-selection bounds,
convertible with `to_energy(unit)`. Converted limits do not resample the native
nodes or change the already-integrated yields. `measured=True` records the
response applied at scoring time; display conversion does not run response
again. Native result, pixel factors, hashes, serialized records and numeric
`SpatialResult.image()` are unaffected. Each view copies only the resulting
2D CPU array, not a detector × energy cube; all Pint read-back magnitudes are
fresh, isolated arrays. Unlike `AcquisitionBatch`, this has neither measured
energy bins nor event realization or exposure-normalized counts.

### Measured-energy acquisition counts

`AcquisitionBatch.quantify()` produces an independent experimental
`QuantifiedAcquisitionBatch` view, with `axis_names=("pixel",
"measured_photon_energy")`, `sampling="bin_counts"`, `measure="per_bin"`,
`normalization="acquisition_exposure"`, and
`count_basis="detector_events_per_exposure"`. The measured-energy axis has
N+1 **edges**; `expected_counts`, optional `realized_counts` and selected
`counts` have `(pixel, bin)` shapes. All count quantities are Pint-dimensionless
but semantically detector events over an exposure, never photon density,
per-incident-electron yield or current-normalized rate.

The underflow, overflow and below-cut channels stay separate from reporting
bins; `total_counts` sums only the latter. `to_energy()` converts measured
energy edges and widths together without changing any count, RNG draw or
observation digest. No re-scoring, response replay, resampling, hidden
beam/exposure normalization or schema change takes place. CPU array snapshots
and fresh Pint read-backs isolate all mutable state.

### Selected-pixel geometry read-back

`pyrite.results.PixelMetadata` exposes semantic quantity properties directly:
`local_position`, `center`, `distance` and `path_length` in mm; `polar` and
`azimuth` in degrees; `solid_angle` in sr. Each returns a fresh CPU array bound
to the shared registry. Pint `.to(...)` converts only that physical value;
there is no paired spectral density or normalization to transform. Shapes,
coordinate frames, selection order and filter order follow the existing
numeric fields. A zero-filter selection retains shape `(n_pixel, 0)`.

Directions, pixel coordinates and tile indices retain their numeric contracts.
In particular, pixel solid angle is acceptance metadata, not a per-sr photon
density. Reading or converting it neither applies acceptance nor rescores
geometry. Numeric dataclass fields and saved observations stay unchanged;
`dataclasses.replace` continues to accept the numeric field names. Quantity
properties are reconstructed on access after loading, with no registry state
in serialized records. Numeric metadata construction remains Pint-lazy.

## Required tests before the first output implementation

1. **Analytic node conversion:** eV/keV density reciprocal scaling;
   trapezoidal integral invariance on the same nodes; scalar and vector
   inputs; nonuniform grids; independent CXR and brem grids.
2. **Binned conversion:** edge/width/count invariance, no N-vs-(N+1)
   confusion, unbinned and below-threshold mass retained separately.
3. **Uncertainty/support:** density standard errors and variances scale
   correctly; directional support mask and NaN values are unchanged.
4. **Normalization refusals:** Pint energy conversion cannot turn per-primary
   into per-second, per-sr into aperture-integrated, or coherent conditional
   density into unconditional probability without explicit context.
5. **Legacy and ownership:** `Result.to_dataframe`, `spectral_flux`,
   `aperture_dataframe`, acquisition counts, old checkpoint/observation loads,
   hashes and manifests remain byte-/value-compatible; read-back mutations
   cannot alter their backing arrays.
6. **Packaging/runtime:** numeric construction and `import pyrite` remain
   Pint-lazy; no new source/worker/checkpoint Pint payloads; API import and
   formatting/docs checks pass.

**Open review questions:** typed carriers for additional output families and
mask/uncertainty models; whether a quantity view may wrap non-CPU arrays (if
not, refuse explicitly); explicit dimensional model for optical coherency;
and preservation of normalized finite-domain quantum densities. Resolve these
*before* adding a generalized output wrapper.
