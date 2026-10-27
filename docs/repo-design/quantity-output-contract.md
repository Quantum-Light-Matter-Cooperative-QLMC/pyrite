# Quantity-output conversion contract (#755)

**Status: proposed for review; no quantity-valued result accessor has been
introduced.** This is the gate before implementation of result views. It
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

No spelling is approved yet for a typed output view, its `to(...)` method, or
any semantic result property. The first API review must choose between small
family-specific accessors and a typed metadata adapter that can express
(node/bin, physical measure, normalization, support mask) without requiring a
per-pixel-energy materialization. Existing Polars/CSV schema fields retain
their unit-suffixed numeric names unless a separately versioned output format
is authorized.

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

**Open review questions:** exact output accessor spelling; typed carrier and
mask/uncertainty model; whether a quantity view may wrap non-CPU arrays (if
not, refuse explicitly); explicit dimensional model for optical coherency;
and preservation of normalized finite-domain quantum densities. Resolve these
*before* adding a generalized output wrapper.
