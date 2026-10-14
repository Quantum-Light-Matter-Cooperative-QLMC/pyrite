# Detector and instrument response

Detector response is a downstream transformation of source spectra. PyRITE keeps source generation, geometric acceptance, absorption/efficiency, energy redistribution, and presentation distinct so that instrument effects can be enabled and audited independently.

## Response stages

1. The source kernels produce differential photon yield per incident electron, photon energy, and steradian.
2. Dataset scaling records detector solid angle and source cadence/charge without changing cached source arrays.
3. Quantum efficiency or optical throughput weights photons by true energy.
4. Energy redistribution maps true photon energy into measured energy through broadening and, for pixel detectors, charge diffusion/sharing.
5. Plotting and tabulation select units and normalization.

Applying solid angle twice is an error. The stored result scale already carries the configured acceptance; see [Detector solid angle](detector-solid-angle.md).

## Supported response families

The Timepix response uses silicon absorption and a cached response matrix with thresholding, statistical broadening, and charge-sharing behavior. The Eagle XO path shares silicon-sensor physics but normally models a CCD as solid angle times measured quantum efficiency; an optional low-occupancy mode adds Fano/read-noise energy resolution. The grazing-grating/ALEXS path combines optical mapping with silicon QE, diffusion, and resolution models.

`pyrite.detectors.Detector` is the read-time scoring boundary. It owns acceptance
and an optional response adapter. `Numerics.energy_bins` owns sampled true-photon
grids: fine, often narrow line samples and a separate wide continuum grid.
Ordinary runs resolve those source grids per case; explicit grids and stored
derivation artifacts remain authoritative. Acquisition reporting choices do
not size source grids or change transport identity.

Three energy axes serve different purposes:

- `Numerics.energy_bins`: numerical sampling of true source photon energies.
- Timepix `dE_mc` and `dE_out`: numerical input-channel and measured-output
  histogram widths of the response model. Separate axes keep response-matrix
  construction and repeated application affordable.
- `Acquisition.measured_edges_eV`: reporting intervals for counted events after
  response. `Acquisition.native(exposure_s=...)` requests automatic selection.

Timepix's histograms approximate a response to continuous recorded energies.
Neither these widths nor acquisition reporting bins implement a hardware
ADC/ToT quantization law.

Stored source arrays remain response-free. `Detector.score()` applies acceptance scaling and, when configured, one of `Timepix3`, `EagleXO`, or the compatibility `LegacyEDS` adapter. `response=None` returns the scaled source density. This single read path keeps response choices out of transport identity and permits the same transport result to be rescored without rewriting its checkpoint. Detector-model modules still contain experiment-specific placeholders; absolute predictions require calibrated hardware geometry and operating conditions.

A response matrix $R(E_m\mid E)$ acts on a true-energy spectrum as

```{math}
N_m(E_m)=\int R(E_m\mid E)\,\eta(E)\,N(E)\,dE,
```

where $\eta(E)$ is efficiency/throughput. Matrix columns must conserve the accepted probability appropriate to the selected detector model; any loss belongs in efficiency, not an unexplained normalization.

## Native measured bins and counting acquisition

`Detector.native_score()` returns event mass on the response model's explicit
half-open measured-energy bins instead of interpolating it back to the true
energy samples. `Timepix3` exposes the native columns of its existing response
matrix; `IdealPhotonCounter` uses unit efficiency and measured energy equal to
true energy. For the ideal response, adjacent true-energy sample midpoints are
cell edges and each endpoint cell uses its nearest half-spacing, clipped at
zero energy. Other response adapters fail explicitly until they define native
measured bins. Validation: `detector-timepix`.

Both Timepix scoring paths integrate the incident density over overlaps between
its midpoint cells and the fixed response-input channels. A source cell that
crosses a channel edge contributes to both channels; no mass is assigned solely
by its node energy. Density is assumed constant inside each source cell, and a
positive source floor has no flux beneath it. Native scoring preserves the
response's measured-bin mass; interpolating that output back onto source nodes
in `Detector.score()` can introduce a separate quadrature or truncation error.
See the [resampling verification](../../validation/detectors/detector-timepix.md).

`Acquisition.native(exposure_s=1.0)` selects the sorted, deduplicated union of
all enabled component native output edges, including characteristic lines and
the continuum. It preserves complete response bounds, including broadened tails
outside narrow source domains; ideal-counter bounds use source midpoint cells.
The union is independent of component evaluation order. More than 65,536 bins
raises an error directing the caller to explicit reporting edges; resolution
never silently coarsens. Unsupported native responses fail explicitly.

The native request is a separate `NativeAcquisition`, replaced by a concrete
`Acquisition` after all source axes are available and before hashing, counting,
or persistence. Stored acquisitions and batch results expose concrete edges;
`provenance["reporting_bins"]` records the requested policy and axis source.
Equal resolved acquisition values share the existing identity and coordinate
Poisson replay, regardless of policy. Policy metadata stays outside identity.
Explicit acquisitions retain their existing payload and numerical behavior.

Acquisition conservatively transfers each native-bin event mass to requested
reporting bins by interval overlap. Because the response has no information
within one native bin, this operation assumes uniform event density inside that
bin. A threshold crossing a native or reporting bin is handled by the same
fractional overlap. The result partitions response output into disjoint
registered, reporting-axis underflow, reporting-axis overflow, and in-range
below-cut channels; these channels sum to the native event mass apart from
roundoff.

For an accepted per-pixel density expressed per incident electron, the number
of physical source electrons in one exposure is

$$
N_e = t_{\mathrm{exp}} f_{\mathrm{rep}}
      \frac{Q_{\mathrm{bunch}}\,10^{-12}}{e},
$$

where exposure is in seconds, cadence in hertz, bunch charge in picocoulombs,
and `scipy.constants.elementary_charge` supplies the exact SI elementary
charge. Expected measured event mass per incident electron is multiplied by
$N_e$. Zero cadence or charge is the exact zero-count limit. This is source
normalization and detector bookkeeping, not an additional transport or
detector-physics approximation. Validation: `pixel-acquisition-counting`.

Expected counts are deterministic. Poisson mode draws every component and
pixel from a coordinate-keyed NumPy Philox stream identified by
`pyrite.coordinate-philox.v1`; the key includes observation digest, user seed,
row, column, and component. Selection order and pixel chunking therefore do
not change a pixel's result. Total and energy-window images sum their reporting
bin/component draws and never make an independent total draw. Exact replay
across a future RNG algorithm change requires retaining the algorithm version
or realized arrays.

## Assumptions and interpretation

- response parameters describe an idealized configured instrument, not an automatic calibration of a particular physical detector;
- convolution cannot create source photons and should be inspected after the unconvolved components;
- measured-energy binning and interpolation can affect narrow features;
- conservative reporting-bin overlap treats native-bin event density as uniform and cannot recover sub-bin structure;
- external background subtraction is analysis-only and is not detector response or source bremsstrahlung.

Validation claims include `detector-solid-angle`, `detector-line-broadening`, `alexs-qe-absorption`, `alexs-charge-diffusion`, `detector-eaglexo`, and `grazing-reflectivity`; consult the [validation ledger](../../validation/physics-validation-ledger.md) for current status. Implementation owners are `pyrite.detectors.spec`, `pyrite.detectors.*_response`, and the legacy analytic helpers in `pyrite.montecarlo.detector`.

## Multi-detector seam

Profiles support multiple detector IDs. Each detector observation receives
response-free line and continuum arrays on source grids resolved for its
acceptance. Counting observations persist factorized true spectra separately
from intrinsic checkpoints and resolve reporting edges over the complete
observation. Response and acquisition edits reuse true-spatial factors without
another transport pass. See [profile detectors](../../guides/sweep-profiles.md).
