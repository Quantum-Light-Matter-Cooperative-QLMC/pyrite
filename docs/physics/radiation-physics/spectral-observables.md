# Spectral observables and conventions

Both radiation kernels return the same physical quantity on the same grid, so
this page collects the conventions that govern how their output is normalized,
combined, and read — and where the source physics stops and the instrument model
begins.

## Units and normalization

The primary quantity is the doubly differential yield per incident electron,

$$
\frac{d^2N}{dE\,d\Omega}
\quad\bigl[\text{photons}\,\text{eV}^{-1}\,\text{sr}^{-1}\,\text{electron}^{-1}\bigr],
$$

evaluated on an energy grid in eV. Internally lengths, times, and inverse
frequencies are all carried in Ångström with $c=1$; energies convert through
$E=\hbar c\,\omega$ with $\hbar c$ in eV·Å.

Normalization is per **incident** electron, not per emitted photon and not per
transported segment: the kernels sum every contribution and divide by the
electron count at the end. That makes the result independent of Monte Carlo
statistics in the mean, and it is why doubling the electron count must leave the
spectrum unchanged within Monte Carlo error — one of the ledgered invariance
checks.

Line and continuum spectra are transported from the same trajectories but may use
different electron populations (`ne-line` and `ne-brem` in a profile). Each kernel
takes the first $N_e$ electrons of the shared transport and normalizes by its own
$N_e$, so the two spectra remain directly addable despite different statistics.

Segments are Monte Carlo histories, not detector events. Nothing in either kernel
is a count rate: converting to counts requires beam current, live time, and the
detector solid angle, all of which live downstream.

## Spectral window

Segments whose resonance energy falls outside the requested grid are dropped,
with a 20% pad on both ends so that sinc tails reaching into the window still
contribute, and with a hard 10 eV lower floor below which no line is evaluated.
The coupling tabulation grid is padded to match. Grid choice is therefore not
purely cosmetic for the line kernel — a window that clips a resonance removes its
tail contribution too.

For the continuum the grid interacts with physics more strongly still: the
infrared rise means the integrated background depends on the low-energy bound,
and the attenuation tables' range determines where $\mu$ is treated as zero. See
[Bremsstrahlung](bremsstrahlung.md).

What a grid entry means — an evaluation node or a bin edge — and how a
photons/eV density is integrated on it are pinned separately in
[Energy-grid semantics](energy-grid-semantics.md).

## Component splits

`components=True` returns the total alongside PXR-only and CBS-only spectra,
computed as $|\chi_{\mathbf g}|^2 f_{\rm pxr}^2$ and
$|eU_{\mathbf g}/m|^2 f_{\rm cbs}^2$.

This is **not an additive decomposition**. The total contains the interference
term $2\,\mathrm{Re}(A_{\rm PXR}A_{\rm CBS}^{*})$, so `spec_pxr + spec_cbs`
differs from `spec` — by +22% at the point checked in the ledger. Treat the two
components as the diagonal contributions they are, and never present their sum as
the spectrum.

The split is refused entirely where it would be ambiguous: under the coherent
policy, and for numerically substepped flights, because the cross term survives
the phased sum in both cases.

## Solid-angle integration

The default evaluation uses a single fixed far-field direction, and the caller
multiplies by a solid angle downstream. `mc_spectrum_solid_angle` instead tiles
the detector face into directions carrying solid-angle weights and sums the
per-direction spectra, returning $dN/dE$ **already multiplied by $\Omega$**.

Because both the resonance energy and the amplitudes depend on the observation
direction, this yields the true, generally asymmetric integrated lineshape and
the across-face intensity gradient, rather than a flat-$\Omega$ approximation
plus an analytic aperture width. Consequently:

- do not multiply the result by the solid angle again;
- drop the analytic aperture-broadening term from the detector convolution, but
  keep the configured detector energy-resolution term;
- a one-direction grid reproduces `spec * Omega` exactly, which is the regression
  anchor;
- the blazed-groove escape is only compatible with a single direction, since
  tiling breaks the relief-facet parallelism the groove geometry assumes.

See [Detector solid angle](../detectors/detector-solid-angle.md).

## External backgrounds

An externally computed continuum can be interpolated onto the same grid and
scale-fitted against sidebands for subtraction. External spectra are treated as
**as-detected**: window efficiency and detector resolution are not re-applied, so
they must not be pushed back through the instrument model. Provenance
requirements for such fixtures — dataset, version, condition, source file,
checksum — are in [external bremsstrahlung
comparison](../../validation/literature-ref/external-bremsstrahlung-comparison.md).

## Read-only error estimators

Two opt-in estimators quantify the error of the one-point-per-row path integral.
Neither is on a default call path and neither changes a spectrum.

- `cxr_endpoint_resonance_drift` evaluates each flight's line energy at both
  endpoint speeds and reports the sweep **in units of that flight's sinc
  linewidth**, $W=2\pi\hbar c/[(1-\hat{\mathbf n}\cdot\mathbf v)t_L]$, as
  percentile summaries over flights. A drift small compared with the linewidth is
  the condition under which freezing the velocity across a flight is safe.
- `brem_endpoint_quadrature_error` reports the difference between the
  midpoint-rule and endpoint evaluations of the continuum integrand.

Both warn above calibrated thresholds documented with the
`radiation-error-estimators` ledger row.

## Where source physics stops

Everything on these pages is **source** physics: what the sample emits toward a
direction. Detector quantum efficiency, window and filter attenuation, energy
resolution, charge diffusion, and pixel geometry are downstream forward models
and must not be folded into the kernels — see [Detector
response](../detectors/detector-response.md). Mosaic broadening is the one place
where the boundary is a genuine choice: the exact Monte Carlo mosaic route lives
in the source kernel, the analytic energy-shift approximation is applied at
detector convolution, and applying both double counts.

## Validation

The normalization, summation order, and per-electron convention are covered by
the `coherent-line-spectrum` row; the component split's non-additivity is
recorded there too. The estimators are `radiation-error-estimators`. Absolute
normalization against an analytic single-segment comparison is the **open
`closed-form-flux` discrepancy**, so absolute flux comparisons against literature
are not yet settled — relative comparisons and ratios are on firmer ground.
Consult the [validation
ledger](../../validation/physics-validation-ledger.md) before scientific use.
