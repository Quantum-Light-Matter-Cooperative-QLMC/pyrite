# Energy-grid semantics

Every spectral quantity in the simulation lives on a photon-energy coordinate,
and almost every numerical error that survives a physics review comes from
disagreeing about what that coordinate *is*. This page pins the conventions:
what a grid entry means, what a spectral value means, how the two combine into
an integral, and which of those combinations are forbidden.

Nothing here is a new physical model. It is the numerical contract the
[spectral observables](spectral-observables.md), the
[bremsstrahlung](bremsstrahlung.md) and
[characteristic](characteristic-radiation.md) kernels, and the
[detector response](../detectors/detector-response.md) all have to share.

The default grids remain uniform. Nonuniform support is incremental: sampled
continuum kernels evaluate arbitrary nodes, detector Poisson scoring uses local
midpoint-cell widths, and Timepix coarse-input rebinning conserves source-bin
mass. Energy-resolution convolution and the final Timepix recorded-density
projection remain uniform-grid work; see
[Energy grids across six decades](../../research/beam-transport/energy-grid-recommendations.md)
for the recommendation and the [uniform-only consumers](#uniform-only-consumers)
section below for what currently refuses them.

## Evaluation nodes are not bin edges

Two different objects are both spelled "an energy grid" in array form, and they
are not interchangeable.

```{list-table} The two energy-coordinate roles.
:name: tbl-grid-roles
:header-rows: 1

* - Role
  - Array of length
  - Meaning of entry $i$
  - Correct use
* - Evaluation nodes $E_i$
  - $N$
  - The spectral function is *sampled* at $E_i$
  - Sampled densities: PXR/CBS lines, bremsstrahlung continuum
* - Bin edges $\epsilon_i$
  - $N+1$
  - Bin $i$ spans $[\epsilon_i, \epsilon_{i+1})$
  - Histogram quantities: detector channels, counts, integrated line masses
```

A sampled spectral function carries no information about bin boundaries; a
histogram carries no information about where inside a bin its content sits.
Converting between them is an approximation, never a relabelling.

Where centres must be turned into edges, the conversion is explicit and uses
midpoints with one-sided outer half-widths:

```{math}
:label: eq-grid-midpoint-edges

\epsilon_i = \tfrac{1}{2}\left(E_{i-1} + E_i\right),
\qquad
\epsilon_0 = E_0 - \tfrac{1}{2}\left(E_1 - E_0\right),
\qquad
\epsilon_N = E_{N-1} + \tfrac{1}{2}\left(E_{N-1} - E_{N-2}\right).
```

{eq}`eq-grid-midpoint-edges` holds for nonuniform centres as written;
`montecarlo/spectrum/characteristic.py::_energy_bin_edges_and_widths`
implements it and is the reference for any new edge construction.

## A density in photons/eV stays a density in photons/eV

The primary quantity is a spectral density,
$\mathrm{d}^2N/\mathrm{d}E\,\mathrm{d}\Omega$, in
photons eV⁻¹ sr⁻¹ electron⁻¹. That is a property of the *physics*, not of the
sampling. Putting the samples on a logarithmic grid does not convert the
quantity to photons per decade, per $\ln E$, or per bin; it only changes where
the same density is evaluated.

Consequently a stored spectrum array is never self-describing enough to
integrate without its coordinate. Nothing may assume that adjacent samples are
separated by a constant width, and nothing may treat a density as if it were
already a per-bin photon count.

## Integrate in physical energy with local widths

The yield over a range is

```{math}
:label: eq-grid-physical-integral

N = \int \frac{\mathrm{d}N}{\mathrm{d}E}\,\mathrm{d}E
\;\approx\;
\sum_i w_i \, \frac{\mathrm{d}N}{\mathrm{d}E}\bigg|_{E_i},
\qquad
w_i = \tfrac{1}{2}\left(E_{i+1} - E_{i-1}\right)
```

with the one-sided half-widths $w_0 = \tfrac{1}{2}(E_1 - E_0)$ and
$w_{N-1} = \tfrac{1}{2}(E_{N-1} - E_{N-2})$ at the ends. Each node carries **its
own** local width. A single $\Delta E$ reused across the grid is a special case
of {eq}`eq-grid-physical-integral`, valid only when the grid is uniform.

### Log-energy integration needs the Jacobian

Sampling uniformly in $u = \ln E$ is a convenience for placing nodes. It does
not change the integrand. The change of variable carries a Jacobian:

```{math}
:label: eq-grid-log-jacobian

\int \frac{\mathrm{d}N}{\mathrm{d}E}\,\mathrm{d}E
=
\int \frac{\mathrm{d}N}{\mathrm{d}E}\, E \,\mathrm{d}u,
\qquad u = \ln E .
```

Summing a photons/eV density against a constant $\Delta u$ — the mistake a
uniform-looking log grid invites — omits the factor $E$ in
{eq}`eq-grid-log-jacobian` and misweights the spectrum by orders of magnitude
across a wide band. Either integrate in physical energy with
{eq}`eq-grid-physical-integral`, or integrate in $u$ with the explicit $E$
factor. Both are correct; mixing them is not.

### The trapezoid/line-mass ban

A node-centred trapezoidal integral of a sampled density and a sum of
bin-integrated line masses are **not** interchangeable conventions, and code
must never swap one for the other:

* The trapezoid sum of {eq}`eq-grid-physical-integral` approximates the area
  under a function that is assumed smooth between nodes. It is wrong by an
  arbitrary factor for a feature narrower than the node spacing.
* A bin-integrated line mass is the exact integral of a known profile over a
  bin — for instance the Lorentzian CDF difference in
  `characteristic.py::_lorentzian_bin_weights`. It conserves the transition
  yield regardless of how coarse the bins are, but it is a *per-bin mass*, not a
  density sample, and dividing it by a width to "make it a density" throws away
  exactly the truncation bookkeeping that made it exact.

Mixing the two in one sum double-counts or drops line yield. Any spectrum that
adds narrow lines to a smooth continuum must state, per component, which of the
two it is, and convert deliberately if they have to meet.

## A logarithmic grid cannot contain zero

$\ln 0$ is undefined, so a log grid has a strictly positive lower bound. That
bound is a physical choice, not a numerical one: it is the low-energy edge of
the modelled band, set by the data support of the attenuation and form-factor
tables and by where the model stops being meaningful. The
[bremsstrahlung](bremsstrahlung.md) infrared rise makes the choice observable —
the integrated background depends on it.

An arbitrarily small epsilon used only to keep the logarithm finite is therefore
not acceptable. It is an unexamined infrared cutoff wearing a numerical
disguise, and it silently sets a physics result.

Detector channels are a separate coordinate from the source mesh, and a real
instrument does have a channel that starts at zero recorded energy — the
Timepix output histogram begins at 0 eV so that charge-loss events below the
input energy are still scored. Represent that channel with an **explicit edge at
zero** in the detector's own edge array. Do not try to obtain it by lowering a
log source grid toward zero.

(uniform-only-consumers)=
## Uniform-only consumers

Consumers that still read the single spacing $E_1 - E_0$ and apply it across the
whole grid call
`pyrite.energy_grid.semantics.require_uniform_grid` and raise
`NonuniformEnergyGridError` rather than returning a silently wrong number:

* `detectors/response.py::convolve_detector` — a Gaussian whose $\sigma$ is
  expressed in samples is a fixed energy width only on a uniform grid. This is
  the energy-resolution path both `response.py` and `eaglexo_response.py` use.
`detectors/_si_sensor.py::poisson_core` converts density to bin mass with each
node's local midpoint-cell width. `TimepixResponse` uses the same explicit
source edges and overlap integrals when aggregating a nonuniform source mesh
onto its independent, uniform response-input channels. Both retain their old
scalar-width arithmetic on uniform grids for bit-for-bit compatibility.
The `sinc_cutoff` line window now searches the actual energy coordinates and is
also nonuniform-safe.

`results/metrics.py::line_metrics` is nonuniform-safe: it maps
`scipy.signal.peak_widths`' fractional sample crossings to energies by
interpolating `E` directly (`_sample_energy`, `_integrate_energy_window`), so
the dominant-line window and its integral are exact on any grid, not only a
uniform one.

Grid identity is a related contract: a cache key of (size, first node, last
node) does not identify a grid, because a linear and a logarithmic grid over the
same interval with the same node count share all three. Detector response caches
key on the complete coordinates through
`pyrite.energy_grid.semantics.grid_identity`.

The guard tolerates the rounding jitter a grid built by `numpy.linspace` or
`numpy.arange` unavoidably carries, including the case where the step is small
compared with the absolute energy (a 2.5 × 10⁻⁵ eV step at 1600 eV is only
~10⁸ float64 ulp wide). It is not a physics tolerance and must not be used as
one.
