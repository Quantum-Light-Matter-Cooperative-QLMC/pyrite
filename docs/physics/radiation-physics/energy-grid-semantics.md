# Energy-grid semantics

Source spectra use photon-energy coordinates in eV. Correct integration depends on whether each stored value is a sampled density, a bin-averaged density, or a bin mass. These conventions are shared by [spectral observables](spectral-observables.md), the radiation kernels, and [detector response](../detectors/detector-response.md).

Default grids are uniform. Continuum kernels also accept nonuniform nodes; Poisson scoring uses local cell widths, and Timepix input rebinning conserves source-bin mass. Energy-resolution convolution and final Timepix recorded-density projection require uniform grids. See [uniform-only consumers](#uniform-only-consumers).

## Evaluation nodes are not bin edges

Two different objects are both spelled "an energy grid" in array form, and they are not interchangeable.

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

A sampled spectral function carries no information about bin boundaries; a histogram carries no information about where inside a bin its content sits. Converting between them is an approximation, never a relabelling.

Where centres must be turned into edges, the conversion is explicit and uses midpoints with one-sided outer half-widths:

```{math}
:label: eq-grid-midpoint-edges

\epsilon_i = \tfrac{1}{2}\left(E_{i-1} + E_i\right),
\qquad
\epsilon_0 = E_0 - \tfrac{1}{2}\left(E_1 - E_0\right),
\qquad
\epsilon_N = E_{N-1} + \tfrac{1}{2}\left(E_{N-1} - E_{N-2}\right).
```

{eq}`eq-grid-midpoint-edges` holds for nonuniform centres as written; `montecarlo/spectrum/characteristic.py::_energy_bin_edges_and_widths` uses this construction with the low edge additionally clamped to 0 eV. Its bin widths, including that clamp, must also be used when integrating characteristic and bin-mean PXR/CBS spectra.

## Density units

The primary quantity is a spectral density, $\mathrm{d}^2N/\mathrm{d}E\,\mathrm{d}\Omega$, in photons eV⁻¹ sr⁻¹ electron⁻¹. Putting the samples on a logarithmic grid does not convert the quantity to photons per decade, per $\ln E$, or per bin; it only changes where the same density is evaluated.

A stored spectrum needs its coordinates and quadrature convention to be integrated. A density is not a per-bin photon count.

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

with the one-sided half-widths $w_0 = \tfrac{1}{2}(E_1 - E_0)$ and $w_{N-1} = \tfrac{1}{2}(E_{N-1} - E_{N-2})$ at the ends. Each node carries **its own** local width. A single $\Delta E$ reused across the grid is a special case of {eq}`eq-grid-physical-integral`, valid only when the grid is uniform.

### Log-energy integration needs the Jacobian

Sampling uniformly in $u = \ln E$ is a convenience for placing nodes. It does not change the integrand. The change of variable carries a Jacobian:

```{math}
:label: eq-grid-log-jacobian

\int \frac{\mathrm{d}N}{\mathrm{d}E}\,\mathrm{d}E
=
\int \frac{\mathrm{d}N}{\mathrm{d}E}\, E \,\mathrm{d}u,
\qquad u = \ln E .
```

Summing a photons/eV density against a constant $\Delta u$ — the mistake a uniform-looking log grid invites — omits the factor $E$ in {eq}`eq-grid-log-jacobian` and misweights the spectrum by orders of magnitude across a wide band. Either integrate in physical energy with {eq}`eq-grid-physical-integral`, or integrate in $u$ with the explicit $E$ factor. Both are correct; mixing them is not.

### Sampled densities and bin-integrated masses

A trapezoidal integral approximates the area under a function sampled at nodes. It can miss a line narrower than the node spacing.

An analytically integrated line profile instead gives the probability or yield inside each bin. Dividing that mass by the bin width produces a valid bin-averaged density, as the characteristic kernel does. Recover the mass by multiplying by the same width. Applying node-based trapezoid weights to that array changes the endpoint weights and need not preserve its yield.

Keep the quadrature convention with each component when combining a sampled continuum with bin-integrated lines.

(sinc-bin-mean-quadrature)=

### Bin-mean quadrature for PXR/CBS lines

The incoherent PXR/CBS line density is a sum of finite-time profiles $\operatorname{sinc}^2\!\left(a_j (E - E_{\mathrm{res},j})/\pi\right)$ with $a_j = (1 - \mathbf{v}\cdot\hat{\mathbf{n}})\, t_{L,j} / (2\hbar c)$ ([finite-time lineshape](../../validation/radiation-physics/finite-time-lineshape.md)). By default each profile is *sampled* at the nodes; a profile narrower than the node spacing then aliases and the trapezoid of {eq}`eq-grid-physical-integral` misses or double-counts its yield. The opt-in `line_quadrature="bin-mean"` instead writes, at each node, the profile's mean over that node's bin from {eq}`eq-grid-midpoint-edges`:

```{math}
:label: eq-grid-sinc-bin-mean

\bar S_{ij}
= \frac{\pi}{a_j\,(\epsilon_{i+1} - \epsilon_i)}
\left[F(x_{i+1,j}) - F(x_{i,j})\right],
\qquad
x_{i,j} = \frac{a_j\,(\epsilon_i - E_{\mathrm{res},j})}{\pi},
```

with the antiderivative of the normalized sinc squared

```{math}
:label: eq-grid-sinc-antiderivative

F(x) = \int_0^x \operatorname{sinc}^2 u\,\mathrm{d}u
= \frac{\operatorname{Si}(2\pi x)}{\pi} - x\operatorname{sinc}^2 x,
\qquad
F(\pm\infty) = \pm\tfrac12 .
```

Differentiating {eq}`eq-grid-sinc-antiderivative` returns $\sin(2\pi x)/(\pi x) - \sin(2\pi x)/(\pi x) + \sin^2(\pi x)/(\pi x)^2 = \operatorname{sinc}^2 x$; $x\operatorname{sinc}^2 x$ is the $\sin^2(\pi x)/(\pi^2 x)$ term with its removable $x = 0$ point made explicit. The result is a bin-averaged density in photons/eV. Consumers must preserve that interpretation; its yield is $\sum_i \bar S_i\,(\epsilon_{i+1} - \epsilon_i)$, which is the in-window line mass exactly, at any spacing and on nonuniform (windowed) grids alike.

Assumptions and scope:

* **Bin mean, not node value.** $\bar S_i$ is a bin average, so bin-mean spectra must be integrated with the bin widths of {eq}`eq-grid-midpoint-edges`. Those equal the trapezoid weights of {eq}`eq-grid-physical-integral` except in the two end bins, where the trapezoid takes half; a line within one bin of either end of the grid is misweighted by a trapezoid.
* **Truncation is reported, not folded back.** Mass beyond the outer edges is dropped, as for the characteristic Lorentzians; the dropped part is returned by `montecarlo/spectrum/lines/_bin_quadrature.py::sincsq_window_mass`, and captured plus truncated equals $\pi / a_j$. This is the bookkeeping the distinction between node values and bin means protects: a bin-mean array is a bin mass per width, the same object `characteristic.py` writes, and it must not be resampled as if it were a node value.
* **Yield only.** Averaging over a bin smooths the profile: peak height falls and apparent width grows once the bin is comparable to $\pi / a_j$. Peak heights and FWHM keep the node quadrature and its resolution policy.
* **Incoherent only.** The coherent route squares a sum of amplitudes, and the flight-grouped reduction (numerical substeps) adds substep amplitudes before squaring; neither has a per-line bin mass, and both refuse `bin-mean`, as does `sinc_cutoff` truncation.

Limiting cases: as $\epsilon_{i+1} - \epsilon_i \to 0$, $\bar S_i$ tends to the node sample of a uniform grid, whose node is the bin centre; as $a_j \to \infty$ all of $\pi / a_j$ lands in the bin containing $E_{\mathrm{res},j}$.

Numerics. $F$ is evaluated through its tail complement $R(x) = F(x) - \tfrac12\operatorname{sgn}^{+}(x)$ (with $\operatorname{sgn}^{+}(0) = +1$), which decays like $-1/(2\pi^2 x)$; a bin mass is $R(x_{i+1}) - R(x_i)$ plus one whole line when $x_i < 0 \le x_{i+1}$. $\operatorname{Si}(t) - \pi/2$ is computed directly, never by subtracting from $\pi/2$, and everything is FP64 until the finished bin mean is cast to the working precision, so narrow bins and far tails do not cancel in float32. $\operatorname{Si}$ has one host evaluator (`scipy.special.sici` below $t = 48$, the auxiliary-function asymptotic series above) and one C evaluator shared by the CuPy fallback and the fused CUDA reduction (power series, continued fraction of $E_1(it)$, the same asymptotic series); the C evaluator is within $7\times10^{-16}$ of 40-digit `mpmath` for $t \le 4$ and within $2.1\times10^{-15}/t$ beyond.

[Validation: `sinc-bin-integration`](../../validation/radiation-physics/sinc-bin-integration.md).

#### Quadrature selection

The line-grid policy's `quadrature` field accepts `node` or `bin-mean`. It resolves from the per-call value, then stored configuration, then the `node` default; there is no environment override. The case's `line_quadrature` key preserves the choice when recomputation pins explicit coordinates and drops the policy.

A `bin-mean` policy uses payload schema 3 and participates in case, checkpoint, and cache identity. Selecting it does not change the automatic `sinc-nyquist` resolution policy, which still targets node-sampled shape observables.

## A logarithmic grid cannot contain zero

$\ln 0$ is undefined, so a log grid has a strictly positive lower bound. That bound is a physical choice, not a numerical one: it is the low-energy edge of the modelled band, set by the data support of the attenuation and form-factor tables and by where the model stops being meaningful. The [bremsstrahlung](bremsstrahlung.md) infrared rise makes the choice observable — the integrated background depends on it.

A small positive epsilon would impose an arbitrary infrared cutoff. `_photon_continuum_floor.py::photon_continuum_floor_eV` instead uses the larger of a material-dependent plasma-energy scale and the data-support floor.

### Continuum floor

**The modelled band.** A continuum node is only meaningful where the medium's optics is. `materials/crystal.py::optical_constants` writes the medium as $n = 1 - \delta - i\beta$ with $\delta = (r_e \lambda^2 / 2\pi)\, n_a f_1$. In the high-frequency limit every electron responds freely, $f_1 \to Z$, and that expression is *identically* the free-electron result

```{math}
:label: eq-grid-plasma-floor

\delta(\omega) = \frac{\omega_p^2}{2\omega^2},
\qquad
\omega_p = \sqrt{\frac{n_e e^2}{\varepsilon_0 m_e}},
\qquad
n_e = \sum_i n_i Z_i ,
```

with $n_i$ the catalog number densities. The implementation uses $\hbar\omega_p$ as a material-dependent lower-bound scale, computed by `materials/attenuation.py::plasma_energy_eV`. Extrapolating the weak-refraction expression to $\omega=\omega_p$ gives $\delta=1/2$, outside its small-$\delta$ regime.

This is a free-electron estimate, not an exact optical threshold for each crystal. It assumes all $Z$ electrons respond freely in a homogeneous, isotropic, undamped bulk medium. Near the resulting scale, binding, collective response, and band structure limit that approximation.

*Limiting case.* $n_e \to 0$ gives $\hbar\omega_p \to 0$: an empty medium imposes no low-energy bound, and the floor falls back to pure table support.

**Data support.** The second bound is the lowest energy at which every table the continuum pipeline evaluates carries a real tabulated value rather than an extrapolation, measured from the packaged data and recorded with its provenance in `_photon_continuum_floor.py::DATA_SUPPORT_LIMITS_EV`: EEDL MF=26/MT=527 photon spectra reach 0.1 eV for every transport element, Chantler/FFAST reaches 1.01 eV (admitted on the strict interior, so 1.01 eV itself reads as out of range), and the digitized Eagle XO QE curve is documented valid from 12 eV. The QE table therefore binds, at 12 eV.

The plasma-energy term sets a material-dependent floor where it exceeds the data-support bound. Node placement above the floor is handled separately.

Uniform production bremsstrahlung grids apply the floor when material and requested band are resolved together in `campaign/sweep.py::build_cases`. If the declared start is too low, `floored_lattice_start_eV(material, step)` raises it to the first step multiple at or above the floor. A declared start above the floor is retained. A profile default of `0.0` means no additional lower bound beyond the material's floor.

The diagnostic band used to derive the upper endpoint, `energy_grid/derive.py::wide_brem_grid`, uses the same floor helper.

[Validation: `photon-continuum-floor`](../../validation/physics-validation-ledger.md).

Detector channels are a separate coordinate from the source mesh, and a real instrument does have a channel that starts at zero recorded energy — the Timepix output histogram begins at 0 eV so that charge-loss events below the input energy are still scored. Represent that channel with an **explicit edge at zero** in the detector's own edge array. Do not try to obtain it by lowering a log source grid toward zero.

`_grid_semantics.py::zero_based_detector_edges` builds exactly that, and keeps the two coordinates distinct once the source mesh carries a positive floor. Its first edge is always exactly 0 eV, reached one of two ways, because the two are physically different:

* the midpoint reflection of the first node lands **below** zero — there is no negative-energy half-bin, so the outer edge is clamped up to zero and the first node keeps one narrower bin;
* the reflection lands **above** zero, the ordinary case for a floored grid — the first node's bin is already correct and must **not** be widened, since stretching it down to zero would multiply that node's density by the extra width and invent photons. A *separate* explicit channel $[0, \epsilon_0)$ is prepended instead, carrying no source mass because the continuum model has no support below its floor. Callers prepend one zero to the density to stay aligned.

So a detector whose channel physically starts at 0 eV keeps its own boundary rather than inheriting the continuum's positive floor.

(continuum-node-refinement)=

### Continuum-node refinement

The floor fixes where a continuum grid *starts*. Where its nodes go between floor and ceiling is a second derived choice, and `energy_grid/refine.py::refined_continuum_grid` makes it.

**Geometric baseline.** Multiplying a density by its local midpoint width {eq}`eq-grid-midpoint-edges` approximates midpoint quadrature for sufficiently fine cells. Its error on an interval of width $h$ is $(h^3/24)\,|n''|$, so the *relative* error contributed by that interval is

```{math}
:label: eq-grid-quadrature-error

\varepsilon_i \simeq \frac{h_i^2}{24}\,\left|\frac{n''(E_i)}{n(E_i)}\right| .
```

On a fine grid uniform in $u = \ln E$, the local spacing is approximately $h_i = E_i\rho$ with $\rho = \ln(E_\mathrm{stop}/E_\mathrm{floor})/(N-1)$, and for a local power law $n \propto E^{-p}$ the combination $h_i^2\, n''/n$ is *independent of $E$*. A geometric grid therefore equidistributes {eq}`eq-grid-quadrature-error` across the smooth parts of the band. Edges and endpoints need separate treatment.

**Edges and endpoints.** {eq}`eq-grid-quadrature-error` assumes $n$ has a bounded second derivative across the interval. The modelled escaping continuum $n(E) = S(E)\,e^{-\mu(E)\ell}$ can have unresolved structure at absorption edges and kinematic endpoints. An unresolved jump produces a first-order quadrature error:

* **Absorption edges.** $\mu$ steps by a finite ratio across an interval far narrower than the local $E\rho$. The emitted $S$ is smooth there — bremsstrahlung has no feature at an absorber's edge — so the whole discontinuity sits in the escape factor.
* **Kinematic endpoints.** $S$ is identically zero above the highest instantaneous electron energy $E^\ast$, which for a run that only loses energy is the incident energy. At $E^\ast$ the density steps to zero: a jump of relative size 1.

**The rule.** Put a grid coordinate *on* the discontinuity, and refine no further than the resolution at which the model itself represents it.

* **Absorption edges.** The escape model attenuates with $\mu$ from the Chantler $f_2$ table, so the jump is found in that table — the steepest adjacent $f_2$ ratio near each xraydb edge energy (`montecarlo/spectrum/line_seeds.py::absorption_edge_brackets`, shared with the line-window seeds so one locator serves both axes). Both nodes of the located native bracket become exact grid coordinates, so the jump lies inside a single interval bounded by tabulated energies, and the surrounding native nodes are sampled at their own median spacing. Below that spacing the model carries no information, so refinement stops. Elements come from the medium's own catalog composition plus the detection path, which in this repository is the silicon sensor shared by both detector models.
* **Kinematic endpoints.** Placing a bin *edge* exactly at $E^\ast$ removes the first-order term outright, and with midpoint edges that is one node pair straddling $E^\ast$ at the grid's own local spacing: a local pair of nodes. The bin below then carries the tip and the bin above is exactly empty.

*Assumptions.* The medium is the absorber whose edges enter the escape factor (detector-path edges act through the response instead, not through $\ell$); the emitted $S$ is smooth across an absorber edge; the edge locator uses the native Chantler tabulation; and the modelled cutoff at $E^\ast$ is sharp — which it is in EEDL, where the photon spectrum is tabulated to $k = T$ and is zero above.

*Limiting case.* A band containing no located edge and no interior kinematic endpoint returns the geometric baseline **identically, node for node**. Adding structure adds nodes; adding none changes nothing. Equally, an edge whose jump falls outside $[E_\mathrm{floor}, E_\mathrm{stop}]$ is reported and dropped, never refused: it places no requirement on a grid over a band it is not in.

Refinement preserves the band endpoints and concentrates nodes around the located features. The derivation, assumptions, and checks are recorded in [Validation: `continuum-node-refinement`](../../validation/physics-validation-ledger.md).

(uniform-only-consumers)=

## Uniform-only consumers

Consumers that still read the single spacing $E_1 - E_0$ and apply it across the whole grid call `pyrite.energy_grid.semantics.require_uniform_grid` and raise `NonuniformEnergyGridError` rather than returning a silently wrong number:

* `detectors/response.py::convolve_detector` — a Gaussian whose $\sigma$ is expressed in samples is a fixed energy width only on a uniform grid. This is the energy-resolution path both `response.py` and `eaglexo_response.py` use.

`detectors/_si_sensor.py::poisson_core` converts density to bin mass with each node's local midpoint-cell width. `TimepixResponse` uses the same explicit source edges and overlap integrals when aggregating a nonuniform source mesh onto its independent, uniform response-input channels. Uniform grids use scalar-width arithmetic. The `sinc_cutoff` line window searches the actual energy coordinates and is also nonuniform-safe.

`results/metrics.py::line_metrics` is nonuniform-safe: it maps `scipy.signal.peak_widths`' fractional sample crossings to energies by interpolating `E` directly (`_sample_energy`, `_integrate_energy_window`), so the dominant-line window and its integral use the actual energy coordinates. This supports nonuniform grids without making unresolved features exact.

Grid identity is a related contract: a cache key of (size, first node, last node) does not identify a grid, because a linear and a logarithmic grid over the same interval with the same node count share all three. Detector response caches key on the complete coordinates through `pyrite.energy_grid.semantics.grid_identity`.

The guard tolerates the rounding jitter a grid built by `numpy.linspace` or `numpy.arange` unavoidably carries, including the case where the step is small compared with the absolute energy (a 2.5 × 10⁻⁵ eV step at 1600 eV is only ~10⁸ float64 ulp wide). It is not a physics tolerance and must not be used as one.
