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

(sinc-bin-mean-quadrature)=
### Bin-mean quadrature for PXR/CBS lines

The incoherent PXR/CBS line density is a sum of finite-time profiles
$\operatorname{sinc}^2\!\left(a_j (E - E_{\mathrm{res},j})/\pi\right)$ with
$a_j = (1 - \mathbf{v}\cdot\hat{\mathbf{n}})\, t_{L,j} / (2\hbar c)$
([finite-time lineshape](../../validation/radiation-physics/finite-time-lineshape.md)).
By default each profile is *sampled* at the nodes; a profile narrower than the
node spacing then aliases and the trapezoid of
{eq}`eq-grid-physical-integral` misses or double-counts its yield. The opt-in
`line_quadrature="bin-mean"` instead writes, at each node, the profile's mean
over that node's bin from {eq}`eq-grid-midpoint-edges`:

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

Differentiating {eq}`eq-grid-sinc-antiderivative` returns
$\sin(2\pi x)/(\pi x) - \sin(2\pi x)/(\pi x) + \sin^2(\pi x)/(\pi x)^2
= \operatorname{sinc}^2 x$; $x\operatorname{sinc}^2 x$ is the
$\sin^2(\pi x)/(\pi^2 x)$ term with its removable $x = 0$ point made explicit.
(A spelling with $\operatorname{Si}(2\pi x)/(2\pi)$ is wrong by a factor two
in that term.) The result stays a density in photons/eV, so every consumer of
the line array is unchanged; its yield is
$\sum_i \bar S_i\,(\epsilon_{i+1} - \epsilon_i)$, which is the in-window line
mass exactly, at any spacing and on nonuniform (windowed) grids alike.

Assumptions and scope:

* **Bin mean, not node value.** $\bar S_i$ is a bin average, so bin-mean
  spectra must be integrated with the bin widths of
  {eq}`eq-grid-midpoint-edges`. Those equal the trapezoid weights of
  {eq}`eq-grid-physical-integral` except in the two end bins, where the
  trapezoid takes half; a line within one bin of either end of the grid is
  misweighted by a trapezoid.
* **Truncation is reported, not folded back.** Mass beyond the outer edges is
  dropped, as for the characteristic Lorentzians; the dropped part is returned
  by `montecarlo/spectrum/lines/_bin_quadrature.py::sincsq_window_mass`, and
  captured plus truncated equals $\pi / a_j$. This is the bookkeeping the
  trapezoid/line-mass ban above protects: a bin-mean array is a bin mass per
  width, the same object `characteristic.py` writes, and it must not be
  resampled as if it were a node value.
* **Yield only.** Averaging over a bin smooths the profile: peak height falls
  and apparent width grows once the bin is comparable to $\pi / a_j$. Peak
  heights and FWHM keep the node quadrature and its resolution policy.
* **Incoherent only.** The coherent route squares a sum of amplitudes, and
  the flight-grouped reduction (numerical substeps) adds substep amplitudes
  before squaring; neither has a per-line bin mass, and both refuse
  `bin-mean`, as does the `sinc_cutoff` truncation (#117 owns the coherent
  route).

Limiting cases: as $\epsilon_{i+1} - \epsilon_i \to 0$, $\bar S_i$ tends to
the node sample of a uniform grid, whose node is the bin centre; as
$a_j \to \infty$ all of $\pi / a_j$ lands in the bin containing
$E_{\mathrm{res},j}$.

Numerics. $F$ is evaluated through its tail complement
$R(x) = F(x) - \tfrac12\operatorname{sgn}^{+}(x)$ (with
$\operatorname{sgn}^{+}(0) = +1$), which decays like $-1/(2\pi^2 x)$; a bin
mass is $R(x_{i+1}) - R(x_i)$ plus one whole line when
$x_i < 0 \le x_{i+1}$. $\operatorname{Si}(t) - \pi/2$ is computed directly,
never by subtracting from $\pi/2$, and everything is FP64 until the finished
bin mean is cast to the working precision, so narrow bins and far tails do not
cancel in float32. $\operatorname{Si}$ has one host evaluator
(`scipy.special.sici` below $t = 48$, the auxiliary-function asymptotic series
above) and one C evaluator shared by the CuPy fallback and the fused CUDA
reduction (power series, continued fraction of $E_1(it)$, the same asymptotic
series); the C evaluator is within $7\times10^{-16}$ of 40-digit `mpmath` for
$t \le 4$ and within $2.1\times10^{-15}/t$ beyond.

[Validation: `sinc-bin-integration`](../../validation/radiation-physics/sinc-bin-integration.md).

#### Why one explicit switch

The opt-in is one `quadrature` field of the line-grid policy
(`_line_grid_policy.py`, values `node`/`bin-mean`), mirrored onto the case as
the divergence-only `line_quadrature` key, rather than a per-observable-class
field. A run produces one line spectrum. A per-observable field would either
compute a spectrum per observable class or let the policy guess which class the
single spectrum serves; the caller already knows, and an explicit switch makes
that statement auditable in the case payload. Precedence is per-call, then
stored configuration, then the `node` default, with no environment layer, for
the same reason as `windows`: it is a structural choice, not a tolerance. A
`bin-mean` policy has payload schema 3 and records its source, so case,
checkpoint and cache identity move; a `node` policy keeps its historical payload
bit-for-bit. The case key exists because a recompute that pins explicit
coordinates drops the policy, and must not drop the quadrature the stored
spectrum was computed with. Selecting it does not change the automatic
`sinc-nyquist` resolution, which still refines for the node quadrature's shape
observables.

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

### The floor is derived, not chosen

`_photon_continuum_floor.py::photon_continuum_floor_eV` returns the larger of two
independently derived bounds, and neither is a tuning knob.

**The modelled band.** A continuum node is only meaningful where the medium's
optics is. `materials/crystal.py::optical_constants` writes the medium as
$n = 1 - \delta - i\beta$ with $\delta = (r_e \lambda^2 / 2\pi)\, n_a f_1$. In
the high-frequency limit every electron responds freely, $f_1 \to Z$, and that
expression is *identically* the free-electron result

```{math}
:label: eq-grid-plasma-floor

\delta(\omega) = \frac{\omega_p^2}{2\omega^2},
\qquad
\omega_p = \sqrt{\frac{n_e e^2}{\varepsilon_0 m_e}},
\qquad
n_e = \sum_i n_i Z_i ,
```

with $n_i$ the medium's own catalog number densities. At $\omega = \omega_p$,
{eq}`eq-grid-plasma-floor` gives $\delta = 1/2$: the weakly-refracting,
transparent-medium expansion that photon escape and self-absorption both rest on
has collapsed, and below $\omega_p$ the medium reflects rather than transmits. So
$\hbar\omega_p$ is where *this repository's own* optics stops being valid, not an
imported convention. `materials/attenuation.py::plasma_energy_eV` computes it.

*Assumptions.* All $Z$ electrons respond as free — exact only for $\omega$ well
above every binding energy; near and below $\omega_p$ the real response is
collective and band-structure dependent, which is the point. Homogeneous
isotropic bulk medium, so no surface, porosity, or anisotropy term. No Drude
damping, which shifts a real plasmon resonance by order $(1/\tau)/\omega_p$.

*Limiting case.* $n_e \to 0$ gives $\hbar\omega_p \to 0$: an empty medium imposes
no low-energy bound, and the floor falls back to pure table support.

*Cross-check.* The PDG/Sternheimer density-effect parameterization carries the
same quantity as $\bar{C} = 2\ln(I/\hbar\omega_p) + 1$. Inverting the packaged
$\bar{C}$ and mean excitation energy for silicon
(`materials/_transport_data.py`) gives 31.0482 eV against the 31.0498 eV
{eq}`eq-grid-plasma-floor` gives from the catalog number density — agreement to
$5\times10^{-5}$ relative, from different data through different code.

**Data support.** The second bound is the lowest energy at which every table the
continuum pipeline evaluates carries a real tabulated value rather than an
extrapolation, measured from the packaged data and recorded with its provenance
in `_photon_continuum_floor.py::DATA_SUPPORT_LIMITS_EV`: EEDL MF=26/MT=527 photon
spectra reach 0.1 eV for every transport element, Chantler/FFAST reaches 1.01 eV
(admitted on the strict interior, so 1.01 eV itself reads as out of range), and
the digitized Eagle XO QE curve is documented valid from 12 eV. The QE table
therefore binds, at 12 eV.

For every condensed medium in the catalog the plasma energy is the larger term —
the smallest is `sio2` at 30.201 eV, the largest `ptbi2` at 66.248 eV — so the
floor is a derived, material-specific number rather than a round one, and data
support binds only in the dilute limit above. HOPG's floor is 30.661 eV.

Choosing where the nodes go *between* floor and ceiling — refinement near
absorption edges and kinematic endpoints — is a separate concern from the
geometric baseline `geometric_continuum_grid` builds.

**Where the floor is applied, and why not in the catalog.** A uniform
production bremsstrahlung band (`E_grid_brem`) declares a `start`, but that
number is a *bandwidth request*, not the band the case gets: the floor belongs
to the medium, and the same catalog row is inherited by profiles that must
agree about the material they share. `campaign/sweep.py::build_cases` therefore
raises a declared start to `floored_lattice_start_eV(material, step)` — the
lowest multiple of the grid's own step at or above the medium's floor — the
first place the band and the material meet. Three consequences follow, and each
is load-bearing:

* **The nodes do not move, they are only dropped.** Snapping to the step
  lattice rather than starting at the floor itself leaves every surviving node
  on the coordinate it had, so a quantity measured on the grid — a cumulative
  coverage quantile, say — is read at the same energies as before.
* **A declared start *above* the floor is kept.** Narrowing the band is an
  ordinary bandwidth choice, exactly as in `geometric_continuum_grid`; only
  widening it downward, past model validity, is refused.
* **A profile-level default stores `0.0`.** Naming no medium, it can carry no
  floor of its own, and `0.0` reads as "no bound beyond the medium's". Storing
  a *per-profile* floor instead would make two profiles disagree about the same
  material's grid and silently stop sharing cases they are meant to share.

The diagnostic band the `stop` is measured on
(`energy_grid/derive.py::wide_brem_grid`) starts at the same energy, through the
same helper, so the band a bound is measured over is the band it is installed
for. On HOPG at 30 keV this drops the nodes at 0 and 25 eV; those carry about
$1.3\times10^{-3}$ of the total escaping intensity, and the derived coverage
energy and catalog `stop` are unchanged by their removal.

Detector channels are a separate coordinate from the source mesh, and a real
instrument does have a channel that starts at zero recorded energy — the
Timepix output histogram begins at 0 eV so that charge-loss events below the
input energy are still scored. Represent that channel with an **explicit edge at
zero** in the detector's own edge array. Do not try to obtain it by lowering a
log source grid toward zero.

`_grid_semantics.py::zero_based_detector_edges` builds exactly that, and keeps
the two coordinates distinct once the source mesh carries a positive floor. Its
first edge is always exactly 0 eV, reached one of two ways, because the two are
physically different:

* the midpoint reflection of the first node lands **below** zero — there is no
  negative-energy half-bin, so the outer edge is clamped up to zero and the first
  node keeps one narrower bin;
* the reflection lands **above** zero, the ordinary case for a floored grid — the
  first node's bin is already correct and must **not** be widened, since
  stretching it down to zero would multiply that node's density by the extra
  width and invent photons. A *separate* explicit channel $[0, \epsilon_0)$ is
  prepended instead, carrying no source mass because the continuum model has no
  support below its floor. Callers prepend one zero to the density to stay
  aligned.

So a detector whose channel physically starts at 0 eV keeps its own boundary
rather than inheriting the continuum's positive floor.

(continuum-node-refinement)=
### Where the nodes go: refinement is derived too

The floor fixes where a continuum grid *starts*. Where its nodes go between
floor and ceiling is a second derived choice, and
`energy_grid/refine.py::refined_continuum_grid` makes it.

**Why geometric is the right baseline.** Multiplying a density by its local
midpoint width {eq}`eq-grid-midpoint-edges` is a midpoint quadrature, whose
error on an interval of width $h$ is $(h^3/24)\,|n''|$, so the *relative* error
contributed by that interval is

```{math}
:label: eq-grid-quadrature-error

\varepsilon_i \simeq \frac{h_i^2}{24}\,\left|\frac{n''(E_i)}{n(E_i)}\right| .
```

On a grid uniform in $u = \ln E$ the step is $h_i = E_i\rho$ with
$\rho = \ln(E_\mathrm{stop}/E_\mathrm{floor})/(N-1)$, and for a local power law
$n \propto E^{-p}$ the combination $h_i^2\, n''/n$ is *independent of $E$*. A
geometric grid therefore equidistributes {eq}`eq-grid-quadrature-error` across
the band — which is exactly why it is the baseline, and exactly why it says
nothing about an integrand that is not smooth on the scale of its own step.

**Where that fails.** {eq}`eq-grid-quadrature-error` assumes $n$ has a bounded
second derivative across the interval. The modelled escaping continuum
$n(E) = S(E)\,e^{-\mu(E)\ell}$ violates that in exactly two places, and in both
the straddling interval's error degrades from $O(h^2)$ to $O(h)$ — a term no
globally finer geometric spacing removes at better than first order:

* **Absorption edges.** $\mu$ steps by a finite ratio across an interval far
  narrower than the local $E\rho$. The emitted $S$ is smooth there —
  bremsstrahlung has no feature at an absorber's edge — so the whole
  discontinuity sits in the escape factor.
* **Kinematic endpoints.** $S$ is identically zero above the highest
  instantaneous electron energy $E^\ast$, which for a run that only loses energy
  is the incident energy. At $E^\ast$ the density steps to zero: a jump of
  relative size 1.

**The rule.** Put a grid coordinate *on* the discontinuity, and refine no
further than the resolution at which the model itself represents it.

* *Edges* are **located, never listed.** The escape model attenuates with $\mu$
  from the Chantler $f_2$ table, so the jump is found in that table — the
  steepest adjacent $f_2$ ratio near each xraydb edge energy
  (`montecarlo/spectrum/line_seeds.py::absorption_edge_brackets`, shared with the
  line-window seeds so one locator serves both axes). Both nodes of the located
  native bracket become exact grid coordinates, so the jump lies inside a single
  interval bounded by tabulated energies, and the surrounding native nodes are
  sampled at their own median spacing. Below that spacing the model carries no
  information, so refinement stops. Elements come from the medium's own catalog
  composition plus the detection path, which in this repository is the silicon
  sensor shared by both detector models.
* *Endpoints* need **no taper at all.** Placing a bin *edge* exactly at $E^\ast$
  removes the first-order term outright, and with midpoint edges that is one
  node pair straddling $E^\ast$ at the grid's own local spacing: two nodes, no
  budget question. The bin below then carries the tip and the bin above is
  exactly empty.

*Assumptions.* The medium is the absorber whose edges enter the escape factor
(detector-path edges act through the response instead, not through $\ell$); the
emitted $S$ is smooth across an absorber edge; $\mu$ is piecewise-linear on the
Chantler tabulation, as the model interpolates it; and the modelled cutoff at
$E^\ast$ is sharp — which it is in EEDL, where the photon spectrum is tabulated
to $k = T$ and is zero above.

*Limiting case.* A band containing no located edge and no interior kinematic
endpoint returns the geometric baseline **identically, node for node**. Adding
structure adds nodes; adding none changes nothing. Equally, an edge whose jump
falls outside $[E_\mathrm{floor}, E_\mathrm{stop}]$ is reported and dropped,
never refused: it places no requirement on a grid over a band it is not in.

*Cost, structurally.* Refinement only ever *adds interior* nodes and never moves
the band endpoints, so the outermost midpoint half-widths can only narrow. Two
consequences follow without measurement: a source mesh already covered by the
Timepix padded input band stays covered, keeping its channels and its seeded
response matrix; and the added node count *falls* as the baseline gets finer,
because more baseline nodes are displaced by mark nodes than are added beside
them.

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
