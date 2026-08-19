# Elastic scattering

The elastic model supplies two quantities to the transport loop: **how far** an
electron flies before its next elastic collision, and **through what angle** it
turns when that collision happens. Both are functions of the electron's kinetic
energy and of the elements present in the current layer.

PyRITE uses a single-scattering (CASINO-style) treatment: every elastic event is
sampled explicitly, and only the inelastic energy loss between events is
condensed into a continuous slowing-down law
([Stopping power](stopping-power.md)).

## Selecting a model

`elastic_model` chooses between two internally consistent parameterizations.

```{list-table} Elastic model options.
:name: tbl-elastic-model-options
:header-rows: 1

* - Value
  - Total cross section
  - Screening parameter
  - External data
* - `"mott"` (default)
  - Browning fit to tabulated Mott totals
  - calibrated per element against NIST SRD 64 Mott transport cross sections
  - `mott_transport_cross_sections/DisplayCalcTCSTableFor<El>.csv`
* - `"sr"`
  - relativistic screened-Rutherford total
  - analytic Bishop/Joy form
  - none
```

The two differ in both the collision *rate* and the momentum-transfer rate. They
are separate physical models, not a fast path and an accurate path; the ledger
records backscatter coefficients for both.

## Free path

Elastic scattering rates are additive over the elements of the active layer. For
element $i$ with number density $n_i$ and total elastic cross section
$\sigma_i(E)$,

```{math}
:label: eq-elastic-mean-free-path

\Sigma(E) = \sum_i n_i\,\sigma_i(E),
\qquad
\lambda(E) = \frac{10^{8}}{\Sigma(E)} .
```

Number densities are stored per cubic ångström and converted once to
$\mathrm{cm^{-3}}$ ($n\,[\mathrm{cm^{-3}}] = 10^{24}\,n\,[\text{Å}^{-3}]$), cross
sections are in $\mathrm{cm^2}$, so $\Sigma$ is in $\mathrm{cm^{-1}}$ and the
$10^{8}$ converts the resulting length to ångström.

A flight length follows from the exponential collision law. The transport draws
one optical depth $\tau = -\ln U$, $U \sim \mathrm{Uniform}(0,1)$, per physical
flight and converts it at the flight's own hazard, $s = \tau\,\lambda(E)$. When
the flight is subdivided into numerical substeps the same $\tau$ is consumed
across them at each substep's hazard; see
[Electron transport](electron-transport.md#physical-flights-and-numerical-substeps).

The elastic hazard is evaluated at the flight-start energy under **both**
propagation rules. `energy_model="midpoint"` controls stopping and the transport
clock only, so a rising hazard along a lossy flight is not resolved by the
propagator — it is resolved, if at all, by `max_dE_frac` substepping.

### Browning total cross section

```{math}
:label: eq-elastic-browning-total

\sigma_{\rm el}(Z,E) =
\frac{3.0\times10^{-18}\,Z^{1.7}}
     {E + 0.005\,Z^{1.7}\sqrt{E} + 0.0007\,Z^{2}/\sqrt{E}}
\quad[\mathrm{cm^2}],
```

with $E$ in keV. This is the Browning empirical fit to tabulated Mott total
elastic cross sections.{cite:p}`browning1994,srd64`

**Stated validity is 0.1–30 keV and $Z \le 92$.** PyRITE evaluates it above
30 keV with no guard, because the sweep axis runs to the 300 keV model ceiling.
The extrapolation is a known and ledgered limitation, not a checked result.

### Relativistic screened-Rutherford total cross section

```{math}
:label: eq-elastic-sr-total

\sigma_{\rm SR}(Z,E) =
5.21\times10^{-21}\,\frac{Z^{2}}{E^{2}}\,
\frac{4\pi}{\alpha(1+\alpha)}
\left(\frac{E + 511}{E + 1024}\right)^{2}
\quad[\mathrm{cm^2}],
```

again with $E$ in keV, and $\alpha$ the analytic screening parameter of
{eq}`eq-elastic-screening-joy`. The final bracket is the standard relativistic
correction factor written with the electron rest energy in keV.

## Angular deflection

Both models sample the polar angle from the screened-Rutherford differential
cross section

```{math}
:label: eq-elastic-sr-differential

\frac{d\sigma}{d\Omega} \propto
\frac{1}{\left(1 - \cos\theta + 2\alpha\right)^{2}},
```

whose cumulative distribution inverts in closed form:

```{math}
:label: eq-elastic-costheta-inversion

\cos\theta = 1 - \frac{2\alpha R}{1 + \alpha - R},
\qquad R \sim \mathrm{Uniform}(0,1).
```

The azimuth is uniform on $[0, 2\pi)$, and the new direction is built by rotating
the incoming unit vector into a non-degenerate local frame. Only the screening
parameter $\alpha$ distinguishes the two models' angular behavior.

### Analytic screening parameter

```{math}
:label: eq-elastic-screening-joy

\alpha_{\rm SR}(Z,E) = \frac{3.4\times10^{-3}\,Z^{0.67}}{E},
\qquad E~\text{in keV},
```

the Bishop/Joy form used directly by `elastic_model="sr"`.

### Mott-calibrated screening parameter

`elastic_model="mott"` does **not** use {eq}`eq-elastic-screening-joy`. It
chooses $\alpha$ per element and per energy so that the screened-Rutherford
angular law reproduces the tabulated Mott **transport** cross section
$\sigma_{\rm tr} = \int (1-\cos\theta)\,d\sigma$. Dividing by the total gives the
mean momentum-transfer fraction, and for {eq}`eq-elastic-sr-differential` that
first moment is available in closed form:

```{math}
:label: eq-elastic-first-moment

\langle 1 - \cos\theta \rangle(\alpha) =
2\alpha\left[(1+\alpha)\ln\!\left(1 + \frac{1}{\alpha}\right) - 1\right].
```

The calibration solves

```{math}
:label: eq-elastic-alpha-calibration

\langle 1 - \cos\theta \rangle(\alpha) =
\frac{\sigma_{\rm tr}^{\rm NIST}(E)}{\sigma_{\rm el}^{\rm Browning}(E)}
```

for $\alpha(E)$. The left side is monotonic in $\alpha$, so the inversion is a
bisection in $\log_{10}\alpha$ over a fixed bracket, run once per element on the
tabulated energy grid (50 eV–300 keV, 401 points) and cached. Transport then
interpolates $\log_{10}\alpha$ linearly in $\log_{10}E$, clamped at both
endpoints.

The result is a model whose collision rate matches Mott totals and whose
momentum-transfer rate matches Mott transport cross sections, while retaining the
analytically invertible angular law. It does **not** reproduce the full Mott
differential cross section: structure beyond the first moment — diffraction
minima, large-angle detail — is absorbed into a single effective screening
parameter.

### Missing tables

Elements without a NIST transport table (tungsten among them) fall back to
{eq}`eq-elastic-screening-joy` for the angular draw while keeping the Browning
total. The miss is cached per element per process and logged once at `DEBUG`
(`CXR_MC_DEBUG=1` to see it). A worker pool re-logs once per worker, since each
worker holds its own cache.

## Compounds and layers

Rates are assembled per layer from element number densities, so a compound or an
alloy is a sum over {eq}`eq-elastic-mean-free-path` with no mixing rule beyond
additivity. When a flight ends in a collision, the scattering element is drawn
with probability

```{math}
:label: eq-elastic-element-choice

P(i) = \frac{n_i\,\sigma_i(E)}{\sum_k n_k\,\sigma_k(E)},
```

which is the same partial-rate decomposition that produced $\lambda$. Layer
switching changes the element list, the densities, and therefore both the rate
and the angular draw; see [Transport geometry](../geometry/transport-geometry.md).

## Limiting cases and assumptions

- $\alpha \to 0$ (high energy, low $Z$) drives {eq}`eq-elastic-costheta-inversion`
  toward $\cos\theta \to 1$: forward-peaked, unscreened Rutherford behavior.
- $\alpha \to \infty$ makes the angular law isotropic and
  {eq}`eq-elastic-first-moment` approach 1.
- Single element with $n$ and $\sigma$ constant reduces
  {eq}`eq-elastic-mean-free-path` to $\lambda = 1/(n\sigma)$ up to the unit
  factor, and the flight law to the textbook exponential.
- Elastic events are treated as instantaneous, energy-conserving direction
  changes: no nuclear recoil energy loss, no spin-polarization bookkeeping, no
  coherent (crystal) elastic scattering. The lattice enters the radiation
  kernels, not the transport deflections.
- Cross sections are isotropic material averages; channeling is not modeled.

## Validation

`Validation: electron-transport` — see the row in the
[physics validation ledger](../../validation/physics-validation-ledger.md) and
the [write-up](../../validation/beam-transport/electron-transport.md), which
records measured backscatter coefficients for both models against
Hunger–Küchler and states the extrapolation ceiling above.
