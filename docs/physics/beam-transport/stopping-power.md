# Stopping power and the energy cutoff

Between elastic collisions PyRITE removes energy continuously rather than
sampling individual inelastic events. This is the condensed-history
(continuous-slowing-down) approximation: discrete excitation and ionization
losses are replaced by their mean rate $dE/ds$, evaluated along the
flight{cite:p}`nistestar,akkerman1978`.

The local scale-separation check is the fractional mean loss over an elastic
free path, $|dE/ds|\lambda_{\rm el}/E$: when it is small, energy evolves slowly
compared with the explicitly sampled directional changes. The inelastic and
elastic mean free paths are distinct, material- and energy-dependent
scales{cite:p}`akkerman1978,shinotsuka2015`. Optional straggling restores
fluctuations around that mean without changing it.

## Joy–Luo modified Bethe law

Below each element's Joy–Luo/Berger–Seltzer crossover (see Per-element splice,
below), stopping is the Joy–Luo modification of the Bethe
expression{cite:p}`joyluo1989`

```{math}
:label: eq-stopping-joy-luo

\frac{dE}{ds} =
-7.85\times10^{-4}\,
\frac{\rho\,Z}{A\,E}\,
\ln\!\left[\frac{1.166\,(E + k J)}{J}\right]
\quad[\mathrm{keV\,Å^{-1}}],
```

with

```{math}
:label: eq-stopping-joy-luo-k

k = 0.731 + 0.0688\,\log_{10} Z,
```

$E$ and $J$ in keV, $\rho$ in $\mathrm{g\,cm^{-3}}$, and $A$ the standard atomic
weight. The prefactor is the conventional $7.85\times10^{4}\ \mathrm{keV\,cm^{-1}}$
coefficient converted to keV per ångström.

Unmodified Bethe stopping has $\ln(1.166\,E/J)$, which passes through zero at
$E = J/1.166$ and turns *negative* below it, so the electron would gain energy.
The $kJ$ term keeps the logarithm's argument above unity down to zero kinetic
energy, and $k$ is fitted so the modified curve tracks measured low-energy
stopping.

## Compounds

Stopping is additive over elements (Bragg's rule). The implementation stores a
per-element coefficient

```{math}
:label: eq-stopping-compound-coefficient

c_i = \frac{n_i\,Z_i}{0.602214076},
\qquad n_i~\text{in Å}^{-3},
```

so that

```{math}
:label: eq-stopping-compound

\frac{dE}{ds} =
-\frac{7.85\times10^{-4}}{E}
\sum_i c_i \ln\!\left[\frac{1.166\,(E + k_i J_i)}{J_i}\right].
```

{eq}`eq-stopping-compound-coefficient` is an exact rewrite of $\rho Z/A$, not an
approximation: $\rho/A = n/N_{\rm A}$, and $10^{24}/N_{\rm A} = 1/0.602214076$
carries $\text{Å}^{-3}$ to $\mathrm{cm^{-3}}$ at the same time. Working from
number densities means the catalog never has to keep a separate mass density
consistent with its lattice.

Each element keeps its own $k_i$ from {eq}`eq-stopping-joy-luo-k` and its own
mean excitation energy $J_i$; there is no single effective $Z$ or $J$ for the
compound.

## Mean excitation energies

$J$ comes from the PDG *Atomic and Nuclear Properties* elemental tables, which
follow the ICRU stopping-power compilation, alongside CIAAW 2024 standard atomic
weights for $A$ (`materials/_transport_data.py`); see
[Elemental transport data](../atomic-physics/elemental-transport-data.md).

This is **not** the source Joy and Luo fitted $k$ against, which used
Berger–Seltzer values. The two disagree for light elements: carbon is 78 eV here
against 100 eV in the fit, worth roughly 4% in stopping power at 25 keV. Silicon
agrees to 0.3%. The discrepancy is recorded in the ledger rather than tuned
away.

## Berger–Seltzer relativistic branch

Above each element's crossover (Per-element splice, below), the low-energy $kJ$
modification is dropped for the closed-form relativistic collision-stopping
expression of ICRU Report 37{cite:p}`icru37,bergerseltzer1982`:

```{math}
:label: eq-stopping-bs

\frac{dE}{ds} =
-\frac{2\pi r_e^2 m c^2 N_{\rm A}}{\beta^2}\,\frac{\rho Z}{A}\,
\left[
\ln\!\frac{T^2(T+2)}{2\,(I/mc^2)^2}
+ F^-(T) - \delta
\right]
\quad[\mathrm{keV\,Å^{-1}}],
```

with

```{math}
:label: eq-stopping-bs-fminus

F^-(T) = 1 - \beta^2 +
\frac{T^2/8 - (2T+1)\ln 2}{(T+1)^2},
```

where $T = E/mc^2$ is the kinetic energy in electron rest-mass units, $\beta^2 = 1 -
(T+1)^{-2}$, and $2\pi r_e^2 m c^2 N_{\rm A}$ is a constant with the value $0.1535\ \mathrm{MeV\,cm^2\,mol^{-1}}$.

Converting that constant to per-$\AA$ units ($10^{3}$ keV/MeV times
$10^{-8}$ cm/Å) gives the prefactor $1.535\times10^{-6}\ \mathrm{keV\,Å^{-1}}$
used in the code, in place of the $7.85\times10^{-4}$ of
{eq}`eq-stopping-joy-luo`.

As a check, the non-relativistic limit

```{math}
T \to 0:
\quad \beta^2 \to 2T,
\quad \frac{T+2}{2} \to 1
```

reduces {eq}`eq-stopping-bs` to:

```{math}
:label: nonrel-eq-stopping-bs

-1.535\times10^{-6}\,\frac{mc^2}{E}\frac{\rho Z}{A}\,[2\ln(E/I) + 1 - \ln2]
```

and
$1.535\times10^{-6}\times mc^2 = 7.844\times10^{-4}$ against the conventional
$7.85\times10^{-4}$ — 0.08%, the rounding in that constant.

$\delta$ is the density-effect correction. It is a bulk property of the medium
rather than of an element, so unlike $I$ it factors out of the Bragg sum and
enters as one scalar per layer rather than per element. It is **omitted**:
every call site passes $\delta = 0$.

With $x = \log_{10}\beta\gamma$, the Sternheimer parameterization is

$$
\delta(x) =
\begin{cases}
\delta_0\,10^{2(x - x_0)} & x < x_0\\
2\ln(10)\,x - \bar{C} + a\,(x_1 - x)^k & x_0 \le x < x_1\\
2\ln(10)\,x - \bar{C} & x \ge x_1
\end{cases}
$$ (eq-stopping-sternheimer)

with $\delta_0 = 0$ for non-conductors. The coefficients are read from the PDG
muon energy-loss table headers already cited for $I$ and land in
`materials/_transport_data.py::STERNHEIMER_DENSITY_EFFECT`; nothing in the
transport path reads them, and `transport.sternheimer_delta` exists only to
size what is being dropped.

Onset is at $x_0$. Graphite's $x_0 = -0.009$, so the swept range
($\beta\gamma \le 1.24$) sits *above* onset for this repository's primary
material, and the omission is justified by its measured size rather than by
sitting below threshold.

Measured over all 24 catalog elements, the fractional error in $|dE/ds|$ from
dropping $\delta$, that is $\delta$ divided by the bracket of
{eq}`eq-stopping-bs`, is

```{list-table} Cost of omitting $\delta$, worst catalog element
:name: tbl-stopping-density-effect
:header-rows: 1

* - Kinetic energy
  - Worst fractional error
  - Element
* - 25 keV
  - 0.13%
  - Pd
* - 100 keV
  - 0.44%
  - C
* - 300 keV
  - 1.50%
  - C
```

$\delta$ rises monotonically with $\beta\gamma$, so the 300 keV row bounds the
whole range. At 25 keV the omission is about 45 times smaller than the 6%
Joy–Luo error {eq}`eq-stopping-bs` was introduced to remove, and below the
unmodeled shell corrections and omitted delta-ray transport listed under
*Assumptions and limits*. At the 300 keV ceiling it is 1.5%, stated and
uncorrected. Applying $\delta$ properly would need per-*material* coefficients:
it does not Bragg-add, so the per-element values above bound a compound's
$\delta$ without being able to compose it. The Sternheimer–Peierls general
rules reproduce the tabulated $\bar{C}$ from $I$ and $\hbar\omega_p$ exactly
(better than $10^{-3}$ for all 24 elements) but place $x_0 \ge 0.2$, above the
top of the swept range, so they return $\delta = 0$ throughout and cannot serve
as the bound. {eq}`eq-stopping-bs-fminus` also omits shell corrections, which is
why NIST restricts ESTAR collision stopping to energies $\ge 10$ keV; that is a
second reason, independent of $\delta$, the low-energy branch stays
in the critical path rather than being replaced outright.

The compound rule reuses {eq}`eq-stopping-compound-coefficient` unchanged:
$c_i = n_i Z_i / 0.602214076$ is an exact rewrite of $\rho Z/A$ regardless of
which stopping law multiplies it, so Bragg additivity has the same form as the
Joy–Luo compound rule.

## Per-element splice

The crossover between {eq}`eq-stopping-joy-luo` and {eq}`eq-stopping-bs` is
strongly $Z$-dependent, so there is no single splice energy that works well
everywhere: measured across the 24 catalog elements the two laws cross at
2.66 keV (B) through 10.46 keV (Bi), monotone in $I$. At 10 keV the ratio of the
two laws runs from 0.976 (B) to 1.003 (Bi), so forcing one global crossover
leaves a step in every other element; the best available global choice, 8.0 keV,
still steps by 1.84% in the worst element.

Because stopping is additive over elements ({eq}`eq-stopping-compound`), the
splice does not have to be global. Each element's term switches at *its own*
crossover, found once per element at table-build time by solving
{eq}`eq-stopping-joy-luo` $=$ {eq}`eq-stopping-bs` numerically; the two forms
differ in the shape of the logarithm, not by a constant, so there is no closed
form for the crossing point. That makes every term continuous by construction,
and therefore the compound sum for any material, with no per-material tuning
and no fitted blend, verified to $10^{-12}$ relative for all 24 elements and
all 50 catalog materials. The crossover is well posed because it sits far
above the energy where the Berger–Seltzer bracket
$\bigl[\ln(\cdot) + F^-(T)\bigr]$ changes sign (below 0.71 keV for every
catalog element, versus a 2.66–10.46 keV crossover).

The splice is continuous in *value* only: $C^0$, not $C^1$. The
log-slope $\left(\mathrm{d}\ln|\frac{\mathrm{d}E}{\mathrm{d}s}| / \mathrm{d}\ln E \right)$ steps across the crossover by 0.0145 (B, 2.0%
of the local slope) to 0.0587 (Bi, 8.9%), worst at high $Z$, where the
crossover sits highest. {eq}`eq-stopping-cutoff-distance`, below, needs only
the stopping-power value at one point, not its slope, so the kink does not
affect the cutoff solve; the transport-energy lookup table resolves the kink
to better than $10^{-5}$ relative on its energy grid.

## Joy–Luo drift above its validated range

Used *unbounded*, {eq}`eq-stopping-joy-luo` drifts from relativistic ICRU-37
stopping past the energy where it is validated. That drift is what motivates the
Berger–Seltzer branch above.

```{list-table} Joy–Luo stopping power relative to relativistic ICRU-37 Bethe, unbounded.
:name: tbl-stopping-validity-ceiling
:header-rows: 1

* - Kinetic energy
  - Joy–Luo / ICRU-37
* - 10 keV
  - 0.98
* - 25 keV
  - 0.94
* - 50 keV
  - 0.88
* - 100 keV
  - 0.78
* - 200 keV
  - 0.63
* - 300 keV
  - 0.52
```

Every element switches to {eq}`eq-stopping-bs` at its own crossover
(2.66–10.46 keV), so above that point results follow the relativistic law rather
than the degrading ratio in {numref}`tbl-stopping-validity-ceiling`. Below the
crossover the two laws agree to a few percent, and Joy–Luo is the validated
branch there.

The practical size of the splice is the CSDA range at the default
`E_cut_keV` = 5 keV, unspliced Joy–Luo versus the spliced model:

```{list-table} Change in CSDA range from the splice.
:name: tbl-stopping-splice-range-change
:header-rows: 1

* - Material
  - 25 keV
  - 100 keV
  - 300 keV
* - Graphite
  - −4.2%
  - −15.1%
  - −35.4%
* - Silicon
  - −4.0%
  - −14.9%
  - −35.2%
* - Tungsten
  - −2.8%
  - −14.2%
  - −34.6%
```

Ranges shorten, as they must: Joy–Luo under-stopped above its validated range.
At 300 keV graphite's CSDA range goes from 643 to 416 μm.

### Catalog-wide spread

The range change is nearly material-independent: measured over all 50 catalog
materials it spans −4.2% to −2.6% at 25 keV, −15.1% to −14.1% at 100 keV, and
−35.4% to −34.6% at 300 keV, a band under 1.1 points wide at every energy. It
has to be narrow, because the only material dependence in the ratio of the two
laws enters through the mean excitation energies, and only logarithmically.
Low-$Z$ materials sit at the strongly-shortened end (graphite and diamond are
the extreme) and high-$Z$ at the weakly-shortened end.

Omitting $\delta$ costs at most 0.13% of $|dE/ds|$ at 25 keV and 1.50% at
300 keV ({numref}`tbl-stopping-density-effect`), so it stays omitted with that
error stated. The 1–10 keV window remains unowned by either form: for the
higher-crossover elements it sits above where Joy–Luo fits best and below where
Berger–Seltzer is used. `E_cut_keV` defaults to 5 keV, which bounds most of that
exposure; the residual uncertainty there is stated and uncorrected.

## What the change does downstream

Observables respond to the shortened range of
{numref}`tbl-stopping-splice-range-change` in proportion to how much of that
range the target occupies.

**Thin films are almost untouched.** At the catalog's 1000 Å production
thickness, about 1/60 of the 25 keV CSDA range, an electron crosses the film
having lost a per-mille fraction of its energy, so which stopping law was used
barely enters. Measured on graphite, silicon, and WSe₂ at 30, 100, and 300 keV,
the total bremsstrahlung yield, its mean photon energy, the characteristic-line
yield, and the coherent-line peak position all move by less than 1%, and the
shape of the normalized bremsstrahlung spectrum by less than 0.1% in any bin.
The largest single shift is the coherent yield at 30 keV, +0.97%, a path-length
effect: the electron's in-film trajectory is very slightly shorter, so the phase
it accumulates changes. Repeated with 4000 electrons per seed over 16 seeds,
every one of those shifts is consistent with zero at its own Monte Carlo
error.

**Thick targets change materially.** Once the target is a sizeable fraction of
the range, the shortened range redistributes the electron fates. Measured with
20 000 electrons per model per seed, four seeds, on a slab about half the old
CSDA range thick:

```{list-table} Electron fates under Joy–Luo versus the splice. Slab thickness ≈ half the unspliced model's CSDA range; $E_\mathrm{cut}$ = 5 keV, Mott elastic scattering.
:name: tbl-stopping-splice-fates
:header-rows: 1

* - Case
  - Backscattered $\eta$
  - Transmitted
  - Stopped in target
* - Graphite, 25 keV, 3 μm
  - 0.0441 → 0.0402 (−8.7%)
  - 0.742 → 0.726 (−2.2%)
  - 0.214 → 0.234 (+9.4%)
* - Graphite, 100 keV, 40 μm
  - 0.0473 → 0.0379 (−19.8%)
  - 0.694 → 0.603 (−13.1%)
  - 0.258 → 0.359 (+38.9%)
* - Graphite, 300 keV, 250 μm
  - 0.0658 → 0.0336 (−48.9%)
  - 0.763 → 0.556 (−27.2%)
  - 0.171 → 0.410 (+140.1%)
* - Silicon, 100 keV, 40 μm
  - 0.165 → 0.138 (−16.1%)
  - 0.404 → 0.310 (−23.2%)
  - 0.431 → 0.551 (+28.0%)
* - WSe₂, 100 keV, 40 μm
  - 0.521 → 0.485 (−6.9%)
  - 0 → 0
  - 0.479 → 0.515 (+7.5%)
```

Both loss channels shrink and the stopped fraction absorbs the difference, the
expected signature of a shorter range: fewer electrons reach the far face, and
fewer survive the walk back out to be counted as backscattered. Backscatter
falls even in the semi-infinite WSe₂ case, where transmission is identically
zero: an electron that loses energy faster on the outbound leg has less left for
the return.

Mean deposition depth moves much less than the range does (−0.1% to −2.2% for
graphite and silicon, −22% for WSe₂ at 300 keV), because in a target thinner
than the range the depth distribution is truncated by geometry rather than set
by the range.

**Emitted yields follow the fates, and change sign with thickness.** The same
16-seed ensembles, at 0.1, 10 and 100 μm:

```{list-table} Change in emitted yields from the splice, graphite. 4000 electrons per seed, 16 seeds; uncertainties are the seed-to-seed standard error.
:name: tbl-stopping-splice-yields
:header-rows: 1

* - Thickness
  - 30 keV
  - 100 keV
  - 300 keV
* - 0.1 μm, bremsstrahlung
  - +0.01 ± 0.13%
  - +0.00 ± 0.07%
  - +0.00 ± 0.01%
* - 0.1 μm, characteristic line
  - +0.09 ± 1.18%
  - +0.00 ± 0.78%
  - +0.00 ± 0.10%
* - 10 μm, bremsstrahlung
  - −5.00 ± 0.07%
  - +1.55 ± 0.43%
  - +0.47 ± 0.10%
* - 10 μm, characteristic line
  - −11.84 ± 4.05%
  - +2.00 ± 0.35%
  - +0.60 ± 0.61%
* - 100 μm, bremsstrahlung
  - −4.90 ± 0.05%
  - −14.55 ± 0.12%
  - +9.13 ± 0.49%
* - 100 μm, characteristic line
  - −7.63 ± 1.00%
  - −23.43 ± 0.61%
  - +9.60 ± 1.30%
```

Both signs come from the same shortened range acting on different sides of the
target thickness:

- Where the target is **thicker than the range** (30 keV, 100 keV at 100 μm),
  the electron was always going to stop inside. A shorter range simply means
  fewer radiating segments, so yields **fall**.
- Where the target is **thinner than the old range but comparable to the new
  one** (300 keV at 100 μm), electrons that used to escape out the far side now
  stop inside and keep radiating, so yields **rise** — +9% here, against a −35%
  change in the range itself.

Mean bremsstrahlung photon energy falls almost everywhere (−0.3% to −7.1%),
which is the expected consequence of the electron spending proportionally more
of its history at low energy. Silicon behaves the same way as graphite to within
a few tenths of a point (−5.04% brem at 30 keV/10 μm, +2.14% at 100 keV).

## Checking against ESTAR

The collision stopping power here is the same quantity NIST's
[ESTAR](https://physics.nist.gov/PhysRefData/Star/Text/ESTAR.html) tabulates,
so ESTAR is available as an external oracle. The comparison is run by hand
rather than wired into the test suite, and the repository packages no ESTAR
data.

Two things must be matched before the numbers are comparable:

- ESTAR's CSDA range integrates to zero energy; the ranges here integrate down
  to `E_cut_keV`. Compare against ESTAR's $R(E_0) - R(5\ \mathrm{keV})$, not
  $R(E_0)$.
- ESTAR's *total* CSDA range includes radiative stopping, which this model
  omits entirely. Below ~300 keV the radiative term is well under a percent for
  low $Z$ but not for tungsten or bismuth, so use ESTAR's collision-only
  column, and expect the residual disagreement to grow with $Z$.

Spliced-model ranges from `E_cut_keV` = 5 keV, in μm, for that comparison:

```{list-table} Spliced-model CSDA ranges [μm], integrated from 5 keV.
:name: tbl-stopping-csda-estar
:header-rows: 1

* - Material
  - 25 keV
  - 100 keV
  - 300 keV
* - Graphite (HOPG)
  - 5.92
  - 70.3
  - 415.4
* - Diamond
  - 3.81
  - 45.3
  - 267.7
* - Hexagonal BN
  - 6.10
  - 72.4
  - 427.8
* - Silicon
  - 6.77
  - 77.9
  - 452.3
* - MoS₂
  - 3.84
  - 43.0
  - 246.2
* - WSe₂
  - 2.63
  - 28.5
  - 160.1
```

`tests/montecarlo/test_stopping_csda_range.py` pins these alongside the
unspliced Joy–Luo values, so a regression that moved both together is still
caught.

## Evaluation along a flight

Where {eq}`eq-stopping-compound` is evaluated is set by `energy_model`: the
frozen rule holds it at the flight-start energy, the midpoint rule
evaluates it at $(E_{\rm start}+E_{\rm end})/2$ through one predictor–corrector
pass. Because $|\frac{dE}{ds}|$ *grows* as $E$ falls, the frozen rule systematically
overstates how far an electron travels for a given loss: measurably, 1.2% on
mean path length in the thick 5 keV carbon case. See
[Electron transport](electron-transport.md#energy-controlled-propagation).

## Transport cutoff

Transport of an electron ends when its kinetic energy falls to `E_cut_keV`
(default 5 keV):

- segments below the cutoff radiate essentially nothing in the spectral window of
  interest, so the discarded path length does not carry the observables, while
  the elastic mean free path keeps shortening and the step count keeps growing;
- the remaining energy is deposited and not tracked. There is no secondary
  electron generation, no cascade, and no energy-balance bookkeeping;
- backscatter and transmission fractions are measured to be insensitive to the
  cutoff over 0.1–2 keV (ledger row `electron-transport`), so 5 keV is a cost
  choice rather than a tuned parameter.

The frozen rule reaches the cutoff by overshooting it and clipping, which inflates
the CSDA range. The midpoint rule instead solves the truncation distance for
$E_{\rm end} = E_{\rm cut}$ exactly,

```{math}
:label: eq-stopping-cutoff-distance

s_{\rm cut} =
\frac{E_{\rm cut} - E_{\rm start}}
     {\frac{\mathrm{d}E}{\mathrm{d}s}\bigl((E_{\rm start}+E_{\rm cut})/2\bigr)},
```

so the overshoot disappears. `E_cut_by_electrons` allows a per-electron cutoff
where a study needs it.

## Assumptions and limits

- **Optional Urban straggling.** Both stopping branches define the mean loss
  rate. With `straggling=True`, transport samples an unrestricted Urban
  compound-Poisson loss whose expectation is exactly that same mean; with the
  default `False`, the historical deterministic path is bit-for-bit unchanged
  {cite:p}`geant4prm,bichsel1988`.
  See `Validation: energy-loss-straggling` and the
  [derivation and observable checks](../../validation/beam-transport/energy-loss-straggling.md).
- **No radiative stopping.** Energy carried off by emitted bremsstrahlung and
  characteristic photons is not removed from the electron. The radiation kernels
  read the trajectories; they never feed back.
- **No delta rays.** All inelastic loss is local and continuous, so knock-on
  electrons do not exist as transported particles.
- **No density-effect correction is applied, in either branch.** Joy–Luo has
  none by construction; {eq}`eq-stopping-bs` carries $\delta$ as a parameter,
  but every call site passes $\delta = 0$. Measured cost: under 0.13% of
  $|dE/ds|$ at 25 keV, under 1.50% at 300 keV, worst in graphite
  ({numref}`tbl-stopping-density-effect`).
- **{eq}`eq-stopping-bs` omits shell corrections**, which is why it is used
  only above each element's crossover; below it, {eq}`eq-stopping-joy-luo`'s
  $kJ$ term is the only correction applied.
- $J$, $Z$, and $A$ are per element, and only elements in the transport table can
  be used; the catalog rejects materials naming anything else.

## Validation

`Validation: electron-transport` for {eq}`eq-stopping-joy-luo` as the
low-energy branch, the compound rule, and the mean excitation energies;
`Validation: relativistic-bethe-stopping` for {eq}`eq-stopping-bs` and the
per-element splice;
`Validation: transport-midpoint-stopping` for the evaluation point and
{eq}`eq-stopping-cutoff-distance`;
`Validation: energy-step-convergence` for the measured path-length bias. See the
[physics validation ledger](../../validation/physics-validation-ledger.md).
