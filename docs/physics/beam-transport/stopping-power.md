# Stopping power and the energy cutoff

Between elastic collisions PyRITE removes energy continuously rather than
sampling individual inelastic events. This is the condensed-history
(continuous-slowing-down) approximation: every inelastic channel is folded into a
single mean energy-loss rate $dE/ds$, evaluated along the flight.

## Joy–Luo modified Bethe law

Below each element's Joy–Luo/Berger–Seltzer crossover — see Per-element splice,
below — stopping is the Joy–Luo modification of the Bethe
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

The $kJ$ term is the whole point of the modification. Unmodified Bethe stopping
has $\ln(1.166\,E/J)$, which passes through zero at $E = J/1.166$ and turns
*negative* below it — the electron would gain energy. Adding $kJ$ inside the
logarithm keeps the argument above unity down to zero kinetic energy, so the law
stays physically signed across the whole range the transport can reach, and $k$
is fitted so the modified curve tracks measured low-energy stopping.

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
agrees to 0.3%. The choice favors a single consistent, citable atomic dataset
across the catalog over exact fidelity to one 1989 fit, and the discrepancy is
recorded in the ledger rather than tuned away.

## Berger–Seltzer relativistic branch

Above each element's crossover (Per-element splice, below), the low-energy $kJ$
modification is dropped for the closed-form relativistic collision-stopping
expression of ICRU Report 37{cite:p}`icru37,bergerseltzer1982`:

```{math}
:label: eq-stopping-bs

\frac{dE}{ds} =
-\frac{2\pi r_e^2 m c^2 N_{\rm A}}{\beta^2}\,\frac{\rho Z}{A}\,
\left[
\ln\!\frac{\tau^2(\tau+2)}{2\,(I/mc^2)^2}
+ F^-(\tau) - \delta
\right]
\quad[\mathrm{keV\,Å^{-1}}],
```

with

```{math}
:label: eq-stopping-bs-fminus

F^-(\tau) = 1 - \beta^2 +
\frac{\tau^2/8 - (2\tau+1)\ln 2}{(\tau+1)^2},
```

$\tau = E/mc^2$ the kinetic energy in electron rest-mass units, $\beta^2 = 1 -
(\tau+1)^{-2}$, and $2\pi r_e^2 m c^2 N_{\rm A} = 0.1535\ \mathrm{MeV\,cm^2\,mol^{-1}}$.
Converting that constant to per-ångström units ($10^{3}$ keV/MeV times
$10^{-8}$ cm/Å) gives the prefactor $1.535\times10^{-6}\ \mathrm{keV\,Å^{-1}}$
used in code, in place of the $7.85\times10^{-4}$ of
{eq}`eq-stopping-joy-luo`. As a check, the non-relativistic limit ($\tau\to0$:
$\beta^2\to2\tau$, $(\tau+2)/2\to1$) reduces {eq}`eq-stopping-bs` to
$-(1.535\times10^{-6}\,mc^2/E)\,\rho Z/A\,[2\ln(E/I) + 1 - \ln2]$, and
$1.535\times10^{-6}\times mc^2 = 7.844\times10^{-4}$ against the conventional
$7.85\times10^{-4}$ — 0.08%, the rounding in that constant.

$\delta$ is the density-effect correction. It is a bulk property of the medium
rather than of an element, so unlike $I$ it factors out of the Bragg sum and
enters as one scalar per layer rather than per element. Over the 1–300 keV
range this repository sweeps, $\beta\gamma \le 1.24$ at every energy, which
sits below the onset $x_1$ tabulated by the same PDG Sternheimer parameters
already cited for $I$ — so $\delta$ is expected to be small — but that is a
bound, not a measurement, and $\delta$ is currently **omitted**: every call
site passes $\delta = 0$ pending a per-material measurement from those
parameters. {eq}`eq-stopping-bs-fminus` also omits shell corrections, which is
why NIST restricts ESTAR collision stopping to energies $\ge 10$ keV; that is a
second reason, independent of $\delta$, the low-energy branch stays
load-bearing rather than being replaced outright.

The compound rule reuses {eq}`eq-stopping-compound-coefficient` unchanged:
$c_i = n_i Z_i / 0.602214076$ is an exact rewrite of $\rho Z/A$ regardless of
which stopping law multiplies it, so Bragg additivity has the same form as the
Joy–Luo compound rule.

## Per-element splice

The crossover between {eq}`eq-stopping-joy-luo` and {eq}`eq-stopping-bs` is
strongly $Z$-dependent, so there is no single splice energy that works well
everywhere: measured across the 24 catalog elements the two laws cross at
2.66 keV (B) through 10.46 keV (Bi), monotone in $I$. The 2% agreement at
10 keV this page used to quote for the crossover is a carbon number and does
not generalize — at 10 keV the ratio runs from 0.976 (B) to 1.003 (Bi).
Forcing one global crossover leaves a step in every other element; the best
available global choice, 8.0 keV, still steps by 1.84% in the worst element.

Because stopping is additive over elements ({eq}`eq-stopping-compound`), the
splice does not have to be global. Each element's term switches at *its own*
crossover, found once per element at table-build time by solving
{eq}`eq-stopping-joy-luo` $=$ {eq}`eq-stopping-bs` numerically — the two forms
differ in the shape of the logarithm, not by a constant, so there is no closed
form for the crossing point. That makes every term continuous by construction,
and therefore the compound sum for any material, with no per-material tuning
and no fitted blend — verified to $10^{-12}$ relative for all 24 elements and
all 50 catalog materials. The crossover is well posed because it sits far
above the energy where the Berger–Seltzer bracket
$\bigl[\ln(\cdot) + F^-(\tau)\bigr]$ changes sign (below 0.71 keV for every
catalog element, versus a 2.66–10.46 keV crossover).

The splice is continuous in *value* only — it is $C^0$, not $C^1$. The
log-slope $d\ln|dE/ds|/d\ln E$ steps across the crossover by 0.0145 (B, 2.0%
of the local slope) to 0.0587 (Bi, 8.9%), worst at high $Z$, where the
crossover sits highest. {eq}`eq-stopping-cutoff-distance`, below, needs only
the stopping-power value at one point, not its slope, so the kink does not
affect the cutoff solve; the transport-energy lookup table resolves the kink
to better than $10^{-5}$ relative on its energy grid.

## Validity ceiling — resolved above each element's crossover

The table below is the measurement that motivated the Berger–Seltzer branch
above: it records how far {eq}`eq-stopping-joy-luo` drifts from relativistic
ICRU-37 stopping when used *unbounded*, past the energy where it is
validated. It is retained as that record, not as a description of what the
transport does today.

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

Per the per-element splice above, every element switches to {eq}`eq-stopping-bs`
at its own crossover (2.66–10.46 keV), so above that point results follow the
relativistic law rather than the degrading ratio in
{numref}`tbl-stopping-validity-ceiling`. Below the crossover — where the
table's own numbers show the two laws already agreeing to a few percent —
Joy–Luo remains the model, and it is the validated branch there.

The practical size of the change is the CSDA range at the default
`E_cut_keV` = 5 keV, analytic Joy–Luo versus the spliced model:

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

Two things stay open. First, $\delta$ is bounded ($\beta\gamma \le 1.24$ over
the whole swept range sits below the Sternheimer onset for every catalog
solid) but not yet measured per material — see the Berger–Seltzer branch,
above. Second, the 1–10 keV window is not cleanly owned by either form: for
the higher-crossover elements it sits above where Joy–Luo fits best and below
where Berger–Seltzer is used. `E_cut_keV` defaults to 5 keV, which bounds most
of that exposure, but the residual uncertainty there is stated rather than
inherited silently.

## Evaluation along a flight

Where {eq}`eq-stopping-compound` is evaluated is set by `energy_model`, not by
this page: the frozen rule holds it at the flight-start energy, the midpoint rule
evaluates it at $(E_{\rm start}+E_{\rm end})/2$ through one predictor–corrector
pass. Because $|dE/ds|$ *grows* as $E$ falls, the frozen rule systematically
overstates how far an electron travels for a given loss — measurably, 1.2% on
mean path length in the thick 5 keV carbon case. See
[Electron transport](electron-transport.md#energy-controlled-propagation).

# Transport cutoff

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
     {(dE/ds)\bigl((E_{\rm start}+E_{\rm cut})/2\bigr)},
```

so the overshoot disappears. `E_cut_by_electrons` allows a per-electron cutoff
where a study needs it.

## Assumptions and limits

- **No straggling.** {eq}`eq-stopping-joy-luo` is a mean loss rate; the
  fluctuation about it (Landau/Vavilov) is not sampled. This biases the mean
  arrival time and not only its variance, at a level above the numerical
  tolerance the midpoint rule reaches — so refining the propagator does not make
  the timing exact, it makes an unmodeled physical spread the limiting error.
- **No radiative stopping.** Energy carried off by emitted bremsstrahlung and
  characteristic photons is not removed from the electron. The radiation kernels
  read the trajectories; they never feed back.
- **No delta rays.** All inelastic loss is local and continuous, so knock-on
  electrons do not exist as transported particles.
- **No density-effect correction is applied, in either branch.** Joy–Luo has
  none by construction; {eq}`eq-stopping-bs` carries $\delta$ as a parameter,
  but every call site passes $\delta = 0$ pending the per-material Sternheimer
  measurement (see the Berger–Seltzer branch, above).
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
