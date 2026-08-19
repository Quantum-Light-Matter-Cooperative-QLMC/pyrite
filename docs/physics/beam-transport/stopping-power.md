# Stopping power and the energy cutoff

Between elastic collisions PyRITE removes energy continuously rather than
sampling individual inelastic events. This is the condensed-history
(continuous-slowing-down) approximation: every inelastic channel is folded into a
single mean energy-loss rate $dE/ds$, evaluated along the flight.

## Joy–Luo modified Bethe law

The stopping power is the Joy–Luo modification of the Bethe expression{cite:p}`joyluo1989`

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

## Validity ceiling

{eq}`eq-stopping-joy-luo` is a low-energy form. Measured against relativistic
ICRU-37 Bethe stopping, the ratio degrades steadily across the energy axis this
repository sweeps:

```{list-table} Joy–Luo stopping power relative to relativistic ICRU-37 Bethe.
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

There is no guard on this, and no automatic switch to a relativistic form. At the
300 keV model ceiling the transport under-stops by roughly a factor of two, which
is the quantitative case behind the gated reference-stopping backlog item.
Results above a few tens of keV should be read with
{numref}`tbl-stopping-validity-ceiling` in hand.

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
- **No density-effect or shell corrections** beyond what the $k J$ modification
  absorbs.
- $J$, $Z$, and $A$ are per element, and only elements in the transport table can
  be used; the catalog rejects materials naming anything else.

## Validation

`Validation: electron-transport` for {eq}`eq-stopping-joy-luo`, the compound
rule, and the mean excitation energies;
`Validation: transport-midpoint-stopping` for the evaluation point and
{eq}`eq-stopping-cutoff-distance`;
`Validation: energy-step-convergence` for the measured path-length bias. See the
[physics validation ledger](../../validation/physics-validation-ledger.md).
