# `energy-loss-straggling`

Ledger row: **none yet** — see [Ledger status](#ledger-status). The claim is
expected to land in
[`ledger-transport-background.md`](../ledger-transport-background.md) when the
sampler becomes reachable from transport.
Code: `montecarlo/transport.py::_urban_levels_scalar`,
`::_urban_channels_scalar`, `::_urban_moments_element_scalar`,
`::_urban_poisson_scalar`, `::_urban_ionisation_keV`,
`::_urban_sample_element_keV`, `::_urban_sample_compound_keV`,
`::urban_element_table`, `::urban_loss_moments_keV`.
Anchors: `tests/montecarlo/test_energy_loss_straggling.py` (134 cases).
Source: Geant4 Physics Reference Manual, *Energy loss fluctuations* (Urban
model, `G4UniversalFluctuation`), after H. Bichsel, *Rev. Mod. Phys.* **60**,
663 (1988).

## Claim

Over a step of length $s$ the energy lost by an electron of kinetic energy $E$
is a random variable whose mean is exactly the transport's own stopping power
times the step, $\langle \Delta E \rangle = C s$ with $C = \lvert dE/dx \rvert$,
and whose distribution is the Urban compound-Poisson mixture of two discrete
excitation levels and one $1/\epsilon^2$ ionisation continuum.

## Independent derivation

### Parameterisation

The Urban model replaces the atom by two excitation levels $E_1, E_2$ with
oscillator strengths $f_1, f_2$ and an ionisation continuum starting at
$E_0$. The parameters are fixed by

$$
E_0 = 10\ \text{eV},\qquad
E_2 = 10 Z^2\ \text{eV},\qquad
f_2 = \frac{2}{Z}\ (Z>2),\qquad
f_1 = 1 - f_2,
$$

with the two sum rules

$$
f_1 + f_2 = 1,
\qquad
f_1 \ln E_1 + f_2 \ln E_2 = \ln I,
$$

the second of which defines $E_1$ from the mean excitation energy $I$:

$$
E_1 = \exp\!\left(\frac{\ln I - f_2 \ln E_2}{f_1}\right).
$$

For $Z \le 2$ the K-shell level is absent, $f_2 = 0$, $f_1 = 1$, and the sum
rule collapses to $E_1 = I$. The single tuned parameter is $r = 0.55$, the
fraction of the mean loss carried by the continuum. With no $\delta$-ray
production cut the continuum ceiling is the Møller kinematic limit for
indistinguishable electrons,

$$
T_{\rm up} = T_{\max} = \frac{E}{2}.
$$

Writing the Bethe-type logarithmic factor

$$
L(\varepsilon) \equiv \ln\!\frac{2 m c^2 \beta^2\gamma^2}{\varepsilon} - \beta^2,
\qquad L_I \equiv L(I),
$$

the per-unit-length rates of the three channels are

$$
\Sigma_i = C\,(1-r)\,\frac{f_i}{E_i}\,\frac{L(E_i)}{L_I},\quad i = 1,2,
\qquad
\Sigma_3 = C\,r\,\frac{T_{\rm up} - E_0}{E_0\,T_{\rm up}\,\ln(T_{\rm up}/E_0)} .
$$

The loss over $s$ is the compound Poisson sum

$$
\Delta E = n_1 E_1 + n_2 E_2 + \sum_{k=1}^{n_3}\epsilon_k,
\qquad n_i \sim \mathrm{Poisson}(s\,\Sigma_i),
$$

with the $\epsilon_k$ drawn independently from the $1/\epsilon^2$ density on
$[E_0, T_{\rm up}]$.

### Mean closure

Only the two sum rules are needed. Summing the two discrete channels,

$$
\Sigma_1 E_1 + \Sigma_2 E_2
= \frac{C(1-r)}{L_I}\bigl[f_1 L(E_1) + f_2 L(E_2)\bigr].
$$

Expanding $L$ and using $f_1 + f_2 = 1$ on the $\varepsilon$-independent part
and $f_1\ln E_1 + f_2\ln E_2 = \ln I$ on the rest,

$$
f_1 L(E_1) + f_2 L(E_2)
= \bigl(\ln 2mc^2\beta^2\gamma^2 - \beta^2\bigr) - \ln I
= L_I ,
$$

so

$$
\boxed{\ \Sigma_1 E_1 + \Sigma_2 E_2 = C\,(1-r).\ }
$$

For the continuum, normalise $g(\epsilon) = N/\epsilon^2$ on
$[E_0, T_{\rm up}]$:

$$
N^{-1} = \int_{E_0}^{T_{\rm up}}\frac{d\epsilon}{\epsilon^2}
= \frac{1}{E_0} - \frac{1}{T_{\rm up}}
= \frac{T_{\rm up}-E_0}{E_0 T_{\rm up}},
\qquad
N = \frac{E_0 T_{\rm up}}{T_{\rm up}-E_0}.
$$

Hence

$$
\langle\epsilon\rangle_3 = N\!\int_{E_0}^{T_{\rm up}}\frac{d\epsilon}{\epsilon}
= \frac{E_0 T_{\rm up}\ln(T_{\rm up}/E_0)}{T_{\rm up}-E_0},
\qquad
\langle\epsilon^2\rangle_3 = N\!\int_{E_0}^{T_{\rm up}}\! d\epsilon
= E_0 T_{\rm up},
$$

and $\Sigma_3\langle\epsilon\rangle_3 = C r$ by direct cancellation. Therefore

$$
\langle\Delta E\rangle
= s\bigl[\Sigma_1 E_1 + \Sigma_2 E_2 + \Sigma_3\langle\epsilon\rangle_3\bigr]
= s\bigl[C(1-r) + Cr\bigr] = C s
$$

identically — for any $r$, any $Z$, any $I$, any $E$. Nothing in the argument
used the numerical values of $E_0$, $E_2$, or $r$; closure rests only on the
two sum rules and on the exact cancellation in $\Sigma_3$.

### Variance

A compound Poisson sum of independent channels has
$\operatorname{Var} = s\sum_i \Sigma_i \langle\epsilon^2\rangle_i$ with no
cross terms, the discrete channels contributing $E_i^2$:

$$
\operatorname{Var}(\Delta E)
= s\bigl[\Sigma_1 E_1^2 + \Sigma_2 E_2^2 + \Sigma_3\,E_0 T_{\rm up}\bigr].
$$

### Continuum inverse CDF

$$
F(\epsilon) = N\left(\frac{1}{E_0}-\frac{1}{\epsilon}\right)
= \frac{E_0 T_{\rm up}}{T_{\rm up}-E_0}
  \left(\frac{1}{E_0}-\frac{1}{\epsilon}\right).
$$

Solving $F(\epsilon) = u$ gives

$$
\frac{1}{\epsilon}
= \frac{1}{E_0}\left(1 - u\,\frac{T_{\rm up}-E_0}{T_{\rm up}}\right)
\quad\Longrightarrow\quad
\epsilon = \frac{E_0}{1 - u\,(T_{\rm up}-E_0)/T_{\rm up}} .
$$

At $u=0$ this returns $E_0$; at $u\to 1$ the denominator tends to
$E_0/T_{\rm up}$ and $\epsilon\to T_{\rm up}$. The denominator lies in
$(E_0/T_{\rm up}, 1]$ for $u\in[0,1)$, so a single continuum quantum never
exceeds the Møller ceiling.

### Units and signs

$C$ is $\mathrm{keV\,\AA^{-1}}$; $f_i$ and $L$ are dimensionless; $E_i$ and
$I$ are keV. Then $\Sigma_{1,2} \sim C f_i/E_i$ is $\mathrm{\AA^{-1}}$, and
$\Sigma_3 \sim C\,\mathrm{keV}/\mathrm{keV}^2$ is $\mathrm{\AA^{-1}}$ as well,
so $\langle n_i\rangle = s\Sigma_i$ is dimensionless and $\Delta E$ is keV.
$C$ must be the *magnitude* $\lvert dE/dx\rvert$ because $\Sigma_i \ge 0$ is a
rate; the sampler returns a positive loss, to be subtracted from the electron
energy.

### Limiting case

Every $\langle n_i\rangle = s\Sigma_i$ is linear in $s$, so as $s\to 0$ all
three Poisson means vanish and $P(\Delta E = 0)\to 1$; both moments vanish
linearly in $s$. The mean equals the deterministic $C s$ at *every* $s$, not
only in the limit, so replacing the deterministic loss by this sampler cannot
move any mean the transport already computes.

## Diff against the implementation

Re-implemented from the derivation above without repo helpers (only the
per-element stopping power `_dEds_spliced_element_scalar` was reused, since $C$
is an input to the model, not part of it) and compared over
{graphite, tungsten, WS$_2$, silicon, PtBi$_2$} $\times$
{1, 2, 5, 10, 25, 50, 100, 200, 300} keV.

| quantity | independent | implementation | agreement |
|---|---|---|---|
| $2mc^2\beta^2\gamma^2$ | $2mc^2\tau(\tau+2)$ | `2.0*_MC2_KEV*tau*(tau+2.0)` | identical ($\beta^2\gamma^2=\gamma^2-1$) |
| $\Sigma_{1,2}$ | as derived | `soft*(f_i/E_i)*(log(...)-beta_sq)` | identical |
| $\Sigma_3$ | as derived | matches term for term | identical |
| $\langle\epsilon\rangle_3$, $\langle\epsilon^2\rangle_3$ | $E_0T_{\rm up}\ln(T_{\rm up}/E_0)/(T_{\rm up}-E_0)$, $E_0T_{\rm up}$ | same | identical; quadrature confirms to $10^{-8}$ |
| inverse CDF | $E_0/(1-u(T_{\rm up}-E_0)/T_{\rm up})$ | same | identical |
| mean | $Cs$ | evaluated through the channels | worst deviation $4.4\times10^{-16}$ (1 ulp) over 45 cells |
| variance | $s(\Sigma_1E_1^2+\Sigma_2E_2^2+\Sigma_3E_0T_{\rm up})$ | same | agree to $<10^{-12}$ relative |

The implementation is faithful to the parameterisation, the moments, the
inversion, the units and the sign convention.

## The `E_2` re-solve: deviation from Geant4

`_urban_levels_scalar` drops the K-shell channel and re-solves the sum rules on
$f_1 = 1$, $E_1 = I$ whenever $E_2 = 10Z^2$ eV is inadmissible, i.e. whenever
$E_2 \ge T_{\rm up}$ (below $E = 20Z^2$ eV) or $L(E_2) \le 0$.

The re-solve is exactly mean-preserving: with $f_1 = 1$, $E_1 = I$ the sum rule
$f_1\ln E_1 = \ln I$ holds identically, $L(E_1) = L_I$, and
$\Sigma_1 E_1 = C(1-r)$ on its own. Clamping a negative $\Sigma_2$ to zero
instead — the behaviour attributed to Geant4 — deletes a negative term and
makes the mean *overshoot*; reproduced independently for tungsten as closure
ratios $1.0187 / 1.0098 / 1.0038 / 1.0010$ at $1/2/5/10$ keV, matching the
figures the code and the task document quote. Since exact closure is the sole
reason the model was selected over Landau, Vavilov and Bohr, the re-solve is
the defensible choice.

The variance is *not* invariant under the re-solve, and the write-ups are only
half right about which way it moves. Measured, relative to the clamped variant:

| case | clamped $\operatorname{Var}/\xi T_{\max}$ | re-solved | driver |
|---|---|---|---|
| W, 1 keV | 1.4188 | 1.5142 | surviving level rises $E_1 = 0.645 \to I = 0.727$ keV |
| W, 25 keV | 0.7762 | 0.7554 | K channel with $E_2 = 54.8\ \text{keV} > T_{\max}$ removed |
| W, 100 keV | 0.7398 | 0.7089 | same |

So the top of the band widens for the reason the task document gives, but the
bottom of the band falls for a different reason: above $E \approx 20$ keV in
tungsten the log factor $L(E_2)$ is still positive and the clamp keeps a
channel whose quantum $E_2$ exceeds $T_{\max}$, which the re-solve deletes.
Deleting it is the physically correct move; the stated causal explanation is
incomplete, not the code.

## Variance deficit: the deficit is structural and explainable

Against the analytic Møller second moment $\xi T_{\max}$ (where
$\xi = 2\pi r_e^2 mc^2 n_e s/\beta^2$, so that
$\int_{}^{T_{\max}}\epsilon^2 (\xi/s)\epsilon^{-2}\,d\epsilon \approx
\xi T_{\max}$), the continuum channel alone gives

$$
\frac{\operatorname{Var}_3}{\xi T_{\max}}
= \frac{C\,r\,(T_{\rm up}-E_0)}{\xi\,\ln(T_{\rm up}/E_0)\,T_{\rm up}}
\;\simeq\; \frac{r\,B}{\ln(T_{\max}/E_0)},
\qquad B \equiv \frac{\lvert dE/dx\rvert}{\xi/s},
$$

with $B$ the Berger–Seltzer stopping logarithm. For carbon at 100 keV,
$B = 14.43$ and $\ln(T_{\max}/E_0) = 8.52$, giving $0.932$; adding the $1.7\%$
excitation share reproduces the implementation's $0.948$. The ratio is
therefore not a tuned or accidental number but the fixed combination
$r B/\ln(T_{\max}/E_0)$ — it drifts with material and energy exactly as
measured. Independently reproduced band over five materials and nine energies:
$0.7006$ (PtBi$_2$, 100 keV) to $1.5475$ (PtBi$_2$, 1 keV), matching the task
document's $0.70$–$1.55$ and the test's asserted band.

Leaving the PRM width correction unapplied is stated as a limitation in the
code comment, the test module docstring and the task document, and the band is
pinned by a test rather than hidden. That is honest.

One caveat the write-ups do not draw out: for high $Z$ at low $E$ the variance
is *dominated* by the re-solved excitation channel — $82.6\%$ of the total for
tungsten at 1 keV — whose quantum $E_1 = I = 727$ eV lies above both
$T_{\max} = 500$ eV and, for part of the flight, above the electron's own
kinetic energy. The upper end of the accepted band ($\approx 1.5$) is thus set
by a kinematically inadmissible channel, so it should not be read as evidence
that the model's width is right there.

## Findings

| # | finding | severity |
|---|---|---|
| 1 | The code's block comment quotes the variance band as "0.73--1.42 ... sigma within -14%/+19%", which is the *pre-re-solve* (clamped) measurement. The code implements the re-solve, whose band is 0.70--1.55; the test and the task document both carry the corrected numbers. The shipped derivation comment contradicts the shipped test. | minor |
| 2 | The task document attributes the whole band widening to raising the surviving level to $I$. That explains the upper end only; the lower end moves because the re-solve deletes a $E_2 > T_{\max}$ channel the clamp retained. | minor |
| 3 | The admissibility test is asymmetric: $E_2 > T_{\rm up}$ disqualifies level 2, but the replacement $E_1 = I$ is never tested against $T_{\rm up}$ and exceeds it for tungsten below 1.45 keV. Disclosed in the comment ("`E_1` can sit above `T_up` after the re-solve"), and unavoidable if closure is to hold, but it means "inadmissible above the Møller ceiling" is applied to one level and not the other. | minor, disclosed |
| 4 | `_urban_poisson_scalar`'s inverse-CDF loop is bounded by `k < 10000`. For $\lambda = 0.1$ and for $\lambda$ near 100 the accumulated `cdf` saturates two ulps below $1$, i.e. below the largest attainable uniform $1-2^{-53}$; a draw in that gap exits on the iteration bound and returns $n = 10000$, a finite but absurd loss. Probability per draw $\sim 2\times10^{-16}$. This is the same fail-open shape as the non-converged resonance root this branch just fixed elsewhere. | low, latent |
| 5 | The claim that Geant4 clamps the negative count to zero could not be checked against Geant4 source in this environment; it is quoted from the task document. The PyRITE choice is mean-exact either way, so nothing downstream depends on it. | unverified |
| 6 | `test_sigma_units_are_inverse_length_and_counts_scale_with_step` comments "`Sigma_i` [1/Ang] times `E_i` [keV] must recover `C`" but only asserts the sum is positive. Closure is asserted elsewhere, so this is a comment/assertion mismatch, not a coverage gap. | cosmetic |
| 7 | No ledger row and no `Validation:` marker. See below. | process |

## Ledger status

Slice C declines a ledger row on the ground that the sampler is unreachable
from every transport core and from `simulate_trajectories`, so a row would
describe behaviour no run has. That is a coherent reading of
`docs/validation/methodology.md`'s "the unit of trust is the equation" — the
equation exists but no result depends on it — and it avoids an orphan
`Validation:` marker. It is nonetheless a deviation from
"New physics lands **with** a ledger row + a limiting-case test, or it doesn't
land": the limiting-case test exists, the row does not. The row and this
document's `Validation: energy-loss-straggling` back-reference should land with
the slice that makes the sampler reachable. Adding this file to the
`Beam physics and electron transport` toctree in
`docs/validation/index.md` is likewise a human edit; the verifier does not make
it.

Verdict from this verification: `rederived` (mean closure, both continuum
moments, the inverse CDF, units, signs and the limiting case reproduce
independently; the anchors are green), with findings 1 and 2 to be corrected in
the comment and task document, and 4 to be judged by the author.
