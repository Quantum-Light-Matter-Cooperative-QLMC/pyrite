# `energy-loss-straggling`

Ledger row: [`energy-loss-straggling`](../ledger-transport-background.md#energy-loss-straggling).
Code: `montecarlo/transport/straggling.py::_urban_levels_scalar`,
`::_urban_channels_scalar`, `::_urban_moments_element_scalar`,
`::_urban_poisson_scalar`, `::_urban_ionisation_keV`,
`::_urban_sample_element_keV`, `::_urban_sample_compound_keV`,
`::urban_element_table`, `::urban_loss_moments_keV`;
`montecarlo/transport/api.py::simulate_trajectories`.
Anchors: `tests/montecarlo/test_energy_loss_straggling.py` (134 cases), the
straggling transport/RNG/core tests, and
`checks/energy_loss_straggling_observables.py`.
Source: Geant4 Physics Reference Manual, *Energy loss fluctuations* (Urban
model, `G4UniversalFluctuation`), after Bichsel's thin-detector
treatment{cite:p}`geant4prm,bichsel1988`.

`Validation: energy-loss-straggling`

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
f_2 = \frac{2}{Z}\ (Z\ge 2),\qquad
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

The PRM separately states $f_2=0$ for $Z=1$. PyRITE uses the two-level branch
only for $Z>2$ and re-solves $Z\le2$ to $f_2=0$, $f_1=1$, $E_1=I$; that is a
source deviation for $Z=2$, outside the currently supported transport-element
set. The single tuned parameter is $r = 0.55$, the fraction of the mean loss
carried by the continuum. With no $\delta$-ray production cut the continuum
ceiling is the Møller kinematic limit for indistinguishable electrons,

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

## Fresh-context post-fix re-validation

The source equations and implementation agree for the channel rates,
continuum moments, inverse CDF, units, and positive-loss convention. The three
previous transport-wide discrepancies have been addressed.

### Bounded-rate exact Poisson counts

For a requested mean $\lambda>0$, define

$$
q=\left\lceil\frac{\lambda}{64}\right\rceil,
\qquad
\mu=\frac{\lambda}{q}\le64.
$$

If $N_j\sim\operatorname{Poisson}(\mu)$ are independent, their probability
generating function gives

$$
\prod_{j=1}^{q}\exp\!\bigl(\mu(z-1)\bigr)
=\exp\!\bigl(\lambda(z-1)\bigr),
$$

so $\sum_jN_j\sim\operatorname{Poisson}(\lambda)$ exactly. Both the host
sampler and all three CUDA channel transcriptions implement this decomposition,
use one counter-addressed uniform per chunk, and sum the counts. A direct
300000-key comparison at $\lambda=100$ gave mean $100.001277$, variance
$99.846795$, and skewness $0.098979$ against the Poisson expectation $0.1$;
the unsplit call was pathwise identical to two sequential
$\operatorname{Poisson}(50)$ calls because both address the same two uniforms.

The former `k < 10000` escape is gone. Each inverse-CDF recurrence now stops
when `next_cdf <= cdf`, returning the current tail count if binary64 addition
can no longer advance the CDF. This removes the absurd $n=10000$ escape. At the
largest representable input uniform, the returned extreme-tail quantile can
still differ from an ideal real-arithmetic Poisson quantile by a few counts
(for example $137$ versus $139$ for $\lambda=64$); this affects one or a few
$2^{-53}$ uniform-grid atoms and is ordinary finite-precision tail rounding,
not the former Gaussian-law discrepancy.

### Evolving-energy substeps

Let $\nu_E(d\epsilon)$ be the frozen-energy jump-intensity measure and
$C(E)=\int\epsilon\,\nu_E(d\epsilon)$. For two small substeps $h_1,h_2$, the
mean relative to one frozen unsplit draw contains

$$
h_1h_2\int
\bigl[C(E-\epsilon)-C(E)\bigr]\,\nu_E(d\epsilon)
=h_1h_2\bigl[-C(E)C'(E)+R(E)\bigr],
$$

with

$$
R(E)=\int
\bigl[C(E-\epsilon)-C(E)+\epsilon C'(E)\bigr]\,\nu_E(d\epsilon).
$$

The updated derivation retains this full generator rather than dropping
$R(E)$. It also correctly uses conditional independence:

$$
\operatorname{Cov}(X_1,X_2)
=h_2\operatorname{Cov}\!\bigl(X_1,C(E-X_1)\bigr),
$$

with the corresponding law-of-total-variance terms. The regression evaluates
the full three-channel generator integral, resolves $R(E)$ from zero, and
matches the sampled mean shift. The covariance identity is now correct in the
derivation, but it is not separately asserted by a numerical regression.

### Cutoff tie convention

All five host applications and the CUDA transcription use

    stragg_loss > delta_cut
    or (stragg_loss == delta_cut and not geometry_event)

matching the ledger's explicit convention that geometry wins an exact row-end
tie. Away from that tie, monotonic nonnegative loss makes the total-loss
threshold an exact indicator of passage by the row end. The
fluid-interpolated crossing location remains approximate as disclosed.

## Findings

| # | finding | severity |
|---|---|---|
| 1 | The code's block comment formerly quoted the pre-re-solve variance band. It now states the re-solved $0.70$--$1.55$ band pinned by the test. | resolved |
| 2 | The task document attributes the whole band widening to raising the surviving level to $I$. That explains the upper end only; the lower end moves because the re-solve deletes a $E_2 > T_{\max}$ channel the clamp retained. | minor |
| 3 | The admissibility test is asymmetric: $E_2 > T_{\rm up}$ disqualifies level 2, but the replacement $E_1 = I$ is never tested against $T_{\rm up}$ and exceeds it for tungsten below 1.45 keV. Disclosed in the comment ("`E_1` can sit above `T_up` after the re-solve"), and unavoidable if closure is to hold, but it means "inadmissible above the Møller ceiling" is applied to one level and not the other. | minor, disclosed |
| 4 | The former `k < 10000` escape now terminates when adding the next recurrence term cannot advance the binary64 CDF, so no absurd count escapes. The largest uniform-grid atom can still differ by a few counts from an ideal real-arithmetic extreme-tail quantile; probability is at most a few times $2^{-53}$. | resolved safety defect; finite-precision tail qualification disclosed |
| 5 | Earlier prose attributed a simple negative-count clamp to Geant4. Current and legacy Geant4 implementations contain explicit low-energy branches, so that attribution is removed. PyRITE's re-solve is documented only as its own mean-preserving deviation. | resolved source attribution |
| 6 | `test_sigma_units_are_inverse_length_and_counts_scale_with_step` comments "`Sigma_i` [1/Ang] times `E_i` [keV] must recover `C`" but only asserts the sum is positive. Closure is asserted elsewhere, so this is a comment/assertion mismatch, not a coverage gap. | cosmetic |
| 7 | The sampler originally landed before production reachability and therefore had no ledger row or `Validation:` marker. Slice I closed both process gaps. | resolved |
| 8 | Means above $64$ are now split into independent equal bounded-rate chunks. Poisson additivity restores the exact count law and frozen-energy infinite divisibility; host and CUDA transcribe the same recurrence and draw order. | resolved |
| 9 | The evolving-energy derivation now retains $\int[C(E-\epsilon)-C(E)]\nu_E(d\epsilon)=-C(E)C'(E)+R(E)$ and the covariance induced by using $E-X_1$ in the next draw. The regression numerically pins the mean-generator remainder; the covariance identity remains symbolic rather than separately anchored. | resolved derivation; covariance anchor gap |
| 10 | The ledger now states that geometry wins an exact row-end tie, matching the strict-$>$ branch used by every host core and CUDA. | resolved |
| 11 | The CUDA first-row parity test compares `E_keV`, which is the row-start energy, rather than `E_end_keV`; it therefore anchors the entering transport state but not the applied sampled loss. | applied-loss parity anchor gap |

## Production integration and observable evidence

The sampler is now applied by every host transport core and the exact CUDA
kernel. The CUDA LUT combination raises rather than returning an unstraggled
result; production selection routes a straggled CUDA run to the exact kernel.
The exact CUDA implementation is verified on hardware: all five straggling
tests pass on an NVIDIA GeForce RTX 5080 (driver 610.47, CuPy 14.1.1), covering
disabled-path bit identity, deterministic replay, finite/nonnegative energy
bookkeeping, first-row transport-state parity, and ensemble distributional
agreement. Direct host/device parity of the first applied loss remains untested
because that check compares the row-start `E_keV`, not `E_end_keV`.
With `straggling=False` (the default), every transport core remains bit-for-bit
identical to the pre-feature path and no straggling output is emitted.

Except for the geometry-event tie convention, the cutoff indicator is exact
for a nonnegative frozen-row loss: $\Delta E \ge
E_{\rm start}-E_{\rm cut}$ if and only if passage occurs by the row end. The
crossing position uses the fluid interpolation
$s_{\rm cut}=s(E_{\rm start}-E_{\rm cut})/\Delta E$ and ends at
$E_{\rm cut}$ exactly. At frozen energy, splitting is distributionally
invariant for all supported means because bounded-rate chunks preserve the
Poisson law. With evolving energy, subdivision discretizes the state-dependent
jump kernel through the full generator and covariance terms derived above.

The committed paired-seed observable check used HOPG at 25 keV. At 1 um of
fixed cumulative material path and 1 keV photon energy, straggling changes the
mean free-clock phase by $+13.607 \pm 2.401$ rad, inside the predeclared
9.8--19 rad interval, and changes its standard deviation from
$0.560 \pm 0.032$ rad to $378.628 \pm 5.683$ rad. This is the free clock term
$\phi=E_\gamma t/(\hbar c)$, not the coherent kernel's complete phase. At 5 um,
backscatter changes by $+0.938 \pm 0.199$ percentage points, transmission by
$+5.229 \pm 0.743$ points, cutoff stopping by $-6.167 \pm 0.860$ points, and
the cutoff-stopped range standard deviation opens from
$0.514 \pm 0.007$ Angstrom to $11692 \pm 168$ Angstrom.

The 250 eV--25 keV bremsstrahlung integral changes by
$+0.140\% \pm 0.627\%$ (not resolved from zero); normalized spectral total
variation is $0.334\% \pm 0.053\%$. In the production coherent HOPG (002)
pure-geometry, zero-bunch-offset limit, integrated line yield falls
$12.52\% \pm 2.54\%$ and peak height falls $19.50\% \pm 2.12\%$. Those line
figures are not an angle- or bunch-averaged experimental observable, and the
separate mean-stopping uncertainty remains owned by
`feature/reference-electron-stopping-data`.

Verdict from this fresh-context post-fix verification: `rederived`. Mean
closure, both continuum moments, inverse CDF, units, signs, $s\to0$, exact
bounded-rate Poisson additivity, evolving-energy generator/covariance
semantics, and the geometry-tie convention reproduce independently. The
covariance term is not separately numerically anchored. Exact CUDA hardware
re-validation passes. Suggested human ledger edit: retain `rederived` and the
covariance and CUDA applied-loss anchor qualifications. Only a human may
adjudicate or move the claim to `signed-off`.
