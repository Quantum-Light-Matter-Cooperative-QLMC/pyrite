# Detector-cone variance reduction: pending verification

Validation: `detector-cone-variance-reduction`

**Status: unverified.** This records the intended CPU prototype estimator for
issue #203. It is implementation-context documentation, not an independent
derivation or a production recommendation. The [line-attribution research
record](../../research/beam-transport/detector-cone-line-variance.md) motivates
the study; it does not validate this estimator.

## Proposed estimator and source law

The source angular law is the repository's
[`elsepa-elastic-sampling`](../ledger-transport-background.md#elsepa-elastic-sampling)
claim: the normalized polar quantile interpolant at the current energy, with
uniform azimuth about the incoming direction. No new elastic cross section is
introduced. The proposed change of measure uses the elementary identity
$\mathbb{E}_q[(p/q)H]=\int pH$ wherever the proposal covers the score's support.

For an outgoing direction in the detector cone, `cone_split_event` samples the
analog polar quantile uniformly on a covering interval of mass $P_B$, followed
by the allowed azimuth arc of half-width $h(\theta)$ in radians. Relative to
the analog polar/azimuth law, the proposed likelihood ratio on that cone is

$$
w(\theta)=P_B\frac{h(\theta)}{\pi}.
$$

The prototype first applies roulette with probability
$s=\min(1,U/w_t)$, where $U$ is its computed bound on that likelihood ratio
and $w_t$ is the requested target weight. A surviving copy carries $w/s$.
The analog primary keeps its path before the event and its future outside
the cone; its future is stopped when the analog outgoing direction lies
inside the cone. Copies continue without further splitting.

The intended sample unit is one incident primary, including all its copies:

$$
X_i=H_i+\sum_{c:\,\mathrm{parent}(c)=i}w_c H_c,
\qquad
\bar X=\frac{1}{N}\sum_i X_i,
\qquad
\mathrm{SE}(\bar X)=\frac{s_X}{\sqrt N}.
$$

Weights, quantile probabilities and the reported relative SE are dimensionless;
$X_i$ retains the units of its underlying tally. Copies of one primary are
correlated and are not separate independent observations. The intended
figure of merit is $1/(\mathrm{RSE}^2 T)$ with wall time $T$ in seconds.

## Assumptions and limits requiring verification

- CPU per-electron, ungrooved ELSEPA transport; no secondary cascade, GDF
  source or electron-block admission. One detector direction per run.
- Linear incoherent tallies only. Coherent amplitude cross terms between
  copies are outside this estimator's contract.
- Independent copy keys depend on primary stream key and flight index;
  split decisions must not consume the primary transport stream.
- A disabled request must preserve analog arrays and RNG behavior. Vanishing
  cone mass must create no weighted future; roulette must preserve the
  conditional expectation. Copy-buffer exhaustion raises instead of silently
  dropping contributions.

These are intended contracts, not verified conclusions. In particular, the
proposal support, polar-quantile interval, bound used by roulette, event and
path partition, copy launch state and every supported radiation tally still
need a fresh-context source-to-code check.

## Available evidence and remaining work

`tests/montecarlo/test_cone_split.py` checks primary-stream prefixes, copy-prefix
stability, per-primary aggregation and a three-seed cone/path-length statistic.
Those geometric tallies do not establish the issue's five-seed PXR/CBS,
bremsstrahlung and characteristic-yield acceptance.

`checks/detector_cone_variance_reduction.py` provides remote baseline and
comparison commands. No measured 5 MeV h-BN baseline, radiation-yield agreement,
FOM improvement or method-selection verdict is asserted here. Complete those
measurements through the remote workflow, compare the alternative angular
importance sampler, and independently verify the estimator before advancing
the ledger to `rederived`. Human sign-off remains separate in #277.

The comparison pilot scores full analytic line mass and a continuum band per
primary, including weighted copies. It bypasses grid resolution in both arms;
it does not measure full-spectrum runtime or characteristic-yield agreement.
The separate baseline retains production grid resolution and spectrum scoring.
The AMD batch driver is `checks/detector_cone_203_amd.sbatch`. Independent
verification is deferred; these runs do not promote the ledger status.

## Independent fresh-context re-derivation (2026-11-15)

Verifier context: fresh, did not write the implementation. Inputs read before
the derivation: the ledger row, the `elsepa-elastic-sampling` row (quantile
law), and the "Proposed estimator and source law" section above. The
implementation bodies were read only after the derivation in this section was
written.

### Cheap filters

- **Units.** $P_B$ is a probability, the half-arc $h$ and $\pi$ are radians,
  so $w=P_B h/\pi$ is dimensionless; $s=\min(1,U/w_t)$ is a ratio of
  dimensionless weights; $X_i$ carries the tally's units; $s_X/\sqrt N$ has
  the tally's units. Pass.
- **Ranges.** $0\le P_B\le1$ and $0\le h\le\pi$ give $0\le w\le1$; $0<s\le1$
  whenever $U>0$. Pass.
- **Limits** (derived below): splitting off gives $X_i=H_i$; $P_B=0$ gives
  $w=0$; $h=\pi$ gives $w=P_B$, the full polar-band mass. Pass.

### Analog law at one elastic event

At an elastic event with energy $E$ and incoming unit direction
$\hat{\mathbf d}$, the analog law draws $\xi\sim U(0,1)$, sets
$\mu=\mu(\xi,E)$ by the ELSEPA quantile map,
$\cos\theta=1-2\mu$, and draws the azimuth $\phi\sim U(0,2\pi)$ about
$\hat{\mathbf d}$. In the variables $(\xi,\phi)$ the analog density is

$$
p(\xi,\phi)=1\cdot\frac{1}{2\pi},\qquad (\xi,\phi)\in[0,1]\times[0,2\pi).
$$

Using $\xi$ rather than $\mu$ as the polar coordinate makes the likelihood
ratio independent of the tabulated DCS shape: only the quantile mass of the
band enters.

### (a) Cone geometry, polar band and azimuthal fraction

Let the detector cone have axis $\hat{\mathbf a}$ and half-angle $\alpha$, and
let $\beta=\angle(\hat{\mathbf d},\hat{\mathbf a})$. Measure $\phi$ from the
plane containing $\hat{\mathbf d}$ and $\hat{\mathbf a}$. The outgoing
direction $\hat{\mathbf d}'(\theta,\phi)$ satisfies

$$
\hat{\mathbf d}'\cdot\hat{\mathbf a}
=\cos\beta\cos\theta+\sin\beta\sin\theta\cos\phi .
$$

It lies in the cone iff $\hat{\mathbf d}'\cdot\hat{\mathbf a}\ge\cos\alpha$,
i.e. iff

$$
\cos\phi\ge c(\theta)\equiv
\frac{\cos\alpha-\cos\beta\cos\theta}{\sin\beta\sin\theta}
\qquad(\sin\beta\sin\theta>0).
$$

The in-cone azimuths at fixed $\theta$ are therefore one arc symmetric about
$\phi=0$, $\phi\in[-h,h]$, with

$$
h(\theta)=\arccos\bigl(\operatorname{clip}(c(\theta),-1,1)\bigr)\in[0,\pi].
$$

The arc's total length is $2h$, so its probability under the uniform azimuth
is

$$
\boxed{\frac{2h}{2\pi}=\frac{h}{\pi}}.
$$

The ledger's $h/\pi$ and the alternative $2h/(2\pi)$ are the same number;
$h/\pi$ is correct **provided $h$ is the half-width of the arc** (the arc is
$[-h,h]$, not $[0,h]$). A code that sampled $\phi$ on a width-$h$ arc, or
interpreted $h$ as a full width, would be off by a factor of two.

Degenerate cases: $c\le-1$ (whole circle inside the cone) gives $h=\pi$;
$c\ge1$ gives $h=0$. For $\sin\theta=0$ the outgoing direction is
$\pm\hat{\mathbf d}$ and is in the cone for all or no $\phi$ ($h=\pi$ or $0$).
For $\sin\beta=0$ the condition is independent of $\phi$, again $h\in\{0,\pi\}$.

The triangle inequality on the sphere gives
$\angle(\hat{\mathbf d}',\hat{\mathbf a})\ge\lvert\beta-\theta\rvert$, with
equality at $\phi=0$, so $h(\theta)>0$ exactly on the polar band

$$
\theta\in\bigl[\max(0,\beta-\alpha),\ \min(\pi,\beta+\alpha)\bigr],
$$

where $\lvert\beta-\theta\rvert\le\alpha$. (For $\beta<\alpha$ the lower edge
is $0$; near $\theta=\pi$ the reflex branch is covered by the same clip.)
Mapping to $\mu=(1-\cos\theta)/2$ (monotone) gives $[\mu_-,\mu_+]$, and by the
quantile law the band mass is

$$
\boxed{P_B=F(\mu_+;E)-F(\mu_-;E)=\xi_+-\xi_-},
$$

with $F$ the same interpolated CDF whose inverse the analog sampler uses. Any
covering interval $[\xi_-,\xi_+]\supseteq$ the exact band is acceptable, with
$P_B$ then the mass of the interval actually sampled; draws in the cover but
outside the band have $h=0$ and weight $0$.

### Proposal and likelihood ratio

The copy proposal is $\xi\sim U(\xi_-,\xi_+)$, then $\phi\sim U(-h,h)$ given
$\theta(\xi)$:

$$
q(\xi,\phi)=\frac{1}{P_B}\cdot\frac{1}{2h(\theta(\xi))},
\qquad
\frac{p}{q}=\frac{P_B\,2h}{2\pi}
=\boxed{P_B\,\frac{h(\theta)}{\pi}}.
$$

The support of $q$ is exactly the in-cone set $C$, so for any score $G$,

$$
\mathbb E_q\!\left[\frac{p}{q}G\right]
=\int_{\xi_-}^{\xi_+}\!\!d\xi\int_{-h}^{h}\!d\phi\;\frac{1}{2\pi}G
=\mathbb E_p\!\left[G\,\mathbf 1_C\right].
$$

The ratio depends on the sampled $\theta$ through $h(\theta)$; it is not the
constant $P_B$ unless $h\equiv\pi$ across the band.

### (b) Partition of the analog future (no double counting)

Write the analog score of one primary as
$H=H^{\rm pre}+H^{\rm post}$ around a split event, and
$H^{\rm post}=H^{\rm post}\mathbf 1_{\bar C}+H^{\rm post}\mathbf 1_C$, where
$\mathbf 1_C$ indicates that the **analog** deflection at this event landed in
the cone. The estimator replaces the second term only:

$$
\hat H=H^{\rm pre}+H^{\rm post}\mathbf 1_{\bar C}+w\,H_c ,
$$

where $H_c$ is the post-event score of an analog (unsplit) copy launched from
the event state (same position, energy, time and medium; elastic events leave
$E$ unchanged) with direction drawn from $q$. Taking expectations conditional
on the pre-event history, the copy term equals
$\mathbb E_p[H^{\rm post}\mathbf 1_C]$ by the identity above, so
$\mathbb E[\hat H]=\mathbb E[H]$.

This fixes the treatment of the analog primary: when its own sampled
deflection is in $C$, its post-event future must contribute **nothing**
(killed at the event, weight $0$, not kept with a reduced weight). Keeping it
would count $\mathbb E[H^{\rm post}\mathbf 1_C]$ twice; killing it whenever it
is in the band rather than the cone would drop the in-band, out-of-cone mass.
The kill test must use the same cone set $C$ as the proposal support. The
primary's pre-event path is scored once, by the primary only; $H_c$ must
start at the event.

The analog primary must sample its deflection exactly as without splitting,
and the split decision at each event must depend only on the state at that
event (non-anticipating). Repeating the construction at every elastic event
of the surviving out-of-cone primary is then unbiased by induction over the
(finite) event sequence: at each event the out-of-cone branch is the analog
law restricted to $\bar C$, whose future is itself estimated unbiasedly, and
copies run analog futures, $\mathbb E[H_c\mid\text{launch}]=V(\text{launch})$.
Copies that do not split again avoid weight nesting. Independence between
copy and primary random numbers is not required for unbiasedness, only for
the marginal laws above; it does affect variance.

### (c) Russian roulette

Decide survival with probability $s\in(0,1]$ that is fixed by the event
state (here $s=\min(1,U/w_t)$, $U$ a bound computed before or after sampling
the copy direction but not from the copy's future), using a uniform
independent of the copy's subsequent transport. A survivor carries $w/s$:

$$
\mathbb E[\text{roulette}\cdot w H_c]
= s\cdot\frac{w}{s}\,\mathbb E[H_c]+(1-s)\cdot 0
=w\,\mathbb E[H_c].
$$

Unbiased for any $s>0$; if $U$ may vanish only when $P_B=0$ (so $w=0$), the
$s=0$ case carries no mass. If $s$ is computed from the bound $U\ge w$, the
surviving weight satisfies $w/s=\max(w,\,w\,w_t/U)\le\max(1,w_t)$, which
bounds copy weights.

### (d) Per-primary grouping and standard error

Primaries have independent stream keys, and copy keys are deterministic
functions of (primary key, flight index), so the vectors
(primary path, its copies) are i.i.d. across $i$, and $X_i$ are i.i.d. with
$\mathbb E[X_i]=\mathbb E[H]$. Within a primary the copies share a parent
history and are correlated; treating copies as separate samples would
underestimate $\operatorname{Var}\bar X$. With weights bounded as above, a
finite number of elastic events per history and bounded tallies,
$\operatorname{Var}X_i<\infty$ and

$$
\mathrm{SE}(\bar X)=\frac{s_X}{\sqrt N},\qquad
s_X^2=\frac{1}{N-1}\sum_i (X_i-\bar X)^2
$$

is the standard (consistent) estimator. The ledger's `sample_std` must be
the $N-1$ form for the usual unbiased $s_X^2$; the $N$ form differs by
$\sqrt{(N-1)/N}$, negligible at production $N$ but a convention to pin.

### Limiting cases

- Splitting off: no kill, no copy, $X_i=H_i$; the analog stream must be
  unchanged, so outputs equal analog bit-for-bit only if splitting draws no
  numbers from the primary stream.
- $P_B=0$: $C$ has zero analog probability, so neither kill nor copy carries
  mass; no copy should be created.
- $h=\pi$ across the band (cone contains every azimuth, e.g. $\beta=0$,
  $\alpha\ge\theta_+$): $w=P_B$, and the proposal is the analog law
  conditioned on the band.
- $\alpha=\pi$ (cone is the whole sphere): $P_B=1$, $h=\pi$, $w=1$; the
  primary is always killed at its first event and replaced by one
  weight-$1$ analog copy, i.e. analog in distribution.

### Comparison with the implementation

Read after the derivation above was written:
`cone_split.py::{cone_split_event, _arc_half_width, _mu_of_xi, history_sums,
weighted_history_statistics, transport_cone_split}` and the
`cone_split` branch of the per-electron ELSEPA elastic event in
`cores.py::make_cpu_transport_core` (compiled by
`core_specializations.py::cone_split_core`).

| Derived term | Implementation | Agreement |
|---|---|---|
| Polar band $\lvert\theta-\beta\rvert\le\alpha$, clipped to $[0,\pi]$ | `theta_lo = max(0, theta_dn - theta_c)`, `theta_hi = min(pi, theta_dn + theta_c)`; `theta_c = arcsin(sin_c)` is exact because `ConeSplit` restricts $\alpha<\pi/2$ | match |
| $[\xi_-,\xi_+]$ = preimage of $[\mu_-,\mu_+]$ under the analog quantile map | 64-step bisection on `_mu_of_xi`, which is term-for-term `_sample_cos_theta_elsepa` (same `_elsepa_bracket`, same two `_elsepa_invert_row` calls, same `ln E` weight); $\mu(\xi)$ is a non-negative combination of monotone quantiles, so the preimage is one interval | match |
| $P_B=\xi_+-\xi_-$ | `p_band = xi_b - xi_a` | match |
| $c(\theta)$, $h=\arccos(\operatorname{clip}(c,-1,1))$, degenerate $h\in\{0,\pi\}$ | `_arc_half_width`: same $c$ (`c_star`), `<= -1` gives $\pi$, `>= 1` gives $0$; $\sin\theta\sin\beta\approx0$ tests $\cos\theta\cos\beta\ge\cos\alpha$ | match |
| $\phi\sim U(-h,h)$, $\phi=0$ toward $\hat{\mathbf a}$ | `psi = (2u - 1) * half`; `e1` is the unit projection of $\hat{\mathbf n}$ orthogonal to $\hat{\mathbf d}$, so $\hat{\mathbf d}'\cdot\hat{\mathbf n}=\cos\theta\cos\beta+\sin\theta\sin\beta\cos\psi$ | match; arc width $2h$, not $h$ |
| $w=P_B h/\pi$ | `p_band * half / np.pi` | match (the factor-two question resolves to $h/\pi$ with $h$ the half-width) |
| Roulette $s$ fixed by event state, $U\ge w$ | `keep = min(1, bound/target_weight)`; `bound = band_bound * arc_bound / pi` with `band_bound` $=\max_kF_k(\mu_+)-\min_kF_k(\mu_-)\ge P_B$ (quantile-interpolation bound) and `arc_bound` $=\arcsin(\sin\alpha/\sin\beta)$ for $\sin\beta>\sin\alpha$, else $\pi$. I verified $\max_\theta h(\theta)=\arcsin(\sin\alpha/\sin\beta)$ (minimum of $c(\theta)$ is $\sqrt{\sin^2\beta-\sin^2\alpha}/\sin\beta$); $\sin\beta\le\sin\alpha$ covers both $\beta\le\alpha$ and $\beta\ge\pi-\alpha$ (antipode of $\hat{\mathbf d}$ in the cone) | match |
| Survivor weight $w/s$; $U=0\Rightarrow$ no copy | `weight = p_band * half / pi / keep`; `bound <= 0` and `p_band <= 0` return no copy | match |
| Roulette uniform independent of copy direction and future | counters 0, 1, 2 of the (split key, flight) event key; copy transport keys re-hashed under a separate salt; no draw from the primary stream | match |
| Kill the analog future iff the analog outgoing direction is in $C$ | after the analog rotation, `dirs . n >= sp_cos_c` sets `EXIT_CONE_SPLIT`, keeps the pre-event segments, and ends the electron; the test is independent of whether a copy was made, and uses the same $C$ as the proposal support | match |
| Copy launched at the event state | position, energy, clock of the electron at the event; parent's cutoff and $t_0$; fresh flight optical depths are equal in law to the parent's remaining ones by memorylessness | match |
| Copies not split again | the copy generation is simulated without `_cone_split` | match |
| $X_i=H_i+\sum_c w_cH_c$ | `history_sums`: `bincount(parent, weights=w*x)` added to the primary values | match |
| $\mathrm{SE}=s_X/\sqrt N$, $N-1$ form | `weighted_history_statistics`: `std(ddof=1)/sqrt(n)` (reported relative to the mean) | match |

Numeric check (scratch script, not committed): a synthetic two-node
screened-Rutherford-shaped table ($a=2\times10^{-3}$ and $5\times10^{-4}$,
401-point $\mu$ grid, $E$ between nodes), cone $\alpha=0.05$ rad about
$\hat{\mathbf z}$, smooth score $G(\hat{\mathbf o})=1+0.5o_x-0.3o_z^2+2o_y$,
$2\times10^6$ calls of `cone_split_event` per case against a $20000\times8000$
midpoint quadrature of $\mathbb E_p[G\mathbf 1_C]$ under the analog sampler
`_sample_cos_theta_elsepa` with uniform azimuth:

| $\beta$ (rad) | quadrature | `cone_split_event` estimate | $z$ |
|---|---|---|---|
| $0$ | $2.295675\times10^{-1}$ | $(2.295679\pm0.00011)\times10^{-1}$ | $+0.03$ |
| $0.02$ | $2.148144\times10^{-1}$ | $(2.146878\pm0.00076)\times10^{-1}$ | $-1.66$ |
| $0.06$ | $1.217616\times10^{-1}$ | $(1.217341\pm0.00024)\times10^{-1}$ | $-1.14$ |
| $0.3$ | $1.065899\times10^{-3}$ | $(1.066040\pm0.00022)\times10^{-3}$ | $+0.63$ |
| $1.2$ | $5.342370\times10^{-6}$ | $(5.343851\pm0.0011)\times10^{-6}$ | $+1.33$ |
| $\pi-0.03$ | $3.601980\times10^{-7}$ | $(3.603839\pm0.0015)\times10^{-7}$ | $+1.23$ |

The quadrature itself carries a discretization error from the cone
indicator at the $10^{-4}$ relative level. With roulette active
(`target_weight` $=10^{-2}$) the estimates agreed with an independent analog
Monte Carlo within $1.8\sigma$ for all six geometries. No copy direction
fell outside the cone. These cover the in-band, out-of-band-forward
($\beta<\alpha$), oblique, and antipodal ($\beta>\pi-\alpha$) branches.

`tests/montecarlo/test_cone_split.py`: 7 passed (2026-11-15).

### Findings (no discrepancy)

- **Step budget.** A copy restarts the per-electron `max_steps` material-step
  budget; the analog future it replaces would have inherited the parent's
  used count. This changes the estimator only for histories that hit the step
  limit, which is a truncation regime, not physics.
- **Near-axis frame.** For $\sin\beta\le10^{-12}$ the azimuth frame is
  arbitrary while `_arc_half_width` still evaluates $c(\theta)$ down to
  $\sin\theta\sin\beta=10^{-300}$. The affected measure is
  $O(10^{-12})$; negligible.
- **Scope.** This verification covers the event-level change of measure,
  the analog/copy partition, roulette, and the per-history aggregation up to
  `history_sums`. It does not cover how any production radiation tally
  consumes copy weights (the row states no production weight support), nor
  coherent tallies.
- Event-key uniqueness across one electron's elastic events affects only
  correlation between copies, not unbiasedness, because each copy's draws
  are marginally uniform.

### Verdict

`rederived`. The likelihood ratio $P_Bh/\pi$ (with $h$ the arc half-width),
the kill-in-cone partition, roulette with surviving weight $w/s$, and the
per-primary $s_X/\sqrt N$ all match the independent derivation. This is not a
radiation-yield, FOM or human sign-off result.
