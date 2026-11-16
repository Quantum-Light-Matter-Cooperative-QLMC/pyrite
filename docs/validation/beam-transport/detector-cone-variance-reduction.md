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

## Efficiency options: energy gate and angle-scaled roulette

Both options are off by default; off, the core takes the paths above
unchanged. They are intended to preserve unbiasedness because each decision
is a function of the event state (energy, incoming direction) and of the
copy's own sampled direction, never of any future of the primary or the copy.
Both still need fresh-context verification.

- **Energy gate** (`ConeSplit.min_energy_keV`, $E_{\min}$). At an elastic
  event with $E<E_{\min}$ the core makes no copy and does not apply the kill
  test: the event is fully analog. The partition
  $H^{\rm post}=H^{\rm post}\mathbf 1_{\bar C}+H^{\rm post}\mathbf 1_C$ is then
  simply not applied at that event, so the event contributes its analog
  future. Kill and copy use the same gate, so no in-cone future is counted
  twice or dropped.
- **Angle-scaled roulette** (`ConeSplit.angle_scale`, $u_{\rm ref}$). After
  the copy direction $\hat{\mathbf o}$ is drawn with weight $w/s$, a second
  roulette keeps it with probability

  $$
  s_u=\min\!\left(1,\frac{w/s}{w_t\,\max\!\left(1,(u/u_{\rm ref})^2\right)}\right),
  \qquad u=\gamma\,\angle(\hat{\mathbf o},\hat{\mathbf a}),
  $$

  and a survivor carries $w/(s\,s_u)$. Conditional on $\hat{\mathbf o}$,
  $\mathbb E[\text{survive}\cdot w/(s\,s_u)]=w/s$, so the event-level identity
  holds unchanged. The uniform is counter 3 of the event key, independent of
  counters 0–2. The $u^2$ target follows the line mass measured in job 13230,
  roughly $\propto u^{-2}$ outside $1/\gamma$
  ([research record](../../research/beam-transport/detector-cone-line-variance.md)),
  so expected weighted copy contributions are roughly equal across the cone.

`tests/montecarlo/test_cone_split_weights.py` checks the angle roulette
against isotropic cap integrals; `tests/montecarlo/test_cone_split.py`
checks that a gate above the beam energy reproduces analog arrays and that
both options together keep cone and path-length tallies within statistical
error of analog.

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

## Fresh-context verification of the efficiency options (2026-11-16)

Verifier context: fresh, did not write the implementation. Scope: only the
two opt-in options, the energy gate `ConeSplit.min_energy_keV` ($E_{\min}$)
and the angle-scaled roulette `ConeSplit.angle_scale` ($u_{\rm ref}$). The
base estimator (likelihood ratio $P_Bh/\pi$, kill-in-cone partition, first
roulette $s=\min(1,U/w_t)$, per-history sums) is taken as established by the
independent re-derivation of 2026-11-15 above and is not redone. Inputs read before the derivation: the ledger row, the
section "Efficiency options: energy gate and angle-scaled roulette" above, and
that base derivation. The implementation bodies were read only after the
derivation below was written.

### Cheap filters

- **Units.** $E$ and $E_{\min}$ are kinetic energies in keV, so the gate
  compares like quantities. $\gamma=1+E/(m_ec^2)$ is dimensionless and the
  angle $\vartheta$ is in radians, so $u=\gamma\vartheta$ and $u_{\rm ref}$
  share units and $u/u_{\rm ref}$ is dimensionless. $s_u$ is a ratio of
  dimensionless weights. Pass.
- **Ranges.** A made copy has $w/s>0$, and $\max(1,(u/u_{\rm ref})^2)\ge1$,
  so $0<s_u\le1$ for any $u_{\rm ref}>0$. No positive-mass copy is ever given
  survival probability zero. The copy lies in the cone, so
  $\vartheta\le\alpha<\pi/2$ and $u\le\gamma\alpha$ is finite. Pass.
- **Limits.** $E_{\min}=0$: the gate never fires. $E_{\min}>E_0$: electron
  kinetic energy never increases in transport, so every elastic event is
  gated and the run is analog. $u_{\rm ref}\to\infty$: the formula's limit is
  not the base estimator; only the disabled path is (finding below). Pass
  with that caveat.

### Derivation: energy gate

Let $g(\mathcal S)\in\{0,1\}$ be a gate evaluated on the pre-deflection event
state $\mathcal S$ (here $g=\mathbf 1[E\ge E_{\min}]$). Allow separate gates,
$g_k$ for the kill and $g_c$ for the copy:

$$
\hat H=H^{\rm pre}+H^{\rm post}\bigl(1-g_k\mathbf 1_C\bigr)+g_c\,w\,H_c .
$$

Conditional on the pre-event history, which fixes $\mathcal S$ and hence
$g_k$ and $g_c$, the base identity
$\mathbb E[wH_c\mid\mathcal S]=\mathbb E[H^{\rm post}\mathbf 1_C\mid\mathcal S]$
gives

$$
\mathbb E[\hat H\mid\mathcal S]-\mathbb E[H\mid\mathcal S]
=(g_c-g_k)\,\mathbb E\!\left[H^{\rm post}\mathbf 1_C\mid\mathcal S\right].
$$

The estimator is therefore unbiased for any gate **iff** $g_c=g_k$ at every
event with nonzero in-cone mass. Mismatched gates bias in a definite
direction. Kill without copy ($g_k=1$, $g_c=0$) drops the in-cone future
(low bias). Copy without kill ($g_c=1$, $g_k=0$) counts it twice (high bias).
With $g_c=g_k=0$ the event is fully analog; with $g_c=g_k=1$ it is the base
event.

The gate must be a function of $\mathcal S$ (energy, position, incoming
direction, element) only. It must not depend on anything drawn at or after
the event's analog deflection, or on any future. A gate that read the analog
outgoing direction would correlate $g_k$ with $\mathbf 1_C$ and break the
conditional identity. The gate must use the energy at the event. An elastic
event leaves $E$ unchanged, so the kill test and the copy see the same $E$.
The induction over events in the base derivation then goes through
unchanged: a gated event contributes its analog future, which later
(possibly ungated) events estimate unbiasedly. The gate need not be monotone
in energy; any event-state gate is unbiased.

For "a gate above the beam energy reproduces analog" to hold bit for bit,
not just in law, the gate must not change the primary's stream consumption.
Split decisions draw from a separate event key, so a gated event consumes the
same primary draws as an ungated one.

### Derivation: angle-scaled roulette

After the first roulette survives and the direction $\hat{\mathbf o}$ is drawn
from $q$, the copy carries $w_1=w/s$ with $w=P_Bh(\theta)/\pi$. A second
roulette keeps it with probability $s_u(\hat{\mathbf o},\mathcal S)\in(0,1]$,
using a uniform $V$, and a survivor carries $w_1/s_u$. Suppose $V$ is
independent of $\hat{\mathbf o}$, of the first-roulette uniform and of the
copy's future. Then, conditional on $(\mathcal S,\hat{\mathbf o})$,

$$
\mathbb E\!\left[\mathbf 1[V<s_u]\,\frac{w_1}{s_u}H_c\,\Big|\,\mathcal S,\hat{\mathbf o}\right]
=s_u\cdot\frac{w_1}{s_u}\,\mathbb E[H_c\mid\mathcal S,\hat{\mathbf o}]
=w_1\,\mathbb E[H_c\mid\mathcal S,\hat{\mathbf o}].
$$

Integrating over $\hat{\mathbf o}\sim q$ returns the base event identity. So
$s_u$ may depend on the sampled direction: the compensating factor $1/s_u$ is
evaluated at the same $\hat{\mathbf o}$, and the correction is exact
pointwise. The requirements are only:

1. $s_u>0$ wherever $w_1H_c$ can be nonzero. This holds: $s_u>0$ for every
   made copy.
2. $s_u$ is fixed before the copy's future is simulated. It uses only
   $\mathcal S$, $w_1$ and $\hat{\mathbf o}$.
3. $V$ is uniform and independent of the draws that produced
   $\hat{\mathbf o}$ and of the first-roulette uniform. If $V$ reused the
   polar or azimuth uniform, $\Pr(\text{survive}\mid\hat{\mathbf o})$ would be
   $0$ or $1$ instead of $s_u$, and the estimator would be biased. The same
   holds for the first-roulette uniform, which is no longer $U(0,1)$ once
   conditioned on survival. Hence $V$ needs its own counter (counter 3) of
   the event key, with counters 0–2 used by roulette 1, the polar quantile
   and the azimuth.

The choice of $\gamma$ (here at the event energy, which is also the copy's
launch energy) and of the exponent $2$ affects efficiency only; any positive
event-state function would be unbiased. Copies are not split again, so this
roulette acts exactly once per copy and no weight nesting arises.

Weight bound. The survivor weight is

$$
w_2=\frac{w_1}{s_u}=\max\!\bigl(w_1,\;w_t\,m(u)\bigr),\qquad
m(u)=\max\!\left(1,\Bigl(\frac{u}{u_{\rm ref}}\Bigr)^2\right)
\le\max\!\left(1,\Bigl(\frac{\gamma\alpha}{u_{\rm ref}}\Bigr)^2\right).
$$

With the base bound $w_1\le\max(1,w_t)$, this gives

$$
w_2\le\max\!\left(1,\;w_t,\;w_t\Bigl(\frac{\gamma_0\alpha}{u_{\rm ref}}\Bigr)^2\right),
$$

where $\gamma_0$ is taken at the beam energy. Weights stay bounded and the
per-history variance stays finite, but the bound grows as $\gamma_0^2$. At
MeV energies and small $u_{\rm ref}$, off-axis copies can carry weights
$\gg1$. That is an efficiency concern, not a bias.

### Limiting cases

- $E_{\min}=0$, $u_{\rm ref}=\infty$ (disabled): this reproduces the base
  estimator bit for bit only if the disabled angle path is skipped rather
  than evaluated. At $u_{\rm ref}\to\infty$ the formula gives $m=1$ and
  $s_u=\min(1,w_1/w_t)$. That is a nontrivial weight-window roulette
  whenever $w_1<w_t$, for example when roulette 1 has $s=1$ (because
  $U\ge w_t$) but the sampled $w<w_t$. The limit is unbiased, but it is not
  the base estimator.
- $E_{\min}>E_0$: every event is gated, with no copies and no kills, so the
  arrays are analog.
- $u\le u_{\rm ref}$ throughout the cone ($\gamma_0\alpha\le u_{\rm ref}$):
  the second roulette reduces to the plain weight window against $w_t$.

### Comparison with the implementation

Read after the derivation above:
`cone_split.py::{ConeSplit, ConeSplit.policy, cone_split_event,
_ConeSplitPass.core_args}` and the cone-split branch of the per-electron
ELSEPA elastic event in `cores.py` (`split_here`, `sp_policy`).

| Derived term | Implementation | Agreement |
|---|---|---|
| Gate on event state; $E<E_{\min}$ is analog | `cores.py:1189` `split_here = cone_split and elastic_model_code == 2 and E_keV[e] >= sp_policy[1]`, computed after the analog `cos_t` sample but from `E_keV[e]` only (pre-event state); the analog draws `cos_u`, `phi_u` come first and do not depend on the gate | match |
| $g_c=g_k$ | the same `split_here` guards the `cone_split_event` call (`cores.py:1190`) and the kill test (`cores.py:1214-1216`); `E_keV[e]` is not modified between them | match |
| Gate leaves the primary stream alone | gated events skip only the event-key draws; primary `draws[e]` advance identically | match |
| $E_{\min}=0$ disables the gate | `E_keV >= 0.0` is always true for physical energies; `ConeSplit` rejects negative or non-finite `min_energy_keV` | match |
| Disabled angle path skipped, not evaluated | `policy()` maps `angle_scale = inf` to `policy[2] = 0.0` (`cone_split.py:107-110`); `cone_split_event` applies the second roulette only `if u_ref > 0.0` (`cone_split.py:293-301`); `angle_scale <= 0` is rejected | match |
| $u=\gamma(E)\,\angle(\hat{\mathbf o},\hat{\mathbf a})$ | `(1.0 + E_keV / _MC2_KEV) * np.arccos(cos_on)`, with `cos_on` clipped, `_MC2_KEV = 510.99895` keV and `E_keV` the event energy (also the copy's launch energy) | match |
| $s_u=\min\bigl(1,w_1/(w_t\max(1,(u/u_{\rm ref})^2))\bigr)$ | `u` is already divided by `u_ref`; `keep_u = min(1.0, weight / (target_weight * max(1.0, u * u)))`, with `weight` $=P_Bh/\pi/s$ | match |
| $V$ independent of counters 0–2 | `_stream_uniform_scalar(event_key, np.uint64(3))`; counters 0, 1 and 2 are roulette 1 (`cone_split.py:236`), the polar quantile $\xi$ and the azimuth $\psi$ | match |
| Survivor weight $w_1/s_u$ | `weight = weight / keep_u` | match |
| Applied once per copy | copies run without `_cone_split`, so neither the gate nor either roulette acts on a copy | match |

### Numeric checks

Tests (2026-11-16, worktree HEAD `25856eece`):
`tests/montecarlo/test_cone_split.py` and
`tests/montecarlo/test_cone_split_weights.py`, 21 passed.

Scratch check (not committed). The check calls `cone_split_event` directly
with an isotropic two-node table (the law the weight test uses) and a cone of
$\alpha=0.3$ rad about $\hat{\mathbf z}$. It compares against the exact
analog integrals $P_C=(1-\cos\alpha)/2=2.233176\times10^{-2}$ and
$\mathbb E_p[G\mathbf 1_C]=7.415928\times10^{-3}$. The score
$G=(\vartheta/\alpha)^4$ is weighted toward the cone edge, where $u$ is
largest and the angle roulette thins copies most. The sweep covers incident
directions $\beta=0$, $0.2$ and $\pi/2$; $E=50$ keV ($\gamma=1.098$) and
$5$ MeV ($\gamma=10.78$); $w_t\in\{10^{-3},0.2\}$; and
$u_{\rm ref}\in\{\text{off},1,0.1,10^{300}\}$, with $2\times10^6$ events per
case. All 48 cases fall within $|z|\le3.0$ for both tallies. The sweep reused
event keys across cases, so its $z$ values are correlated. The worst group
($5$ MeV, $u_{\rm ref}=0.1$, $w_t=10^{-3}$, $z$ down to $-3.0$) was rerun
with independent keys and $10^7$ events per row:

| $\beta$ | $w_t$ | $z(P_C)$ | $z(\mathbb E[G\mathbf 1_C])$ |
|---|---|---|---|
| $0$ | $10^{-3}$ | $+0.03$ | $+0.29$ |
| $0$ | $0.2$ | $-0.78$ | $-0.54$ |
| $0.2$ | $10^{-3}$ | $-1.68$ | $-1.49$ |
| $0.2$ | $0.2$ | $-0.60$ | $-1.32$ |
| $\pi/2$ | $10^{-3}$ | $+0.51$ | $+0.89$ |
| $\pi/2$ | $0.2$ | $-0.05$ | $-0.25$ |

In every active case the maximum survivor weight matched the derived bound
$w_t\max(1,(\gamma\alpha/u_{\rm ref})^2)$: for example $2.169$ against
$2.169$ ($50$ keV, $u_{\rm ref}=0.1$, $w_t=0.2$), and $209.3$ against
$209.4$ ($5$ MeV, $u_{\rm ref}=0.1$, $w_t=0.2$). With
$u_{\rm ref}=10^{300}$ the copy fraction differs from the disabled path, for
example $0.1116$ against $0.3061$ at $\beta=0.2$, $w_t=0.2$. This confirms
the limiting-case finding; the estimates remain unbiased.

### Findings (no discrepancy)

- **A large finite $u_{\rm ref}$ is not "off".** Only `angle_scale = inf`
  (`policy[2] = 0`) disables the second roulette. Any finite value, however
  large, still applies $s_u=\min(1,w_1/w_t)$, a weight window that removes
  low-weight copies near the axis. This is unbiased, but the formula's
  $u_{\rm ref}\to\infty$ limit does not join continuously onto the disabled
  path. The ledger row and the `ConeSplit` docstring correctly say "inf
  disables"; readers should not assume continuity.
- **Weight growth at MeV energies.** Survivor weights reach
  $w_t(\gamma_0\alpha/u_{\rm ref})^2$, about $209$ at $5$ MeV with
  $\alpha=0.3$, $u_{\rm ref}=0.1$ and $w_t=0.2$. The weights stay bounded, so
  the variance is finite, but rare heavy copies can dominate the per-history
  variance. This is an efficiency question for the FOM study, not a bias.
- **Gate placement.** `split_here` is computed after the analog `cos_t`
  sample but depends only on `E_keV[e]`, so the gate is non-anticipating. A
  future edit that let the gate read the post-rotation direction would break
  unbiasedness: $g_k=g_c$ must be a function of pre-deflection state only.
- **Counter independence.** The derivation needs $V$ to be independent of
  counters 0–2. The implementation relies on the splitmix64 counter-based
  hash for this, the same assumption every per-electron stream already makes,
  so it adds no new assumption.

### Verdict

`rederived` for both efficiency options. The energy gate is a single
event-state predicate shared by copy and kill, so each event either applies
the full analog/copy partition or skips it entirely. The angle roulette is a
direction-conditional Russian roulette with an independent uniform and an
exactly compensating weight $w/(s\,s_u)$. The defaults
`min_energy_keV = 0.0` and `angle_scale = inf` (`policy[2] = 0`) follow the
base path exactly. This is not a radiation-yield, FOM or human sign-off
result.

Suggested ledger change: this section alone does not change the status; the
task owner decides. Optionally add to the row's limiting cases: "only
`angle_scale = inf` disables the angle roulette; a finite `u_ref` always
applies the `w_t` weight window".
