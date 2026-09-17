# Line-spectrum error budget

Issue #101. How the accuracy tolerances that gate a PyRITE line spectrum are
divided among the independent error terms that produce it, and which measured
result bounds each term.

This page allocates; it derives no physics and certifies no equation. It has no
ledger row of its own. Every number it quotes is either a code constant, named
with its defining symbol, or a measurement that belongs to another row, cited to
that row. Where a term is unmeasured it says so instead of assigning it a
plausible value.

Related: [`line-grid-sinc-convergence`](line-grid-sinc-convergence.md) (uniform
resolution and backend precision),
[`line-absorption-tabulation`](../radiation-physics/line-absorption-tabulation.md)
(interpolation), [`energy-step-convergence`](energy-step-convergence.md) and
[`radiation-error-estimators`](radiation-error-estimators.md) (transport).

## Decomposition

A reported line observable $q$ — integrated yield, centroid, dominant-line
FWHM, or a detected count rate — carries five independent error terms plus a
backend-precision term:

```{math}
:label: eq-line-budget-decomposition

\left|\frac{\hat q - q}{q}\right| \;\le\;
\underbrace{\varepsilon_\mathrm{band}}_{\text{bandwidth}}
+ \underbrace{\varepsilon_\mathrm{quad}}_{\text{quadrature}}
+ \underbrace{\varepsilon_\mathrm{interp}}_{\text{interpolation}}
+ \underbrace{\varepsilon_\mathrm{fp}}_{\text{backend precision}}
+ \underbrace{\varepsilon_\mathrm{trans}}_{\text{transport}}
+ \underbrace{\varepsilon_\mathrm{stat}}_{\text{Monte Carlo}} .
```

The sum is linear, not in quadrature. Only $\varepsilon_\mathrm{stat}$ is
random; the other five are deterministic biases of one case and may align, so a
root-sum-square would understate the worst case. Bandwidth truncation and
quadrature aliasing in particular have fixed signs — truncation always removes
yield, and an under-sampled $\operatorname{sinc}^2$ comb aliases into the same
bins on every rung of a ladder.

**The first four terms are grid error; the last two are not.** That split is
what makes the budget measurable. The refinement ladder of
`energy_grid/convergence.py` evaluates every rung on one fixed set of transport
segments and proves the identity with a segment fingerprint, so
$\varepsilon_\mathrm{trans}$ and $\varepsilon_\mathrm{stat}$ are held exactly
constant across rungs and cancel in every rung-to-rung difference. What the
ladder gates is therefore
$\varepsilon_\mathrm{band} + \varepsilon_\mathrm{quad} + \varepsilon_\mathrm{interp} + \varepsilon_\mathrm{fp}$,
and that is the quantity the production tolerances of
`pyrite._line_grid_policy.DEFAULT_RTOL` bound. Those tolerances are the **grid
share of the budget, not the total error of a spectrum**: at production electron
counts $\varepsilon_\mathrm{stat}$ is usually the largest single term in
{eq}`eq-line-budget-decomposition`.

## Allocation of the grid share

```{list-table} Grid-error allocation per observable class. The class totals are the production tolerances of DEFAULT_RTOL (intrinsic source, detected counts) and the harness-only convergence.SHAPE_RTOL.
:name: tbl-line-budget-allocation
:header-rows: 1

* - Term
  - Intrinsic source (total $10^{-3}$)
  - Detected counts (total $10^{-2}$)
  - Shape (harness, total $10^{-2}$)
  - Controlled by
* - bandwidth
  - $0$
  - $0$
  - $0$
  - `kinematic_line_stop_eV`, `line_start_eV`
* - quadrature, backbone
  - $2\times10^{-4}$
  - $2\times10^{-3}$
  - $2\times10^{-3}$
  - `sinc_feature_spacing`, `DEFAULT_MAX_SPACING_EV`
* - quadrature, feature windows
  - $5\times10^{-4}$
  - $5\times10^{-3}$
  - $5\times10^{-3}$
  - `samples_per_feature`, `tail_widths`
* - interpolation
  - $2\times10^{-4}$
  - $2\times10^{-3}$
  - $2\times10^{-3}$
  - `_interp_elemental_mu`, detector response resampling
* - backend precision
  - $1\times10^{-4}$
  - $1\times10^{-3}$
  - $1\times10^{-3}$
  - `DEFAULT_BACKEND_SAFETY_ULPS`, `PYRITE_FP64`
```

The shares are a policy split of a policy tolerance. They are not derived from
any physical law: they encode that the windowed quadrature term is the one the
grid policy actively trades against cost, so it gets the largest share, and that
the three terms already measured far inside their shares get the remainder. The
detected-counts and shape columns are the intrinsic-source column times ten,
matching the ratio already fixed by `DEFAULT_RTOL`.

## The terms

### Bandwidth — $\varepsilon_\mathrm{band}$

Intensity outside $[\texttt{start}, \texttt{stop}]$ is not sampled at all, so
bandwidth error is pure truncation loss, always of one sign.

The automatic policy sets the upper edge from the closed-form direction-maximized
resonance bound of the ledgered
[`line-grid-kinematic-bandwidth`](../ledger-core-coherent-physics.md#line-grid-kinematic-bandwidth)
row,

$$
E_\mathrm{res} \le \frac{\hbar c\,\beta\,|\mathbf g|_\mathrm{max}}{1 - \beta},
$$

evaluated at the incident speed over the case's own reflection set. No PXR/CBS
line of the case can exist above it, so **coherent bandwidth truncation is zero
by construction** and the term consumes none of the budget. This is a bound and
not an estimate: it is deliberately loose, and pays for its exactness in points,
not in accuracy.

Two qualifications bound the claim:

- **`coverage-0.95` forfeits the entire budget.** The stored catalog artifacts
  carry a *bandwidth* policy that stops where 95% of integrated coherent-line
  intensity has accumulated (`DEFAULT_BANDWIDTH_COVERAGE`). Its truncation is
  $5\times10^{-2}$ by definition — fifty times the whole intrinsic-source grid
  budget, five times the detected-counts budget. A run on a legacy
  `coverage-0.95` grid is outside this budget entirely and cannot be brought
  inside it by refinement. `COVERAGE_BANDWIDTH_POLICY` is a fallback for
  reproducing stored results, never an accuracy statement.
- **Characteristic lines are reported, not budgeted.** A characteristic line
  centred inside the window still loses its Lorentzian tails past the grid
  edges. Since #88 that loss is exact data rather than a silent bias:
  `characteristic.py::characteristic_line_window_mass` returns each line's
  captured/truncated split, and `mc_characteristic_spectrum` warns when an
  in-window line is truncated below 50% of its mass by the grid's own edges. A
  consumer that needs characteristic yield inside this budget must read that
  split; the 50% warn threshold is a loudness floor, not a budget share.

### Quadrature — $\varepsilon_\mathrm{quad}$

The dominant controllable term, and the one the window plan exists to buy down.
The line kernels evaluate the finite-time lineshape of the ledgered
`finite-time-lineshape` row,

$$
\frac{d^2N}{dE\,d\Omega} \propto t_L^2\,
\operatorname{sinc}^2\!\left(\frac{a_w\,(E - E_\mathrm{res})}{\pi}\right),
\qquad
a_w = \frac{(1 - \boldsymbol\beta\cdot\hat{\mathbf n})\,t_L}{2\hbar c},
$$

and the spectrum is the trapezoid rule over the grid's own coordinates. The
feature is band limited: the trapezoid rule is exact for $h \le \pi/a_w$, the
first-zero step returned by `sinc_feature_spacing`, and aliases above it. The
width $\pi/a_w$ carries no dependence on the photon energy $E$, which is why the
axis is piecewise uniform and not logarithmic.

The term splits along the window plan:

- **Backbone.** The uniform piece spanning $[\texttt{start}, \texttt{stop}]$ at
  `DEFAULT_MAX_SPACING_EV` $=3$ eV, carrying the continuum between features.
  Measured under uniform refinement by
  [`line-grid-sinc-convergence`](line-grid-sinc-convergence.md): at the derived
  `sinc-nyquist` spacing, hopg yield error is at most $7.7\times10^{-5}$ and
  centroid shift at most $0.017$ eV, inside the $2\times10^{-4}$ share. The same
  measurement shows the fixed 3 eV catalog rows carrying 0.12–4.7% yield error
  and up to 30 eV centroid shift — three to four orders outside the share —
  which is the evidence that the backbone alone cannot hold the budget where
  features are unresolved.
- **Feature windows.** The fine uniform pieces the planner
  (`_line_windows.build_window_plan`) lays over deterministically seeded
  features. The starting heuristic is
  `line_seeds.DEFAULT_SAMPLES_PER_FEATURE` $=8$ nodes across the narrowest
  shape-bearing feature, with `DEFAULT_TAIL_WIDTHS` $=2$ sinc widths of tail on
  each side. **The heuristic is not the certificate.** The window-refinement
  ladder (`convergence_case.window_ladder`, submitted by
  `convergence_job start-windows`) refines samples per feature on a deliberately
  fixed backbone and judges each observable under the Richardson gate of
  {eq}`eq-line-grid-richardson-gate`, then compares the finest windowed rung
  against a dense uniform reference. The $5\times10^{-4}$ share is what that
  comparison must meet. It is met in all twelve measured cases
  ({numref}`tbl-line-budget-window-measured`).

Absorption-edge windows are a deliberate exception inside this term: their
spacing follows the Chantler table's own node density and does not refine with
`samples_per_feature`, so the ladder's Richardson gate measures kinematic and
characteristic window convergence only. Edge-region error appears in the
dense-reference comparison instead. The edges are not optional seeds — 95% of
30 keV coherent-line intensity lands in one 3 eV bin at the carbon K edge, where
$\mu$ jumps by a factor of 18.

#### Measured window quadrature

Campaign of 2026-09-16: `convergence_job start-windows` at $N_e = 2000$ through a
1 mm slab, samples per feature 2/4/8/16/32 on a fixed 3 eV backbone, all three
seed providers, two tail widths, against a uniform reference capped at 400 001
points. Twelve cases — hopg and wse2, 30/100/300 keV, tilts 5 and 85 deg — each
evaluated on one transport, every rung sharing its segment fingerprint.

**Every observable of every case meets its share.** Worst case across the twelve,
against the dense uniform reference:

```{list-table} Worst relative difference between the finest windowed rung and the dense uniform reference, over twelve cases, against the window row of tbl-line-budget-allocation.
:name: tbl-line-budget-window-measured
:header-rows: 1

* - Observable
  - Worst relative difference
  - Case
  - Window share
* - yield
  - $1.4\times10^{-4}$
  - hopg 300 keV, tilt 5
  - $5\times10^{-4}$
* - centroid
  - $1.3\times10^{-4}$
  - hopg 300 keV, tilt 85
  - $5\times10^{-4}$
* - dominant-line FWHM
  - $8.7\times10^{-4}$
  - hopg 300 keV, tilt 85
  - $5\times10^{-3}$
* - line/background
  - $7.1\times10^{-4}$
  - wse2 100 keV, tilt 85
  - $5\times10^{-3}$
* - Timepix3 counts
  - $2.6\times10^{-4}$
  - hopg 30 keV, tilt 5
  - $5\times10^{-3}$
* - EagleXO counts
  - $1.2\times10^{-4}$
  - wse2 30 keV, tilt 85
  - $5\times10^{-3}$
```

Three results qualify that headline.

**The point saving falls with beam energy, and the windows degenerate toward
uniform.** Measured against a uniform grid at the finest window's own spacing
over the same bandwidth, the windowed axis costs 10.4x and 13.4x fewer points at
hopg 30 keV, 5.1x and 5.6x at 100 keV, and only 2.5x and 2.9x at 300 keV; wse2
runs 5.6x down to 1.9x over the same range. The issue's ~10x estimate therefore
holds at 30 keV and not at 300 keV. The mechanism is visible in the plans: the
kinematic seed spans the weighted quantile range of $E_\mathrm{res}$ over the
case's own segments, and multiple scattering broadens that population with
energy, so windows widen and merge. At hopg 100 keV tilt 5 they merge to cover
the axis outright — the plan's largest spacing is 0.01 eV, not the 3 eV backbone,
which is a windowed grid that has become a fine uniform one. Accuracy is
unaffected; the cost argument for windows is what erodes.

**One case accepts no spacing under the Richardson gate while agreeing with the
reference to $2.7\times10^{-8}$.** wse2 100 keV tilt 85 converges in yield,
centroid, FWHM and both detector counts, but its line/background ratio never
passes {eq}`eq-line-grid-richardson-gate` anywhere on the ladder. The gate is a
statement about rung-to-rung stability, not about distance from truth, and a
shape observable riding on a small background can be unstable while the
integrated quantities are exact. The same shape-class limitation that
`line-grid-sinc-convergence` found under uniform refinement survives windowing.

**Why `samples_per_feature` means what it says — and the fringe question.**
`sinc_feature_spacing` returns the width at the $\varepsilon$-weighted *lower
quantile*, not the narrowest width, so the window spacing
$w(\varepsilon)/\texttt{samples}$ nominally resolves a quantile feature and
leaves $\varepsilon$ of $t_L^2$-weight aliased. Whether that still puts eight
nodes across the *narrowest* shape-bearing feature — what the issue's plan
actually asks for — depends on the tail of the width distribution, which is a
property of the transport rather than of the grid policy.

Measured directly over each case's own segments (hopg 30/100 keV, wse2 30 keV,
$N_e = 60$ and 240):

```{list-table} Width at the intrinsic-source quantile against the narrowest feature of the same segment population.
:name: tbl-line-budget-width-tail
:header-rows: 1

* - Case
  - $w(10^{-3})$
  - narrowest
  - ratio
* - hopg 30 keV, tilt 5, $N_e=60$
  - 0.915 eV
  - 0.915 eV
  - 1.00
* - hopg 30 keV, tilt 5, $N_e=240$
  - 1.095 eV
  - 0.983 eV
  - 1.11
* - hopg 100 keV, tilt 5
  - 0.403 eV
  - 0.403 eV
  - 1.00
* - wse2 30 keV, tilt 85
  - 7.40 eV
  - 7.40 eV
  - 1.00
```

The width distribution is bounded below rather than heavy tailed: with
$\pi/a_w \propto 1/\big((1 - \boldsymbol\beta\cdot\hat{\mathbf n})\,t_L\big)$,
the longest single flight floors $t_L$, and the elastic mean free path gives
flight lengths no long tail. So the quantile width *is* the narrowest width to
within 11%, the nominal eight nodes do land across the narrowest feature, and
the interference fringes inside a window — the $\operatorname{sinc}^2$ sidelobes,
whose period is the same $\pi/a_w$ as the main lobe — are sampled at the same
eight nodes per period.

Two consequences. First, **fringe resolution inside windows is already separate
from backbone continuum spacing** by construction: window spacing runs
0.03–0.9 eV against a fixed 3 eV backbone. A *further* separate fringe control,
a finer quantile used only inside windows, would buy at most 11% finer spacing
at proportionally more points, and nothing at all in three of the four cases
measured — the same shape of argument that retired the $\mathrm{ulp}/\mathrm{rtol}$
spacing floor in `line-grid-sinc-convergence`. It is not warranted by these
data. Second, because this is a measured property of four cases and not a
theorem, `kinematic_line_seeds` reports `narrowest_feature_width_eV` and
`samples_at_narrowest` in its seed summary, so a case whose quantile drifts
above its narrowest feature shows it rather than under-resolving silently.

This covers the incoherent route only. The coherent route sums complex
amplitudes before squaring, where the fringe period is set by phase differences
across segments rather than by any single $\pi/a_w$; that is #117's scope and
remains open.

**The hopg 300 keV ladder is truncated.** At 16 and 32 samples per feature the
plan asks for 630 280 and 1 260 300 points, over the 400 000-point budget, so
those two rungs were dropped and recorded in `skipped_samples`. The 300 keV
acceptance therefore rests on the 2/4/8 triple only, and the reference comparison
above is against a windowed rung at 8 samples per feature rather than 32.

### Interpolation — $\varepsilon_\mathrm{interp}$

Two distinct resamplings, both onto the line axis.

**Self-absorption.** $\tau = L_\mathrm{esc}\,\mu(E_\mathrm{res})$ reads per-element
$\log \mu_i$ tables interpolated linearly in $\log E$ and summed as
$\mu = \sum_i \mu_i$ ([`line-absorption-tabulation`](../radiation-physics/line-absorption-tabulation.md),
status `rederived`). Measured against direct xraydb evaluation: $\le 2\times10^{-12}$
relative in float64 and $\le 1\times10^{-4}$ in float32 on HOPG. The float64 figure
is eight orders inside the $2\times10^{-4}$ share; the float32 figure sits at half
of it, which is one more reason the backend-precision term below is kept
separate rather than folded in here. That row's open item — a spectrum-level
exact-versus-tabulated A/B — is the missing evidence that the point-wise bound
propagates to an integrated observable.

**Detector response.** Detected counts convolve the line density with a response
whose input channels are coarser than the line axis, so refining the grid moves
the response's own sampling. `line-grid-sinc-convergence` measures Timepix3
counts drifting by up to $10^{-3}$ per halving from that effect alone. Against
the detected-counts share of $2\times10^{-3}$ that is a factor of two of headroom
and no more, and it is why `convergence.NOISE_FRACTION` exists: rung-to-rung
changes at or below 10% of the tolerance are treated as resampling noise rather
than as non-monotone convergence.

### Backend precision — $\varepsilon_\mathrm{fp}$

Float32 coordinates quantize the line axis, and locally fine windows are where
that first bites. Two guards, one floor:

- `_grid_semantics.validate_backend_coordinates` judges **each interval at its
  own ulp** against `DEFAULT_BACKEND_SAFETY_ULPS` $=8$, the interval-wise
  counterpart of the uniform `validate_backend_spacing`. A plan that would
  collapse nodes after the cast is refused, never silently coarsened.
- `_setup.py`'s post-cast monotonicity check (#111) remains the final gate.

Measured share consumption, on identical segments (wse2 300 keV, 1.22 M
segments): float32 deviates from FP64 by at most $3.4\times10^{-6}$ in yield,
$2.3\times10^{-5}$ in dominant-line FWHM, 0.39 ulp in centroid, and
$5.8\times10^{-5}$ pointwise. Against the $1\times10^{-4}$ intrinsic-source share
that is one to two orders of headroom.

The important negative result: the deviation is **flat** in
$h/\mathrm{ulp}(E_\mathrm{max})$ from 8 to 3000 across the 5 and 10 keV binades.
It does not scale as $\mathrm{ulp}/h$, so a spacing floor proportional to
$\mathrm{ulp}/\mathrm{rtol}$ would buy nothing, and the 2026-09-14 decision keeps
only the 8-ulp collapse floor. The residual appears to track
$\mathrm{ulp}/(\text{feature width})$ — resonance-energy rounding — which is
inferred from the binade dependence, not proven.

**The 20 keV binade is now measured, and dominant-line FWHM breaches its
share there.** `line-grid-sinc-convergence` left the $[16384, 32768)$ eV binade
unmeasured because no hopg or wse2 line reaches it. Thirty-three of the
forty-nine catalog materials do clear it kinematically at 300 keV, and diamond
is the first with real line yield inside it: its automatic bandwidth runs to
18 900 eV and `precision_ladder` locates a window on a line at 17 084 eV
(peak yield $5.2\times10^{-9}$, $\mathrm{ulp} = 1.95\times10^{-3}$ eV). On
identical trajectories (diamond, 300 keV, tilt 5, $N_e = 60$, 48 061 segments,
both precisions on the same CUDA device):

```{list-table} Float32 deviation from FP64 in the two upper binades of one diamond case, over spacing/ulp. Window 1 is the 20 keV binade.
:name: tbl-line-budget-binade
:header-rows: 1

* - Window (top)
  - $h/\mathrm{ulp}$
  - yield
  - FWHM
  - centroid / ulp
  - pointwise
* - 11 802 eV
  - 8 … 3000
  - $5.3\times10^{-7}$ … $1.3\times10^{-5}$
  - $3.9\times10^{-6}$ … $1.2\times10^{-4}$
  - 0.12 … 0.46
  - $\approx 6.7\times10^{-4}$
* - 17 094 eV
  - 8 … 3000
  - $1.7\times10^{-6}$ … $8.9\times10^{-5}$
  - $1.2\times10^{-5}$ … $1.8\times10^{-3}$
  - 0.21 … 1.24
  - $\approx 1.8\times10^{-3}$
```

Three readings. **The flatness holds**: deviation does not scale with
$h/\mathrm{ulp}$ in the 20 keV binade either, so the #109 conclusion — an
$\mathrm{ulp}/\mathrm{rtol}$ spacing floor protects nothing — extends to it, and
no interval collapsed at any rung. **Yield and centroid stay inside their
shares**: worst yield $8.9\times10^{-5}$ against $10^{-4}$, and the worst
centroid shift of 1.24 ulp is $2.4\times10^{-3}$ eV on a 17 keV line, a relative
$1.4\times10^{-7}$. **FWHM does not**: at $1.0$–$1.8\times10^{-3}$ it sits at one
to two times the $10^{-3}$ shape backend-precision share, against
$2.3\times10^{-5}$ measured in the 5 and 10 keV binades. Lineshape distortion
arrives well before node collapse, exactly as the #97 handoff predicted, and the
8-ulp floor does not see it.

So a window reaching the $\ge 20$ keV binade carries yield and centroid inside
this budget in float32, but **dominant-line FWHM from such a window needs
`PYRITE_FP64=1`** to stay inside its share. Caveats: one case, $N_e = 60$, one
window per binade, and the binade window's line is weak
(integral $5.2\times10^{-9}$ against $2.6\times10^{-8}$ one binade down), so a
shape statistic on it is the noisiest thing in the table. Whether the automatic
policy should warn when a plan's top node lands in that binade is a policy
question this measurement does not settle.

### Transport — $\varepsilon_\mathrm{trans}$

Not grid error, and not bounded by any tolerance in
{numref}`tbl-line-budget-allocation`. The spectrum is built from segments whose
emission coefficients are frozen at each flight's start energy, so the term is
the discretization error of the flight sequence itself.

What is measured ([`energy-step-convergence`](energy-step-convergence.md)):
replacing the frozen rule with the midpoint rule shifts no exit fraction,
retained energy, or transit clock by more than 1.6 combined Monte Carlo standard
errors at 4000 electrons across 14 cases. The one resolved difference is the
mean path length of the 5 keV carbon stopping case, where the frozen rule
overstates the CSDA range by 1.2% — below the CXR regime of this budget. For
coherent CXR the binding constraint is the absolute emission phase, met by the
propagation rule rather than by the substep count.

What controls it per run ([`radiation-error-estimators`](radiation-error-estimators.md)):
the CXR endpoint resonance-drift estimator reports each flight's resonance sweep
in units of its own sinc linewidth and warns above
`DEFAULT_RESONANCE_DRIFT_WARN` at p99. A drift of one linewidth means the frozen
line is displaced by its own width over a single flight — that is the regime
where a window seeded at the start-energy resonance no longer covers the feature
it was seeded for, and both the seeding and this budget fail together. The
estimator measures the sweep, not the resulting spectral error; the conversion
to a share of {eq}`eq-line-budget-decomposition` is **not established**.

### Monte Carlo statistics — $\varepsilon_\mathrm{stat}$

The only random term, scaling as $N_e^{-1/2}$, and at production electron counts
usually the largest in {eq}`eq-line-budget-decomposition`. It has no allocated
share because it is not a property of the grid: the run's own standard errors
report it, and it is driven down by electrons rather than by points.

Its role here is a methodological constraint, and it is absolute:

> **Never compare grid error across separate Monte Carlo runs.**

At any practical $N_e$, $\varepsilon_\mathrm{stat}$ dwarfs the entire
$10^{-3}$ grid budget, so a rung-to-rung difference taken across two runs
measures statistics and not grid. The harness isolates the grid share by
evaluating only the spectrum phase on one cached `simulate_trajectories` result,
or on a fixed seed with `n_segments` asserted identical, and proves it with
`convergence.segment_fingerprint` before any rung is compared. That the grid does
not enter the RNG stream is empirical — the 30 keV ladder reproduces to
$10^{-7}$ — and the seeding code is the place that has to keep it true.

## Where each gate cites its share

Each tolerance that gates a line observable names its share of
{numref}`tbl-line-budget-allocation`:

```{list-table}
:name: tbl-line-budget-gate-citations
:header-rows: 1

* - Gate
  - Constant
  - Share cited
* - automatic policy, intrinsic source
  - `_line_grid_policy.DEFAULT_RTOL["intrinsic_source"]`
  - the intrinsic-source column total, $10^{-3}$
* - automatic policy, detected counts
  - `_line_grid_policy.DEFAULT_RTOL["detected_counts"]`
  - the detected-counts column total, $10^{-2}$
* - ladder, yield and centroid
  - `convergence.HARNESS_RTOL["intrinsic_source"]`
  - the same $10^{-3}$, unchanged from the policy
* - ladder, detected counts
  - `convergence.HARNESS_RTOL["detected_counts"]`
  - the same $10^{-2}$, unchanged from the policy
* - ladder, FWHM and line/background
  - `convergence.SHAPE_RTOL`
  - the shape column total, $10^{-2}$; the ungated
    `SHAPE_DIAGNOSTIC_RTOL` reports the distance to $10^{-3}$
* - window plan, node collapse
  - `_line_grid_policy.DEFAULT_BACKEND_SAFETY_ULPS`
  - the backend-precision row, $10^{-4}$ intrinsic source
* - ladder, resampling noise
  - `convergence.NOISE_FRACTION`
  - 10% of whichever class tolerance applies
```

A tolerance raised above its share, per call or through
`PYRITE_ENERGY_GRID_RTOL[_<OBSERVABLE>]`, leaves this budget. The policy reports
the source of every tolerance it resolved, so a provenance record states which
budget a result was produced under.

## Open

- The window-quadrature share is measured on twelve hopg/wse2 cases and met in
  every one, but the hopg 300 keV ladder is truncated at 8 samples per feature
  by the point budget, so its acceptance rests on a single Richardson triple.
- The cost argument for windows is energy dependent and erodes above ~100 keV:
  broadening of the resonance population merges windows until the plan is a fine
  uniform grid. Narrower seeding — per-reflection rather than per-case windows —
  is the obvious lever and is not attempted here.
- wse2 100 keV tilt 85 accepts no spacing because its line/background ratio never
  stabilises rung to rung, though it matches the dense reference to
  $7.1\times10^{-4}$. Whether that is a real shape instability or an artifact of
  the ratio's background window is unresolved.
- Coherent-route (`coherent=True`) aliasing is not analyzed; the band-limit
  argument does not transfer to a sum of complex amplitudes squared after the
  fact, and the in-window fringe period is then set by phase differences across
  segments rather than by any single $\pi/a_w$. Owned by #117, which gates
  coherent acceptance of windows.
- The width-tail measurement behind {numref}`tbl-line-budget-width-tail` covers
  four cases at $N_e \le 240$; the campaign's own $N_e = 2000$ minima are not
  measured, only the quantile ratio that the argument rests on.
- The $\ge 20$ keV float32 binade FWHM breach rests on one diamond case at
  $N_e = 60$ with a weak in-window line; it is not reproduced on a second
  material, and no runtime guard warns when a plan reaches that binade.
- The interpolation share rests on a point-wise $\mu$ bound; the spectrum-level
  exact-versus-tabulated A/B of `line-absorption-tabulation` is still open.
- The resonance-drift estimator's conversion from linewidths of sweep to a share
  of the spectral error is not established.

## Reproduce

```bash
# window-refinement ladder against a dense uniform reference (remote; heavy)
uv run python -m pyrite.energy_grid.convergence_job start-windows \
    --json-out line_window_convergence.json
uv run python -m pyrite.energy_grid.convergence_job status
uv run python -m pyrite.energy_grid.convergence_job pull \
    --json-out line_window_convergence.json

# uniform resolution ladder and the float32/FP64 precision table (#109)
uv run python -m pyrite.energy_grid.convergence_job start
uv run python -m pyrite.energy_grid.convergence_job start-precision
```
