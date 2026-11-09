# `cr-axial-depth-calibration`: measured depth budgets from paired batches

**Status:** see the [ledger row](../ledger-channeling.md#cr-axial-depth-calibration). Task-owner derivation (2026-11-08, #993).

This claim covers how `axial_depth_calibration.py` turns chunk-paired
fibre-source arms and unit-weight fate samples into the inputs of the
[`cr-axial-depth-budget`](cr-axial-depth-budget.md) contract. The contract's
arithmetic is that claim's. Here the claim is that the supplied bounds hold
at the stated joint confidence under the stated assumptions. The measured
values for each case are in the
[research note](../../research/physics/axial-depth-calibration-993.md).

## Observable and intervals

The observable is the on-axis photon yield per incident electron per sr,
integrated over one photon-energy band, for **yield at crystal thickness**.
For each checkpoint thickness $c$ the interval is $(c-s,c]$ with span $s$
(15 Å). $Y_a(z)$ is the yield of arm $a$ for a slab of thickness $z$.

All arms share seed and incident chunks. Each arm scores every recorded
thickness from the same primary histories
(`fiber_thickness_spectra`): restricting a history's two-time estimate to
depths $z_0..z_k$ is the estimator for a slab of thickness $z_k$ on that grid,
because records up to $z_k$ never depend on later depths and the trapezoid
weights and exit path $z_k-z$ are those of the truncated grid. The anchor test
checks that a prefix equals an independent run on the truncated grid, events
included.

## Batch bounds

Let $x_b$ be the mean of a paired linear combination of arm yields over chunk
$b$ ($n_b$ primaries, $B$ chunks, $N=\sum n_b$). Chunks hold disjoint
incidents, so the $x_b$ are independent. With
$\bar x=\sum n_bx_b/N$ and
$\hat\sigma^2=\sum n_b(x_b-\bar x)^2/((B-1)N)$, the one-sided bound is
$\bar x\pm t_{B-1}(1-\alpha')\hat\sigma$. For equal $n_b$ this is the
Student-$t$ bound on the mean of $B$ normal batch means. **Assumption:** the
batch means are close to normal. The record keeps the ratio of the standard
error from pairs of adjacent chunks to that from single chunks, which is
near one when this holds.

Each interval has error rate $\alpha$, split evenly over all its one-sided
bounds (Boole's inequality), with an absolute value counting two. The
intervals of one record split the source risk evenly, so the #884 union
bound gives joint source confidence $1-\sum\alpha_i$.

## Interval inputs

With $D_a=Y_a-Y_\mathrm{ctl}$ and $\mathcal Z$ the recorded thicknesses in
$[c-s,c)$ shared by all arms:

- Reference: $r=\bar Y_\mathrm{ctl}(c)$, statistical lower allowance
  $t\hat\sigma$, and drift
  $\kappa\max_{z\in\mathcal Z}[\overline{Y(c)-Y(z)}+t\hat\sigma]^+$.
- Probe $a$ (return flux, unresolved, matching surface, grid, source model):
  candidate $r+\bar D_a(c)$, paired statistical bound $t\hat\sigma[D_a(c)]$
  and drift $\kappa\max_{z\in\mathcal Z}(|\overline{D_a(z)-D_a(c)}|+t\hat\sigma)$.
- Interaction:
  $I=D_\mathrm{all}-\sum_aD_a$, bounded by $|\bar I|+t\hat\sigma$ plus its
  drift. The bound is added to the source-model probe's statistical term.
  Because $D_\mathrm{all}=\sum_aD_a+I$ exactly, the probes plus this allowance
  cover the all-on change, which is the #884 requirement on interaction.

**Assumption (drift):** between recorded thicknesses the yield varies by no more
than $\kappa$ (default 2) times its largest change over the recorded points
of the interval. The grid step (≤ 15 Å) is below a quarter of the main-line
formation length $2\gamma^2\hbar c/E$ (≈ 70 Å at 3.5 MeV, 3.4 keV).

## Fate response caps

These are mean-response caps, as decided for #993, not the per-electron
worst case the #884 note describes.

- Outgoing. In the all-on arm the window and cut are wide enough that a
  fraction of control-outgoing electrons is retained, so its emission includes
  their return. With $F_o$ the outgoing CDF at $c$,
  $C_o=\max_c\,\mathrm{UCB}|D_\mathrm{all}(c)|/\mathrm{LCB}[F_o^\mathrm{ctl}(c)-F_o^\mathrm{all}(c)]$.
  **Assumption:** electrons still outgoing in the all-on arm have mean return
  response no larger than the measured one.
- Unresolved.
  $C_u=\max_c\,\mathrm{UCB}\,Y_\mathrm{ctl}(c)/\mathrm{LCB}\,F_r^\mathrm{ctl}(c)$,
  with $F_r$ the retained fraction. **Assumptions:** an unresolved electron,
  had it stayed retained, would emit no more than the mean electron retained
  to $c$, and group-conditional mean emission is nonnegative, so
  $F_r\cdot E_r\le Y$.

Fate fractions use one-sided Hoeffding bounds
$\sqrt{\log(1/\alpha')/(2N)}$; yields use the batch rule. The cap
failure probability is the Boole total of these bounds. Control fates come
from the #883 instrument (`sample_physical_handoff`). All-on fates come from
the all-on arm's own primary events, whose entry weight is exactly one under
$P=R$; the instrument needs a finite matching cut and cannot run that arm.

The joint confidence of a record is
$1-\delta_\mathrm{source}-\delta_\mathrm{sampling}-\delta_\mathrm{cap}$
(defaults 0.03, 0.01, 0.01).

## Limiting cases

- Identical arms give zero probe differences and bounds that shrink as
  $1/\sqrt N$.
- An exactly additive all-on arm gives an interaction bound of
  $t\hat\sigma$ only.
- A nonpositive lower bound on the returned or retained fraction refuses:
  there is no cap, hence no budget.

## Not claimed

- Physical accuracy of the fibre model against measurement, which belongs to
  [`cr-axial-persisted-channel`](../ledger-channeling.md#cr-axial-persisted-channel)
  and #994.
- Free→free ownership with native CBS.
- Depth-resolved use.
- Any admission.

## Independent verification

Fresh-context verifier, 2026-11-09. The rederivation below was written from
the ledger row and the method decisions fixed for #993 before the
implementation body was read. The upstream contract
[`cr-axial-depth-budget`](cr-axial-depth-budget.md) is taken as given.

### Rederivation

**Batch bound.** The chunks hold disjoint incidents, so chunk means $x_b$ are
independent. For equal counts $n_b=n$,
$\sum_b n(x_b-\bar x)^2/((B-1)Bn)=s_x^2/B$, the squared standard error of the
mean of $B$ batch means. A one-sided bound at level $\alpha'$ is therefore
$\bar x\pm t_{B-1}(1-\alpha')\hat\sigma$. A two-sided bound on $\lvert x\rvert$,
$\lvert x\rvert\le\lvert\bar x\rvert+t\hat\sigma$, uses two one-sided events.

**What the contract needs.** The contract forms
$L=\sum_j[r_j-\sigma^{\rm ref}_j-\Delta^{\rm ref}_j]$ and, per probe,
$\sum_j[\lvert\tilde Y_{a,j}-r_j\rvert+\sigma_{a,j}+\Delta_{a,j}]$, then sums the
probes linearly. Over $z\in(c-s,c]$ this needs:

1. $Y_\mathrm{ctl}(z)\ge Y_\mathrm{ctl}(c)-[Y_\mathrm{ctl}(c)-Y_\mathrm{ctl}(z)]$.
   This takes one lower bound on $Y_\mathrm{ctl}(c)$ and, for the drift, one
   *upper* bound on $Y(c)-Y(z')$ at each recorded $z'\in\mathcal Z$. The
   drift is clipped at zero and multiplied by $\kappa$. Both bounds are one-sided.
2. $\lvert D_a(z)\rvert\le\lvert D_a(c)\rvert+\lvert D_a(z)-D_a(c)\rvert$.
   This takes two one-sided bounds at $c$ and two at each $z'$, with the
   second term multiplied by $\kappa$.
3. $D_\mathrm{all}=\sum_aD_a+I$ exactly, so $\lvert D_\mathrm{all}\rvert\le\sum_a\lvert D_a\rvert+\lvert I\rvert$.
   Bounding $\lvert I\rvert$ like a probe and adding it to any one probe makes the
   linear sum cover the all-on change.

The bound count per band and interval is therefore

$$
K=\underbrace{1+\lvert\mathcal Z\rvert}_{\rm reference}
+\underbrace{2\cdot 6\,(1+\lvert\mathcal Z\rvert)}_{5\ {\rm probes}+I}
=13\,(1+\lvert\mathcal Z\rvert),
$$

with $\alpha'=\alpha_i/K$ and $\alpha_i=\delta_\mathrm{source}/m$ over $m$ intervals.

**Caps.** With $E_r$ the mean emission of an electron retained to $c$ and
nonnegative group means, $F_rE_r\le Y_\mathrm{ctl}$, so
$E_r\le\mathrm{UCB}\,Y_\mathrm{ctl}/\mathrm{LCB}\,F_r$ when the denominator is
positive. The returned fraction
$F_o^\mathrm{ctl}-F_o^\mathrm{all}$ comes from two independent samples. Its
lower bound is $(\hat F_o^\mathrm{ctl}-h_\mathrm{ctl})-(\hat F_o^\mathrm{all}+h_\mathrm{all})$,
with one-sided Hoeffding allowance $h=\sqrt{\ln(1/\alpha'')/(2N)}$ from
$P(\hat p-p\ge h)\le e^{-2Nh^2}$. A cap must hold at every $c$ at once, so
it takes the maximum over $c$ and Bonferroni over all $c$. Per checkpoint the
count is $2$ (for $\lvert D_\mathrm{all}\rvert$) plus $1$ (for $Y_\mathrm{ctl}$)
per band, plus $3$ Hoeffding bounds ($F_o^\mathrm{ctl}$, $F_o^\mathrm{all}$,
$F_r$), giving $K_\mathrm{cap}=m(3\,n_\mathrm{bands}+3)$.

**Joint confidence.** Boole's inequality over the three families gives
$1-\sum_i\alpha_i-\delta_\mathrm{sampling}-\delta_\mathrm{cap}=1-0.03-0.01-0.01=0.95$.
The upstream sampling family is its $2m$ Hoeffding CDF bounds at total
$\delta_\mathrm{sampling}$. Reusing the control fate sample in both the cap
and sampling families needs no independence under the union bound.

**Units.** The spectra are in photons/(eV sr incident electron). Integrating
over a band in eV gives photons/sr per incident, which is the unit of $Y$, $D$,
$\sigma$ and $\Delta$. Caps are photons/sr per electron of the fate class.
Multiplying a cap by a fate fraction gives photons/sr per incident again.
$\varepsilon$ is dimensionless.

### Comparison with the implementation

| Item | Rederived | `axial_depth_calibration.py` | Result |
|---|---|---|---|
| Batch SE | $\sqrt{\sum n_b(x_b-\bar x)^2/((B-1)N)}$ | `batch_mean_se` L98–105 | match |
| Interval count $K$ | $13(1+\lvert\mathcal Z\rvert)$ per band | L164, `bands*(1+len(shifts)+12*(1+len(shifts)))` | match |
| Reference allowances | lower $t\hat\sigma$; one-sided drift, clipped, $\times\kappa$ | L172–177, L209 | match |
| Probe stat/drift | $\lvert\bar D\rvert+t\hat\sigma$ via contract; drift $\kappa\max(\lvert\bar\cdot\rvert+t\hat\sigma)$ | L185–202 | match |
| Interaction | $I=D_\mathrm{all}-\sum_{5}D_a$, $\lvert\bar I\rvert+t\hat\sigma$ plus drift, on `source_model` | L179–196 | match |
| Shared thicknesses | recorded $z\in[c-s,c)$ for all arms | `_common_shift` L116–126 | match (the left endpoint $c-s$ is included, which is conservative) |
| Caps | $C_o$, $C_u$ as above; $K_\mathrm{cap}=m(3n_\mathrm{bands}+3)$ | `response_caps` L257–274 | match |
| Hoeffding | $\sqrt{\ln(1/\alpha'')/(2N)}$, signs as above | L112–113, L267–268 | match |
| Fate fractions | CDF at $c$ (`depth <= c`); retained $=1-$ outgoing $-$ unresolved | `_fractions` L223–231 | match (fate vocabulary is exactly retained/physical_outgoing/unresolved) |
| Source risk split | $\alpha_i=\delta_\mathrm{source}/m$ | L311; contract sums $1-\mathrm{level}$ | match |
| Joint confidence | $0.97-0.01-0.01$ | `add_physical_fate_source_bounds` | match |
| Units | photons/sr per incident | `checks/axial_depth_calibration.py::arm_batches` (trapezoid over eV, divided by count) | match |

The fate multiplier is applied only in the upstream
`add_physical_fate_source_bounds`. This module and its driver compile only
yield-at-thickness records (`sampled_max_depth_ang` is the slab thickness).
Depth-resolved use stays unclaimed.

### Limiting cases

- **Identical arms.** With shared histories, $D_a\equiv0$ in every chunk.
  The probe and interaction terms are then zero, and the reference lower
  bound loses only $t\hat\sigma$ plus drift. With independent noise, the
  terms scale as $\hat\sigma\propto1/\sqrt N$. This is the stated limit.
- **Zero fates.** The fate terms do **not** vanish. If the control
  outgoing fraction is zero, the returned-fraction LCB is $-h_\mathrm{ctl}-h_\mathrm{all}<0$
  and calibration refuses (L269–270), as the ledger limit states. If the caps
  exist but no fates fall in a range, the upstream term is
  $\min(1,h)\cdot C$, a finite-$N$ allowance, which matches the contract's
  stated limit.

### Spot recompute

Case A, band `whole`, interval $(2985,3000]$ Å, recomputed from the stored
record with `scipy.stats.t` only:

| Quantity | Recomputed | Stored |
|---|---|---|
| $K$ ($\lvert\mathcal Z\rvert=1$) | 26 | 26 |
| $t_{199}(1-0.01/26)$ | 3.416407 | 3.416407 |
| $t_{199}(1-0.01/18)$ (caps) | 3.309060 | 3.309060 |
| $N$ from Hoeffding, control / all-on | 50000 / 20000 | — |
| $C_u=\max_c(\bar Y+t\hat\sigma)/(F_r-h)$ | $6.61883\times10^{-4}$ | $6.61883\times10^{-4}$ |
| fate return term $(0.17324+0.007998)\,C_o$ | $2.7185\times10^{-6}$ | — |
| fate unresolved term $(0.00866+0.007998)\,C_u$ | $1.10257\times10^{-5}$ | — |
| $\varepsilon$ from stored components | 0.0397207 | 0.0397207 |
| joint confidence | 0.95 | 0.95 |

The record does not store the drift terms, the reference drift or the
$D_\mathrm{all}$ numerator of $C_o$. Subtracting the recomputed stat and fate
terms from the stored channel totals leaves implied drifts. These are all
nonnegative and small: $\Delta^{\rm ref}=4.13\times10^{-6}$ against
$r=5.32\times10^{-4}$, and channel drifts of
$4.9\times10^{-8}$ (return), $7.2\times10^{-9}$ (unresolved),
$5.5\times10^{-8}$ (matching plus grid) and $6.1\times10^{-9}$ (source model).
That is consistent with the formulas, but those terms cannot be checked from
the record alone. The same recompute on the last intervals of B and C
reproduces $K$, both quantiles, $C_u$, $\varepsilon$ and 0.95, again with
nonnegative implied drifts.

### Findings

- Comment at `axial_depth_calibration.py:162` says "2 one-sided reference
  bounds per shift-and-checkpoint". The code counts one per point, which is
  correct: the reference bounds are all one-sided. Only the comment is wrong.
- The check records omit the per-term drifts and the $C_o$ numerator, so a
  reviewer cannot fully recompute an interval from the record. This is an
  evidence gap, not an arithmetic defect.
- Focused tests: `tests/montecarlo/quantum/test_channeling_axial_depth_calibration.py`
  5 passed.

**Verdict:** `rederived`. The modelling assumptions in the ledger row
(normal batch means, drift factor $\kappa$, mean-response caps) are method
decisions and are not certified here.
