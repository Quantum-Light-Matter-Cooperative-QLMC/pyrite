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
