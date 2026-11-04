---
orphan: true
---

# Axial entrance-zone photon-source depth budget — #884 (research contract)

**Status:** numerical contract and preflight only; **not calibrated**. No physical
axial depth is admitted by this work. Standard Scene/Sweep/CLI axial runs remain
refused under #624/#886. The outgoing instrument in #883 is anchored for its
own scope, but that status does not establish source accuracy or a depth budget.

## Observable and error definition

A budget is resolved by three exact strings: `case_key` (material, axis,
incident energy/phase space, collision physics, matching rule, domain, grids,
solver versions), `observable_key` (photon energy/angle/polarization bin and
radiation ownership), and `model_id` (source implementation). Reusing bounds
from another source model or photon bin is prohibited.

For a given observable, let $Y_\mathrm{ref}(z)$ be the **incident-normalized**
expected photon yield [photons / incident electron]. For every requested
$z\in(a,b]$, the supplier must establish a **simultaneous** lower confidence
bound $Y_\mathrm{ref}(z)\geq L>0$ and conservative absolute upper bounds (same
units) on contributions to source error from omitted return flux $R$, unresolved
outgoing weight $U$, grid/supercell and Bloch discretization $G$, and
source-model closure/partition $S$. The implemented preflight enforces

$$
\sup_{z\in(a,b]}
\frac{|Y_\mathrm{model}(z)-Y_\mathrm{ref}(z)|}{Y_\mathrm{ref}(z)}
\leq \frac{R+U+G+S}{L} \leq \epsilon_\mathrm{max}.
$$

The first inequality requires independently justified **disjoint error
coverage** (or valid triangle-bound overlap treatment): the code verifies
arithmetic, not the premise. ``unresolved`` is not itself a photon yield;
it must first be mapped through a defensible photon-yield envelope. The old
`AxialDepthEvidence` sums three supplied dimensionless fractions; it is not
interchangeable with this photon-specific definition. In particular, a 1%
uncertainty in retained electrons need not imply 1% uncertainty in a weak
spectral line. Bounds may exceed one photon per incident in magnitude and are
never blindly clipped to probabilities.

The confidence level must cover **all error channels and all depth intervals
jointly**. Histories, paired grid comparisons, step refinements and the
source-model oracle must justify that confidence with retained independent
seeds, estimator uncertainty and systematic envelopes. An SEM or a pointwise
confidence interval is **not** an upper confidence bound uniform over an
interval. The existing Si [001] studies do not provide these bounds.

## API and refusal contract

`axial_depth_budget.AxialPhotonBoundInterval` records $(a,b]$, $L$, the four
absolute bounds and provenance IDs. `AxialPhotonDepthEvidence` ties disjoint,
ordered intervals to a case, observable, source model and simultaneous
confidence. `require_axial_photon_depth_budget` selects **only** an interval
covering the requested depth, refuses zero-depth relative normalization,
missing/mismatched data, inadequate joint confidence and a bound exceeding the
requested error tolerance. No interpolation or extrapolation is performed;
a depth gap is a refusal, even when its endpoints passed. Adjacent intervals
may have different bounds, and a failing later interval does not invalidate
an independently certified earlier one.

`sample_budgeted_physical_handoff` is an **opt-in research adapter**. It checks
the source-depth budget *before* importing/initializing the physical outgoing
sampler and before any sampling work. It does not persist results, change the
unbudgeted #883 instrument, or enable production axial admission. Creating a
Python evidence object does **not** constitute independent validation.

## Required next measurements (still open)

1. Establish a photon-source reference and a non-overlapping radiation
   ownership contract (#885). Without a finite lower reference yield and an
   upper bound on omitted source channels the source-relative budget is
   undefined; refusal is required.
2. Obtain depth-uniform return-flux bounds from the #883 outgoing instrument,
   including return from escaped trajectories, and source-weight the unresolved
   tail. Separately quantify grid/supercell/Bloch and matching-cut effects.
3. Retain raw independent histories, exact resolved case and observable keys,
   per-component uncertainty attribution, versioned model identity and joint
   coverage evidence. Register and independently rederive
   `cr-axial-depth-budget` before marking any calibrated envelope anchored.
4. Only then integrate a trusted calibration lookup and early refusal across
   the production API, CLI, remote and persistence paths (#886). A caller-made
   research object must not be accepted as a production eligibility token.
