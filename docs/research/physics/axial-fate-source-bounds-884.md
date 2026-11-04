---
orphan: true
---

# Physical outgoing fates → photon-source error — #884 (research only)

**Status:** numerical envelope composition only, **not calibrated**;
`cr-axial-depth-budget` is **unverified**. This does not admit native axial
runs, identify a radiation partition, or certify a return calculation.

## What is measured, and what must be supplied

The #883 `sample_physical_handoff` instrument gives exactly one terminal
`AxialHandoffRecord` per sampled incident: `physical_outgoing`, `unresolved`,
or `retained`. For independently sampled, *unit-weight* incident histories,
`axial_fate_source_bounds.physical_fate_source_uppers` counts terminal events
by interval stop. Let $\hat p_{o,b}$ and $\hat p_{u,b}$ denote outgoing and
unresolved fractions at stop depth $b$ for $N$ incident histories.

With $m$ depth intervals and total sampling failure probability $\delta_s$,
a one-sided Hoeffding/union-bound allowance is

$$
 h = \sqrt{\frac{\log(2m/\delta_s)}{2N}},\qquad
 p_{c}(z)\leq \min(1,\hat p_{c,b}+h),\quad z\in(a,b],\ c\in\{o,u\}.
$$

The uniformity over `(a,b]` follows from monotonicity of a terminal-event
CDF, **not** from smoothness or endpoint interpolation. It is valid only
for independently drawn incident fates under the same resolved physical
case and sampling law. Reusing one finite trajectory set across depths is
allowed; the union bound does not require independence between depth checks.
Importance weights, correlated incident samples, changed beam distributions,
and runs sampled to a shorter maximum depth are unsupported/refused.

**A fraction of electrons cannot be substituted for a fraction of photons.**
The caller must separately supply a *uniform worst-case upper bound* on the
absolute photon-source error caused by **each** outgoing or unresolved fate,
in every integrated photon observable bin. These are the
`AxialFatePhotonResponseCaps` arrays $C_{o,j}$ and $C_{u,j}$ (photons per
outgoing/unresolved electron). The response caps must include later return
and omitted radiation into the entrance zone; no return probability smaller
than one is assumed. They may exceed one photon/electron, and must not be
inferred from #883 electron balance or a finite subset of trajectories.
The caps have an external joint failure-probability allowance $\delta_c$,
with zero reserved for **rigorous deterministic** external bounds.

The source-error upper terms, in photons per *incident* electron, are

$$
 R_{(a,b]}=\min(1,\hat p_{o,b}+h)\sum_j C_{o,j},\qquad
 U_{(a,b]}=\min(1,\hat p_{u,b}+h)\sum_j C_{u,j}.
$$

Bin-wise absolute caps add, with **no** cancellation between disjoint photon
bins or fate classes. Even when no outgoing or unresolved events are seen,
the finite-$N$ bound is nonzero unless the cap is rigorously zero.

## Composition and limits

`add_physical_fate_source_bounds` adds these two estimates to the *existing*
source-depth interval `return_upper` and `unresolved_upper` values (it never
silently replaces paired-comparison errors). With original joint source
confidence $c_s$, the result carries the conservative lower confidence

$$ c_{\mathrm{combined}} = c_s - \delta_s - \delta_c. $$

Every interval must lie within the sampled maximum depth and have the **same**
case, observable bins, source model, and disjoint evidence provenance.
The result can be supplied to the existing fail-closed
`require_axial_photon_depth_budget` and research-only
`sample_budgeted_physical_handoff` preflight. No production entry point has
been modified.

This is **not** an automatic physical certificate: #885 must establish the
photon response caps including the exact native/axial radiation partition,
source normalization and omitted emission; #884 must independently validate
return response and interval-uniform applicability. Naive use of identical
uncertainty sources in both paired source comparisons and response caps will
be *conservative but double counted*. A tighter decomposition requires proof
of independent error coverage, rather than subtracting overlapping terms.
Independent seed/history provenance, solver and cap versions, raw records,
and fresh-context `rederived` review must precede any calibration claim.
