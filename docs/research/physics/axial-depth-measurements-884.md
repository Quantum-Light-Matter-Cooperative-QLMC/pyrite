---
orphan: true
---

# Axial photon-source depth measurements — #884 (research producer)

**Status:** numerical contract only, ledgered as `cr-axial-depth-budget`; that claim
covers arithmetic, not calibration. This module does not establish a physical radiation source,
matching-surface convergence or a calibrated operating depth. It cannot turn
existing electron population / return-deficit records into photon-source
bounds without an independently validated radiation response (#992).

`axial_depth_measurements.compile_axial_photon_depth_evidence` supplies the
source-normalized interval contract in `axial_depth_budget.py` from paired
photon-yield comparisons. The output is accepted by the existing research
`require_axial_photon_depth_budget` preflight and
`sample_budgeted_physical_handoff` adapter; it grants **no** native run
admission (#994).

## Required evidence for each interval

Each `AxialPhotonSourceIntervalMeasurement` carries a *single checkpoint*
source reference, separate one-sided statistical and between-depth downward
allowances for that reference, and five comparisons to the **same** reference:
return flux, unresolved outgoing, matching surface, grid/supercell, and source
model/partition. Every comparison supplies candidate means, a simultaneous
statistical bound on the **paired difference** and an additional upper bound
for the difference's variation over the **whole** stated depth interval.
This is an assertion of measured/derived bounds, not something inferred by
sampling the interval endpoints. Missing continuum bounds must not be set to
zero merely because no difference was observed at a checkpoint.

All vectors are nonnegative, finite *integrated* photon yields per incident
in exactly matched spectral/angle/polarization bins (not density at bin
centres). For bins $j$, reference means $r_j$, reference allowances
$u_j,v_j$, comparison means $q_{c,j}$ and their paired-error and drift
allowances $s_{c,j},t_{c,j}$, the producer computes

$$
L = \sum_j r_j - \sum_j(u_j+v_j) > 0, \qquad
B_c = \sum_j(|q_{c,j}-r_j|+s_{c,j}+t_{c,j}), \qquad
\epsilon = \frac{\sum_c B_c}{L}.
$$

Absolute differences add *before* the spectral sum so opposite-sign changes
never cancel. The guard controls the **aggregate observable**; to protect an
individual narrow photon bin, give it a distinct `observable_key` and a
one-bin evidence record. For compatibility with the #884 interval contract,
matching-surface and grid/supercell bounds are combined into
`grid_supercell_upper`. The original six provenance IDs remain attached.

The reported confidence for several intervals is **at most**
$1-\sum_i(1-c_i)$ (union bound), not the minimum $c_i$. Each $c_i$ must itself
cover all channels, bins and depths **simultaneously**. Dependence between
intervals is allowed by the union bound. A bare SEM, confidence interval at
one checkpoint or a zero drift allowance without a derivation is insufficient.

## Physical and numerical limits

The output is a declared research envelope, **not independently verified
measurement evidence**. The supplier must retain raw paired histories,
uncertainty construction, source-partition proof and the physical return/tail
upper envelope. A positive reference lower yield on `(0,z]` is generally
impossible for a cumulative photon yield that vanishes continuously at the
entrance: use a strictly positive first interval start (and refuse shallower
depths), or independently justify a nonzero surface contribution. Never
extrapolate from a measured endpoint into this region.

A finite error envelope on several intervals does not imply all intermediate
or later depths are admissible. The underlying guard refuses a requested depth inside a gap, zero-depth
relative error, inadequate joint confidence and any interval exceeding the
caller-selected source tolerance. Standard axial runs remain refused until #992, #993 and #994 are complete.
