# Joint history roll and mosaic tilt (#1527, Phase IId)

This is an opt-in fixed-grid/fixed-count campaign estimator, not a new texture
or transport model. Select `FiberAverage(..., roll_sampling="history")` and
`CxrReflections(..., mosaic_route="mc-sampled", mosaic_nodes=1)` on a mosaic
HOPG slab. `mc` retains deterministic Gauss-Hermite quadrature; `auto` and all
legacy defaults are unchanged. A deterministic fiber roll still refuses
sampled mosaic, and the direct prepared-domain scorer stays quadrature-only.

For primary `i` the existing fiber counter stream generates one sector roll
`phi_i`; the existing mosaic stream generates `K` Gaussian rotation-vector
tilts `t_ik`, using its shifted rank-1 lattice. Each marginal draw is uniform
in roll and Gaussian in tilt. The vector is `R_m(t_ik) R_z(phi_i) g_0`;
basal rows use the same tilt stream and do not need the roll. A primary keeps
these draws across every physical flight, substep, and reflection, including
replayed segments. The contribution of each draw is weighted `1/K` and all
histories are normalized by the original incident count. K > 1 reduces tilt
noise but does not supply K independent rolls or K independent histories.

The already persisted mosaic route/count, fiber RNG version, and case seed
fully name the estimator; no artifact schema change. Sparse mosaic shifts
match the old counter construction exactly, without allocating across gaps
in global IDs. The native route evaluates row rotations on device and reuses
its bounded orientation-independent coupling cache. The isotropic continuum
subtraction and thermal tensor remain unchanged.

Qualification: compare the full native spectrum and basal/nonbasal components
against explicit per-primary scalar rotations for K=1 and K>1. Compare
independent orientation seeds at increasing populations and against increasing
GH orders on identical frozen trajectories, with no artificial smoothing.
A finite GH reference is not assumed converged. Verify live and HDF5 replay
with joint mosaic and unequal line/continuum counts. Automatic grid audits,
adaptive precision and persisted checkpoint/restart remain outside this slice.
