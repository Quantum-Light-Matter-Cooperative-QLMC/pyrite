# `finite-footprint-longitudinal-decoherence`

## Claim

For one fixed sampled transverse/transport realization, let $S_e(E)$ be the coherent segment field from electron $e$, including its actual transverse launch phase, finite-crystal hit/miss history, escape distance, attenuation, and transport randomness. Let $t_e$ be an independent Gaussian longitudinal arrival offset with RMS spatial duration $\sigma_z=c\sigma_t$. The arrival-time ensemble average is

$$
\left\langle\left|\sum_e S_e e^{i\omega t_e}\right|^2\right\rangle_{t\mid S}
= (1-F_z)\sum_e|S_e|^2
+ F_z\left|\sum_e S_e\right|^2,
\qquad
F_z=\exp[-(\omega\sigma_z)^2].
$$

The intended quantity is conditional on the sampled transverse/transport realization. It is not the additional transverse ensemble average.

## Implementation-context derivation

Expanding before averaging gives

$$
|A|^2=\sum_e|S_e|^2
+\sum_{e\ne e'}S_eS_{e'}^*e^{i\omega(t_e-t_{e'})}.
$$

For independent zero-mean Gaussian offsets, $\langle e^{i\omega t}\rangle=\exp[-(\omega\sigma_z)^2/2]$; independence gives $\langle e^{i\omega(t_e-t_{e'})}\rangle=F_z$ for every $e\ne e'$. Substituting and using $\sum_{e\ne e'}S_eS_{e'}^*=|\sum_eS_e|^2-\sum_e|S_e|^2$ yields the claim.

This derivation was written in implementation context. It establishes cheap filters only; it is **not** independent fresh-context verification.

## Filters

- Units: $\omega$ is in $\mathrm{\mathring A}^{-1}$ and $\sigma_z=c\sigma_t$ is in $\mathrm{\mathring A}$, so the exponent is dimensionless.
- Short-bunch limit: $\sigma_t\to0$ gives $F_z\to1$ and the full sampled finite-crystal field $|\sum_eS_e|^2$.
- Long-bunch limit: $\omega\sigma_z\to\infty$ gives $F_z\to0$ and the per-electron coherent floor $\sum_e|S_e|^2$.
- Single electron: both terms reduce to $|S_1|^2$ for every $F_z$.
- Assumptions: longitudinal arrival offsets are independent of transverse launch and transport state, independent across electrons, and Gaussian with the supplied RMS duration.

## Source-to-code boundary

`pyrite.montecarlo.spectrum.lines.mc_spectrum` retains sampled `seg_r` and its finite-prism escape attenuation in both reductions, removes only `t0_ang`, and blends the grouped and flat row intensities with the analytic $F_z$. The blend is applied before the incoherent sum over reflection/mosaic rows.

Regression anchors:

- `tests/montecarlo/test_coherent_emission.py::test_finite_footprint_partially_coherent_longitudinal_blend`
- `tests/montecarlo/test_xray_dispersion_cuda.py::test_coherent_decoherence_gaussian_bunch_finite_footprint_uses_jit_blend`

## Status

`filtered`. Units, limits, and source-to-code structure match. Independent fresh-context verification remains required before promotion to `rederived`; human sign-off remains pending.
