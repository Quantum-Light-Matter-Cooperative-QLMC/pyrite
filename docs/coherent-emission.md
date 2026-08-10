# Coherent-emission tracking

`pyrite run --coherent` enables an experimental phased sum across trajectory
segments and electrons. Default scans remain incoherent and retain their
historical checkpoint identity. Coherent scans use identity-qualified stems, so
the two modes cannot overwrite each other.

## Model

For one reciprocal vector and polarization, the opt-in path computes

```text
E_tot(E, n_hat) = sum_j A_j Q_j(E) exp[i E(t_abs,j - n_hat.r_j)/(hbar c)]
dN/(dE dOmega) proportional to |E_tot|^2
```

where `t_abs = t_ang + L_ang/(2 beta) + t0_ang`: transport's `t_ang` remains
segment-start age, the half-flight term pairs time with `r_mid`, and `t0_ang`
carries each electron's longitudinal bunch offset. `Q_j` is the unsquared
finite-segment sinc amplitude.
Polarizations, reciprocal vectors, and mosaic orientations still add
incoherently. `components=True` is rejected because coherent PXR/CBS cross terms
make a uniquely additive component split impossible.

Expected limits:

- `--incoherent` and the default preserve the previous path bit-for-bit.
- One segment gives the same self-term in both modes.
- One straight constant-velocity flight is unchanged when represented by two
  contiguous half-segments.
- Identical in-phase emitters give the `N^2` intensity limit before
  per-electron normalization.
- Randomized or sufficiently spread arrival phases should approach the
  incoherent sum.
- A Gaussian longitudinal distribution should give bunch form factor
  `exp[-(omega sigma_z)^2]`.

## Validation boundary

Status: **unverified**. The implementation and regression limits do not
independently establish the phase model. Before scientific use:

1. derive whether the spatial phase is `k.r` or `(k + g).r`, including sign and
   its relation to the open line-energy-dispersion discrepancy;
2. reproduce the Gaussian bunch form factor and decoherent limit independently;
3. anchor representative CPU/GPU cases and quantify complex-grid memory cost;
4. obtain human sign-off through the
   [physics validation workflow](validation/README.md).

Track status in the [`coherent-emission` ledger row](physics-validation-ledger.md).
Emission is a profile policy (`incoherent`/`coherent`/`both`) rather than a
transient CLI flag: `both` runs a **single** electron transport per case and
stores both the incoherent `spec` and a `spec_coherent` from the same segments,
so a paired coherent-vs-incoherent comparison (peak/integrated-flux ratios) is
now one checkpoint with two spectra instead of two separate runs. The analysis
app exposes an Emission selector, gated per checkpoint on the stored spectra, to
switch every plot between the two.
