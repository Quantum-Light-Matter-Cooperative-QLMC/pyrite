# Validation: mosaic-mc

## Claim

- **ID:** `mosaic-mc`
- **Symbol:** `montecarlo/spectrum/lines.py::mc_spectrum` (`mosaic_route="mc"`)
- **Source:** Gaussian mosaic-block orientation average; product Gauss--Hermite quadrature.
- **Intended quantity:** incoherent expectation of the full single-crystallite spectrum over two independent small tilts.

## Independent derivation

Let `S(E; dx, dy)` be the complete spectral intensity from a crystallite whose reciprocal vectors are rotated by tilt coordinates `(dx, dy)`.  For an isotropic Gaussian rocking distribution with per-axis standard deviation `sigma = eta_FWHM / (2 sqrt(2 ln 2))`, incoherent mosaic blocks require an intensity average, not an amplitude average:

```text
S_mosaic(E) = integral integral S(E; dx,dy)
              exp[-(dx^2+dy^2)/(2 sigma^2)]
              d dx d dy / (2 pi sigma^2).
```

With `dx = sqrt(2) sigma x`, `dy = sqrt(2) sigma y`, this becomes a product Gauss--Hermite integral:

```text
S_mosaic(E) ~= (1/pi) sum_i sum_j w_i w_j
                S(E; sqrt(2) sigma x_i, sqrt(2) sigma x_j),
```

where `(x_i,w_i)` integrate `exp(-x^2)`.  Thus normalized product weights are `W_ij = w_i w_j / pi` and must sum to one.  Each node must rotate the reciprocal vectors before recomputing resonance, polarization, amplitudes, and lineshape.

Units: tilt coordinates, `sigma`, and FWHM are radians (dimensionless); quadrature weights are dimensionless; output retains the spectrum units. Limits and symmetries: `eta -> 0` gives the perfect-crystal spectrum; one node lies at zero and gives the same result; constant `S` is preserved exactly; weights are positive and symmetric under either tilt reversal; integrated intensity is averaged incoherently.

## Implementation comparison

`_mosaic_quadrature` uses `hermgauss(nodes)`, `sigma = eta_FWHM / 2.3548200450309493`, nodes `sqrt(2) sigma x_i`, and per-axis weights `w_i/sqrt(pi)`.  The product rule is therefore exactly `w_i w_j/pi`.  `_small_tilt_R` is a proper Rodrigues rotation. `mc_spectrum` rotates `g` at every node, recomputes resonance, polarization, PXR/CBS amplitudes, lineshape, and energy-dependent attenuation, then adds intensities with the product weights.  It does not average complex amplitudes. This matches the independent expression term-for-term.

Independent numeric point: seven nodes per axis at `eta_FWHM=2 deg` gave weight sum error `0.0`; the weighted radial second moment matched `2 sigma^2` with relative error `1.2e-13`.  Focused tests cover identity, proper rotation, normalization, node scaling, and analytic/MC mutual exclusion; the combined focused run passed (`36 passed`).  The existing heavier check remains the line-shape/transport anchor and was not rerun for this analytic verification.

Follow-up applied after independent verification: both `mc_spectrum` and `_mosaic_quadrature` now contain `Validation: mosaic-mc`.

## Verdict

- **Claim**: `mosaic-mc` — `montecarlo/spectrum/lines.py::mc_spectrum` — Gaussian mosaic-block intensity average by product Gauss--Hermite quadrature
- **Filters**: units `pass`; limits `pass`; signs/conventions `pass`
- **Re-derivation**: `matches` — no divergent factor; normalization is exactly `1/pi`
- **Verdict**: `rederived`
- **Write-up**: `docs/validation/materials/mosaic-mc.md`
- **Suggested ledger change**: applied — status `rederived`, check retained as anchor, exact back-references added
