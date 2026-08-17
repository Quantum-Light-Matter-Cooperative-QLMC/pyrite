# Bremsstrahlung

PyRITE models the smooth incoherent background emitted along transported
electron segments with a Born Bethe--Heitler cross-section and Elwert Coulomb
correction.

For element atomic number $Z$, electron kinetic energy $T$, and photon
energy $k$, the implemented energy-differential form is

```{math}
\frac{d\sigma}{dk}=\frac{16}{3}\alpha r_e^2 Z^2
\frac{1}{k p_i^2}\ln\!\left(\frac{p_i+p_f}{p_i-p_f}\right)f_E,
\qquad 0<k<T,
```

with relativistic momenta used as a weakly relativistic extension and

```{math}
f_E=\frac{\beta_i}{\beta_f}
\frac{1-e^{-2\pi\alpha Z/\beta_i}}
     {1-e^{-2\pi\alpha Z/\beta_f}}.
```

For a segment of length $L$ in number density $n_Z$, the contribution is
$n_Z L(d\sigma/dk)/(4\pi)$, multiplied by Beer--Lambert escape transmission.
Compound emission adds element contributions with their own $Z^2$ weighting.
The returned quantity is photons/(eV sr incident-electron).

## Limits and assumptions

- emission is isotropic, appropriate only to the intended weakly relativistic
  regime;
- the unscreened Born form is approximate, especially for high $Z$ or outside
  the tens-of-keV regime;
- `k <= 0` and `k >= T` are hard-zero bins;
- the infrared spectrum rises approximately as `ln(4T/k)/k` and therefore
  depends on the configured low-energy bound;
- Born alone vanishes at the tip, while Born times Elwert approaches a finite
  value immediately below the hard cutoff;
- segment contributions and incident electrons are summed incoherently;
- each row contributes one evaluation of the integrand rather than a quadrature
  along the flight. Under `energy_model="midpoint"` that evaluation uses the
  representative energy `E_repr_keV`, making it a midpoint rule; frozen rows keep
  the left-endpoint one. `brem_endpoint_quadrature_error` measures the
  difference, opt-in and read-only.

Self-absorption uses the same layered escape model as line radiation; see
[Multilayer materials](../materials/multilayer-materials.md).

Validation: `brem-spectrum`; the representative-energy evaluation is
`substep-radiation-invariance` and the estimator is
`radiation-error-estimators`. The full derivation, dimensional analysis, limits,
and numeric comparison are in [Bremsstrahlung spectrum
validation](../../validation/radiation-physics/brem-spectrum.md). Implementation
owner: `pyrite.montecarlo.spectrum.brem.mc_brem_spectrum`.
