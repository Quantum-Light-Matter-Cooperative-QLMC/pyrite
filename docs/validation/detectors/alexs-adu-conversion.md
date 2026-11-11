# `alexs-adu-conversion` — independent verification

**Code**: `src/pyrite/detectors/alexs_response.py::{signal_electrons, alexs_frame}`; `src/pyrite/detectors/spec.py::AlexS` (`score`). **Ledger claim**: charge-integrating CCD conversion $\mathrm{ADU}=g\,\min\!\big(\sum N\,\mathrm{QE}(E)\,E/W+I_\mathrm{dark}t,\;\mathrm{FW}\big)$ plus zero-mean Gaussian read noise of variance $(g\sigma_\mathrm{read})^2$ ADU², with optional Poisson draws of absorbed photons (fixed mean charge per photon) and dark electrons. **Source**: greateyes DS-ALEXs datasheet Rev05 (May 2025), 2k512 BI UV1 constants; photon-to-charge conversion at the mean pair-creation energy (Janesick, *Scientific Charge-Coupled Devices*, SPIE 2001).

Verifier: fresh context (2026-11-10). The derivation in sections 1–2 was written from the ledger row and the docstrings before the bodies of `alexs_frame` and `AlexS.score` were read. The body of `signal_electrons` is a single line directly after its docstring, so it was visible during that step.

## 1. Scope and inputs

- `signal_electrons(E_eV, photons, qe)`: expected collected charge [e⁻], or charge density per eV when `photons` is a density.
- `alexs_frame(signal_e, *, exposure_s, readout_rate_hz, gain_mode, photons, rng, shot_noise, read_noise, dark)`: per-pixel frame [ADU]. `signal_e` is the expected signal charge [e⁻]; `photons` is the expected number of *absorbed* photons per pixel.
- `AlexS.score(energy_eV, intrinsic_density, *, fwhm_eV, scale)`: expected ADU density per eV.
- Datasheet constants, as transcribed in the ledger and module: $W=3.66$ eV/e⁻, $\mathrm{FW}=10^5$ e⁻, $g\in\{1,\,0.34\}$ ADU/e⁻ (standard, high capacity), $\sigma_\mathrm{read}\in\{3.5,\,6.8,\,10.7\}$ e⁻ at $\{0.05,\,1,\,3\}$ MHz, $I_\mathrm{dark}=2.5\times10^{-4}$ e⁻ px⁻¹ s⁻¹ at −100 °C. The datasheet PDF itself was not available in this context. I checked these values for agreement across the ledger row, the module constants and the `test_datasheet_constants` anchor, and found them consistent. I did not re-read them from the vendor PDF.

## 2. Independent derivation

**Photon to charge.** A photon of energy $E$ absorbed in silicon produces on average $E/W$ electron-hole pairs, where $W$ is the mean pair-creation energy. The Fano variance $F\,E/W$ ($F\approx0.1$) is neglected. $N$ incident photons, each absorbed with probability $\mathrm{QE}(E)$, therefore give the mean signal charge

$$
\langle S\rangle = N\,\mathrm{QE}(E)\,\frac{E}{W}\quad[\mathrm{e^-}].
$$

Units: photons × dimensionless × eV / (eV/e⁻) = e⁻. For a spectral density $dN/dE$, the charge density $dS/dE=(dN/dE)\,\mathrm{QE}(E)\,E/W$ integrates over $E$ to the total charge. Summing over photons (or integrating over energy) is linear, as the ledger's $\sum$ states.

**Shot noise.** If incident photons are Poisson with mean $N$, Bernoulli thinning with probability $\mathrm{QE}$ leaves Poisson absorbed photons $n\sim\mathrm{Pois}(\bar n)$ with $\bar n=N\,\mathrm{QE}$. For monochromatic photons the charge is $S=n\,E/W$:

$$
\langle S\rangle=\bar n\,\frac{E}{W},\qquad \operatorname{Var}S=\bar n\left(\frac{E}{W}\right)^{2}.
$$

The draw must therefore act on the photon count, with a fixed charge $E/W$ per photon. Drawing electrons directly as $\mathrm{Pois}(\langle S\rangle)$ would understate the variance by the factor $W/E$. The ledger's "$N(E/W)^2$" uses $N$ for the absorbed-photon count $\bar n$.

For a polychromatic pixel (photon energies $E_i$ with absorbed means $\bar n_i$), the exact compound-Poisson variance is $\sum_i \bar n_i (E_i/W)^2$. Drawing the total count with one mean charge per photon $\bar q=\langle S\rangle/\bar n$ gives $\bar n\,\bar q^{2}\le\sum_i\bar n_i q_i^2$ (Jensen). The two are equal only when the pixel is monochromatic.

**Dark charge.** Thermal generation is a Poisson process at rate $I_\mathrm{dark}$, so $D\sim\mathrm{Pois}(I_\mathrm{dark}t)$, with $\langle D\rangle=\operatorname{Var}D=I_\mathrm{dark}t$ e⁻. Units: e⁻ px⁻¹ s⁻¹ × s = e⁻ px⁻¹.

**Order of operations.** The physical chain is:

1. Charge $Q=S+D$ accumulates in the pixel well. The well cannot hold more than FW, so $Q\to\min(Q,\mathrm{FW})$. Charge is non-negative.
2. The output amplifier adds read noise $\varepsilon\sim\mathcal N(0,\sigma_\mathrm{read}^2)$ in electrons. This happens *after* the well, so read noise is not clipped by FW.
3. The ADC multiplies by $g$ [ADU/e⁻].

$$
\mathrm{ADU}=g\left[\min\!\big(S+D,\ \mathrm{FW}\big)+\varepsilon\right],
\qquad
\langle\mathrm{ADU}\rangle_{\text{unsat}}=g\left(\bar n\frac{E}{W}+I_\mathrm{dark}t\right),
$$

$$
\operatorname{Var}\mathrm{ADU}_{\text{unsat}}=g^{2}\left[\bar n\left(\frac{E}{W}\right)^{2}+I_\mathrm{dark}t+\sigma_\mathrm{read}^{2}\right]\ \mathrm{ADU^2}.
$$

In saturation, $\langle\mathrm{ADU}\rangle=g\,\mathrm{FW}$ and $\operatorname{Var}=g^2\sigma_\mathrm{read}^2$.

**Score integral.** For a line of $N$ photons at $E_0$ with density $dN/dE$ (scaled by `scale`), the expected ADU density is $(dN/dE)\cdot\mathrm{scale}\cdot\mathrm{QE}(E)\,E/W\cdot g$. Its integral is $N\,\mathrm{scale}\,\mathrm{QE}(E_0)\,E_0\,g/W$ when QE and $E$ vary negligibly across the line. A charge-integrating CCD has no per-photon energy measurement, so an energy-resolution kernel (`fwhm_eV`) has nothing to act on. If one were applied, it would have to be unit-normalized to preserve the integral.

**Limiting cases (derived).**

| case | expected |
|---|---|
| $\mathrm{QE}=0$ | $S=0$; ADU is dark plus read noise only |
| one absorbed photon, $E=W$ | $S=1$ e⁻, ADU $=g$ |
| $S+D\ge\mathrm{FW}$ | ADU $=g\,\mathrm{FW}$ = $10^5$ (standard) or $3.4\times10^4$ (high capacity), plus noise $g\sigma_\mathrm{read}$ |
| $t=0$ | $D=0$ exactly |
| $\sigma_\mathrm{read}\to0$, no shot noise | deterministic $g\min(\langle S\rangle+I_\mathrm{dark}t,\mathrm{FW})$ |

## 3. Diff against the implementation

`signal_electrons`:

```python
return np.asarray(photons, dtype=float) * np.asarray(qe, dtype=float) * E / EV_PER_ELECTRON
```

This is identical to $N\,\mathrm{QE}\,E/W$.

`alexs_frame` (body, `alexs_response.py:182-203`):

```python
charge = np.clip(np.asarray(signal_e, dtype=float), 0.0, None)
if shot_noise:
    per_photon = np.divide(charge, expected, out=np.zeros_like(charge), where=expected > 0)
    charge = rng.poisson(expected) * per_photon
dark_e = DARK_CURRENT_E_PER_PX_S * float(exposure_s) if dark else 0.0
if dark_e > 0.0:
    if shot_noise:
        charge = charge + rng.poisson(dark_e, charge.shape)
    else:
        charge = charge + dark_e
charge = np.minimum(charge, FULL_WELL_E)
if read_noise:
    charge = charge + rng.normal(0.0, sigma, charge.shape)
return gain * charge
```

| element | derivation | code | result |
|---|---|---|---|
| shot noise | $n\sim\mathrm{Pois}(\bar n)$, charge $n\,\bar q$ | `rng.poisson(expected) * per_photon` | match |
| dark mean / draw | $I_\mathrm{dark}t$ / $\mathrm{Pois}(I_\mathrm{dark}t)$ | `DARK_CURRENT_E_PER_PX_S * t`; `rng.poisson(dark_e)` | match |
| clip | $\min(S+D,\mathrm{FW})$, $\ge0$ | `clip(.,0,None)` then `minimum(., FULL_WELL_E)` after dark | match |
| order | clip → read noise → gain | clip → `normal(0, sigma)` → `gain *` | match |
| read variance | $g^2\sigma_\mathrm{read}^2$ | `gain * (charge + N(0, sigma))` | match |
| $t=0$ | $D=0$ | `dark_e = 0`, branch skipped | match |

`AlexS.score` (`spec.py:365-372`) returns `intrinsic_density * scale * QE(E) * E / 3.66 * gain` and discards `fwhm_eV` (`del fwhm_eV`). This matches the derived expected-ADU density, and dropping the kernel is the right choice for a charge-integrating sensor.

## 4. Numeric spot checks

These were run with an independent Monte Carlo expectation computed from the section 2 formulas, not from code helpers. The script is in the verifier scratchpad and is not committed.

| check | independent | code | rel. diff |
|---|---|---|---|
| composite mean: $\bar n=40$, $E=900$ eV, $t=4\times10^4$ s, 1 MHz, $g=0.34$ ($n=4\times10^5$) | 3347.662 ADU | 3347.955 | $+8.7\times10^{-5}$ |
| composite variance, same case ($n=2\times10^6$, 3 seeds; $1\sigma\approx10^{-3}$) | $g^2[\bar n q^2+I t+\sigma^2]$ | — | $-1.5,\,-0.3,\,-0.5\times10^{-3}$ |
| saturated pixel ($5\times10^5$ e⁻), 3 MHz, $g=1$: mean / var | $10^5$ / 114.49 | 99999.96 / 114.40 | within MC error |
| $t=0$ dark; $\mathrm{QE}=0$ score | 0; 0 | 0; 0 | exact |
| `score` integral, top-hat line, scale 2, QE 0.6, both gains | trapezoid of $2\,(dN/dE)\,0.6\,E\,g/W$ | identical | $<10^{-12}$ |
| polychromatic pixel, half 500 eV / half 1000 eV: code var / exact compound var | $(q_1+q_2)^2/[2(q_1^2+q_2^2)]=0.900$ | 0.903 | as predicted |

The anchor file `tests/detectors/test_alexs_response.py` passes: 14 passed.

## 5. Findings

No exact factor, sign, exponent, unit or ordering diverges from the independent derivation. The following are model limitations, not discrepancies.

1. **Polychromatic-pixel shot noise (omission).** `alexs_frame` draws one Poisson count per pixel with the mean charge per photon `signal_e/photons` (`alexs_response.py:190-191`). For a pixel whose absorbed photons span energies, this underestimates the shot-noise variance by the Jensen factor $\bar q^2/\overline{q^2}$; it is 0.90 for an equal 500/1000 eV mix. For a dispersed spectrometer image, each pixel's bandwidth is a small fraction of $E$, so the error is second order in that fractional bandwidth and negligible. It matters for undispersed or broadband use. The docstring says "each drawn photon keeps the pixel's mean charge", but the ledger Notes do not state the monochromatic-pixel assumption.
2. **ADC saturation not modelled (already noted in the ledger).** In standard gain, $g\,\mathrm{FW}=10^5$ ADU. If the camera ADC is 16-bit, it would saturate at 65535 ADU below full well. I have not checked the ADC bit depth against the datasheet in this context. Until a camera-level ceiling is modelled, standard-gain frames above about 65.5 ke⁻ should be treated as unphysical.
3. **Read noise can drive ADU negative.** No bias offset or ADC floor is applied. This is consistent with a bias-subtracted frame and should be stated as that convention.
4. **Edge case.** If `photons == 0` while `signal_e > 0` with `shot_noise=True`, the signal charge is silently dropped (`np.divide(..., where=expected > 0)`, `alexs_response.py:190`). Consistent caller inputs, such as `grating_image.alexs_spectrometer_image`, never produce this.
5. **Fano noise neglected.** The variance $F\,\bar n\,E/W$ is smaller than the Poisson term $\bar n (E/W)^2$ by the factor $FW/E$, about $4\times10^{-4}$ at 1 keV. Neglecting it is justified for an integrating frame, as the docstring states.

## 6. Verdict

**`rederived`.** The implementation matches the independent derivation symbolically, dimensionally and numerically, and all ledgered limiting cases hold. Datasheet constants are consistent across the ledger, code and tests, but were not re-read from the vendor PDF in this context.
