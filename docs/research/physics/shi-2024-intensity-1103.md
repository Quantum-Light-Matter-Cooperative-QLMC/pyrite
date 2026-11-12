# Sub-MeV PXR intensities against Shi et al. (2024) Fig. 2: attribution

**Status:** Research note for [#1103](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1103), 12 November 2026. It follows the [#527 transverse-recoil note](transverse-recoil-527.md) §5.1 and changes no production model. Evidence: `checks/shi_2024_matrix_elements_1103.py` (independent matrix elements), `checks/digitize_shi_2024_fig2_1103.py` (reference data), `tests/montecarlo/spectrum/test_shi_2024_fig2_intensity.py` (regression). Ledgered as `shi-2024-fig2-intensity-attribution` (`anchored`; [verification](../../validation/radiation-physics/shi-2024-fig2-intensity-attribution.md)).

**Finding.** The two codes are not computing the same quantity, so the 1.9–2.6× gap is not a PyRITE defect.

- Shi's Fig. 2 "CL" curve is the classical **PXR-only** rate: Eq. (1) with the trajectory deflection δv(t) dropped.
- Shi's "QM" curve carries PXR, coherent bremsstrahlung (CB) and their interference.
- PyRITE's classical line is PXR + CBS + interference.

At 30 keV, n_z = −2, PyRITE's classical peak is 26 % PXR, 31 % CBS and 43 % PXR×CBS interference. Comparing PyRITE's full line with Shi's PXR-only curve gives the reported 1.9× and 2.6×.

Shi's QM area exceeds its CL area for the same reason: CB is present in one curve and absent from the other. PyRITE's exact-root and classical areas agree, as they must when both include CBS.

The rest of the PXR-only difference is the crystal potential (cause 3 of the issue):

- PyRITE applies Si K-edge anomalous dispersion. At 1.42 keV, f′ = −1.78 reduces |χ₀₂₂|² to 0.64× of f₀² (0.60× with Debye–Waller).
- PyRITE also applies Debye–Waller damping and Waasmaier–Kirfel f₀.
- Shi's Salvat Yukawa χ_g has neither f′ nor Debye–Waller.

Swapping the couplings accounts for the PXR-only factor: Yukawa/PyRITE summed PXR is 1.99× at 30 keV n_z = −2, and 1.0–1.3× elsewhere.

## 1. Reference data

Fig. 2b/d/f are vector paths in the publisher PDF (filled stroke outlines; red = QM, black = CL). `checks/digitize_shi_2024_fig2_1103.py` does the following:

- rasterizes each outline at 40 px/pt and skeletonizes it, because the medial axis of a constant-width stroke is its centreline;
- calibrates the scale from tick-label spacing and the zero from the CL curve's flat floor;
- writes `tests/data/recoil/shi-2024-fig2.json`, which carries the PDF sha256 and conventions.

Peaks reproduce the issue's manual reads within 3 %. At 5 MeV the QM/CL area ratio is 1.13 / 1.06, close to the ≈1 that the correspondence limit requires.

**Axis caveat.** Panels d and f carry left and right y scales that differ by 3.3× (d) and 3.5× (f). Reading n_z = −3 on the right axis, which is the convention used here and in the issue, gives heights about 3× above an independent evaluation of Shi's own inputs. Reading it on the left axis gives agreement (30 keV tallest QM line 2.58e-10 against 2.61e-10 from the check and 2.92e-10 from the validator's line-shape model), although the 300 keV n_z = −3 CL peak favours neither axis. The n_z = −3 absolute heights are therefore not used as an anchor. QM/CL area ratios within one n_z group do not depend on the axis and are the comparison quantity.

## 2. Independent matrix elements

Shi SI Eqs. (S12)–(S17) are evaluated one reflection at a time. The convention is q_i + g = q_f + k, and the common factor −e√(ħ/2ε₀ωV) is removed. Second-order perturbation theory gives

$$
M_P=\frac{\mathbf j_{fi}}{c}\cdot\bigl[k^2\mathbf e-\mathbf K(\mathbf K\cdot\mathbf e)\bigr]\frac{\chi_{\mathbf g}}{K^2-k^2},\qquad
M_C=e\phi_{\mathbf g}\Bigl[\frac{\mathbf j_{f1}\cdot\mathbf e/c}{E_i-E_1}+\frac{\mathbf j_{2i}\cdot\mathbf e/c}{E_f-E_2}\Bigr],
$$

with K = k − g, intermediate momenta q₁ = q_i + g and q₂ = q_i − k, and transition current j_{q′q}/c = ħc(q + q′)/(E + E′). The line height of each row is ω Σ_σ |M|², up to one constant per electron energy.

- **Classical limit.** At 5 MeV, fed PyRITE's |χ_g| and |U_g|, the model matches PyRITE's `A_PXR`, `A_CBS` and their interference row by row. The PXR scale spread is ≤ 2.1 % and the CB/PXR and interference/PXR ratios are within 2.3 %. This independently confirms PyRITE's CBS amplitude and its PXR/CBS relative sign; the interference is constructive in both codes.
- **Quantum CB corrections.** Below 1 MeV the model's CB/PXR ratio exceeds PyRITE's by a median 1.36× (n_z = −2) and 1.19× (n_z = −3) at 30 keV, 1.08× / 1.05× at 300 keV and 1.01× / 1.01× at 5 MeV (n_z = −2 / −3). These are the exact intermediate-state denominators E_i − E_1 = −(ħv·g + ħ²g²/2γm) and the transition currents, which PyRITE's classical CBS omits. They are tens of percent, not the 2–6× in question.
- **Shi's stated approximation.** With u_q ≈ u_{q_i} every current becomes v_i. The two CB orderings then nearly cancel and the leading g·e (transverse-acceleration) term vanishes. That suppresses CB intensity to 0.06–0.35× of the classical value (amplitude 0.25–0.59×), and the resulting QM/CL area ratios are only 1.29 / 1.13 / 1.03, so it cannot produce Shi's QM excess. Shi's curves imply that CB was kept with momentum-dependent currents.

## 3. Shi's figure from Shi's inputs

The model uses the Yukawa potential, no f′, no Debye–Waller and |h|,|k| ≤ 8, scaled by the calibration constants above.

| T, n_z | Shi CL / model PXR-only CL | Shi CL / model PXR+CB CL | Shi QM/CL area | model QM (PXR+CB) / CL (PXR) | model, CB on both sides |
|---|---|---|---|---|---|
| 30 keV, −2 | **1.06** | 0.27 | 3.88 | **4.11** | 1.02 |
| 300 keV, −2 | **0.73** | 0.33 | 2.91 | **2.25** | 1.02 |
| 5 MeV, −2 | 1.16 | 1.02 | 1.13 | 1.14 | 1.00 |
| 30 keV, −3 | (axis) | (axis) | 5.83 | 4.47 | 1.04 |
| 300 keV, −3 | (axis) | (axis) | 4.31 | 2.26 | 1.02 |
| 5 MeV, −3 | 1.04 | 0.97 | 1.06 | 1.07 | 1.00 |

On the axis-unambiguous n_z = −2 lines, Shi's CL height is the PXR-only rate, and Shi's QM/CL area ratio is the full-CXR/PXR-only ratio. The tallest QM line also matches: 1.27e-9 against 1.12e-9 at 30 keV, and 5.31e-11 against 5.46e-11 at 300 keV, both on the (0,±2,2) root.

A factor of 0.5–0.8 remains in the 300 keV and n_z = −3 area ratios. It is shared by the model of Shi's own inputs, so it is not a PyRITE effect. It sits within what Shi's undisclosed CB implementation and figure windows could change.

## 4. PyRITE production

Production exact-root PXR+CBS area over classical-hook PXR-only area, against Shi's QM/CL area ratio:

| T | n_z = −2: PyRITE / Shi | n_z = −3: PyRITE / Shi |
|---|---|---|
| 30 keV | 4.12 / 3.88 | 3.00 / 5.83 |
| 300 keV | 1.93 / 2.91 | 1.99 / 4.31 |
| 5 MeV | 1.12 / 1.13 | 1.06 / 1.06 |

With CBS on both sides PyRITE's ratio is 0.89–1.07: the recoil Jacobian plus energy-dependent couplings evaluated at the shifted roots. The 30 keV n_z = −3 lines straddle the Si K edge, where |F|² and f″ change across the comb. The independent model, without anomalous dispersion, gives 1.00–1.04.

## 5. Consequences

- **No production change.** PyRITE's classical PXR/CBS coupling is not wrong; the comparison was between different observables. `pxr-amplitude` and `cbs-amplitude` are untouched, and no `*_MODEL` marker changes.
- **Quantum CB corrections are a separate, real gap.** The exact intermediate denominators and transition currents raise sub-100 keV CB by tens of percent per row, growing with g⊥. This is tracked as a follow-up issue, not attributed to #1103.
- **Absolute 30/300 keV intensities cannot be anchored to Shi.** Shi's crystal potential omits f′ and Debye–Waller, which are physical near the Si K edge, and the n_z = −3 axes are ambiguous. The anchor is the axis-independent QM/CL area ratio together with the MeV heights.
- **The #527 check is repaired.** Since #1099 the production line sits on the exact root. `checks/transverse_recoil_527.py` now takes its classical spectra under the ħ → 0 hook and reproduces every §5.2 number of the #527 note.
