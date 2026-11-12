---
orphan: true
---

# Planar absolute-yield and line-width anchors (#1057)

**Status:** research evidence for #1057, recorded against
[`cr-line-shape`](../../validation/ledger-channeling.md#cr-line-shape) and
[`cr-yield-assembly`](../../validation/ledger-channeling.md#cr-yield-assembly).
This note compares the model to measurements and does not change it. Harness:
`checks/planar_line_anchors.py`. Evidence:
`docs/validation/check-records/planar_line_anchors/` (SLURM arrays 11693,
11726, 12000).

## Sources

All sources come from Zotero collection Channeling Radiation (`YZQDRG55`).
The transcriptions and their stated conditions are in `tests/data/channeling/`.

| Source | Crystal, energy | Observable used | Conditions stated |
|---|---|---|---|
| Genz et al., PRB 53, 8922 (1996) | diamond (110) 5.2/9.0 MeV, (111) 5.2/9.0 MeV; 13/20/30 µm Ia, 55 µm IIa | absolute yields (Table I), FWHM (IV), centroids (VII) | divergence ~0.3 mrad, ΔE/E 2e-3, Si(Li) 2.6e-6 sr at 250 cm; detector resolution not stated |
| Kephart et al., PRB 40, 4249 (1989) | Si (110), (100), 16.92 MeV, 1–120 µm | absolute yield fits A(1−e^{−Z/L}), absorption-corrected | divergence inferred 2.2 mrad; no Si widths published |
| Azadegan et al., PRB 74, 045209 (2006) | diamond (110) 42.5 µm, 14.6/17/30/34 MeV | measured FWHM, detector FWHM, effective Θ_ms,ch | detector 0.47–0.63 keV FWHM |
| Klein et al., PRB 31, 68 (1985) | natural diamond (110)/(100) 23 µm, 16.9/30.5/54.5 MeV | measured FWHM (Table III), with their defect-free and 0.9 µm-defect calculations | read from the Zotero full-text cache; the PDF attachment is not on disk |

Watson & Koehler are excluded by the issue. Gouanère (1988) is the
existing intrinsic-width comparison in `cr-line-shape`.

## Method

`channeling_spectrum` at fixed energy, with the stated divergence, aperture,
energy spread and (where the source is not absorption-corrected) the
self-absorption layer. The model's divergence is constant in depth, so the
harness adds in-plane angular diffusion by linearity. Depth is split into
bins, and bin k contributes S(Z_k, σ_k) − S(Z_{k−1}, σ_k), where σ_k is the
in-plane RMS at the bin. Three σ_y models give a bracket. `kernel` uses the
solver's own planar-uniform decoherence rate weighted by bound populations.
`none` keeps the entry divergence. `amorphous` uses Highland. One full-depth
ρ(z) history is reused for every bin prefix. A no-reuse control (arm 26)
reproduces the reused result exactly (Y 0.00375, W 0.607 keV, E 2.981 keV).
Per line, the harness reads three values in a window around the line: the
bound-only yield, the FWHM (Gaussian detector smear where stated) and the
centroid.

Default grid: 8 Bloch cells, `H = 16`, `dz = 100 Å`, 6 bins. Convergence
arms (16 cells, `H = 24`, 50 Å, 12 bins) move yields ≤ 6% and widths ≤ 2%.

## Absolute yields

The Kephart Si(110) slopes are the yield per µm at entry, where dechanneling
has not acted yet:

| Line | Measured slope (photons/(e sr µm)) | Model/measured, 1 µm | 19 µm | 80–120 µm |
|---|---|---|---|---|
| 1→0 | 0.0031 | 1.03 | 0.77 | 0.65 |
| 2→1 | 0.0029 | 0.97 | 0.73 | 0.63 |
| 3→0 | 0.00027 | 1.35 | 1.11 | 1.03 |
| (100) 1→0 | 0.0031 | 0.72 | 0.62 | 0.61 |

**Entry anchor.** The Si(110) entry yields of the strong lines agree within
3%, and 3→0 is 35% high. The model's initial 1→0 population is 6.4%. Kephart
measured 5.5% and calculated 6.5%. **Thick crystals.** The model saturates
earlier than the measured fits. At 80–120 µm its plateau is 35% low for the
strong lines, so its effective occupation length is shorter than Kephart's
16–20 µm. Si(100) is 28% low already at entry.

The Genz diamond (110) 1→0 yield ratios at 9.0 MeV are 0.83, 0.56, 0.55 and
0.42 at 13, 20, 30 and 55 µm. At 5.2 MeV they are 1.17, 1.14, 0.69 and 0.40.
Genz's own theory needed a ×0.38 normalization for (111). The model's (111)
2→1 ratios at 5.2 MeV are 3.2/2.1/1.6/0.5, and the (111) yields carry that
same source ambiguity. Doubling the divergence changes the yields by ≤ 9%.

## Line widths

| Data | Line | Model FWHM (keV) | Measured (keV) | Ratio | Source's own theory (keV) |
|---|---|---|---|---|---|
| Genz (110) 9 MeV, 13 µm | 1→0 | 0.58 | 0.73 | 0.80 | 0.66 |
| Genz (110) 9 MeV, 55 µm | 1→0 | 0.65 | 0.93 | 0.69 | 0.72 |
| Genz (110) 5.2 MeV, 13–55 µm | 1→0 | — | — | 0.69–0.77 | — |
| Azadegan 42.5 µm, 14.6 MeV | 1→0 | 1.02 | 1.51 | 0.68 | — |
| Azadegan 42.5 µm, 17 MeV | 1→0 | 1.24 | 2.00 | 0.62 | — |
| Azadegan 42.5 µm, 30 MeV | 1→0 | 3.08 | 5.88 | 0.52 | — |
| Azadegan 42.5 µm, 34 MeV | 1→0 | 3.87 | 7.84 | 0.49 | — |
| Klein 23 µm, 16.9 MeV | 1→0 | 2.54 | 3.3 ± 0.4 | 0.77 | 2.3 |
| Klein 23 µm, 30.5 MeV | 1→0 | 3.02 | 6.0 ± 0.3 | 0.50 | 3.6 |
| Klein 23 µm, 30.5 MeV | 2→1 | 2.69 | 5.2 ± 0.3 | 0.52 | 2.2 |
| Klein 23 µm, 30.5 MeV | 3→2 | 5.80 | 4.8 ± 0.3 | 1.21 | 5.7 |

The model's centroids are within 3–6% for diamond (110) at 9 MeV, Azadegan
and Klein. At 5.2 MeV they sit 11–13% below Genz (2.98 vs 3.35–3.45 keV;
Genz's theory gives 3.2–3.27).

**Width deficit.** Measured natural-diamond widths are 20–50% wider than the
model, and the gap grows with energy (Azadegan 0.68 → 0.49, Klein 0.77 →
0.50). The model agrees with each source's own defect-free calculation to
within about 20%: 2.54/3.02/2.69/5.80 vs Klein's 2.3/3.6/2.2/5.7 keV, and
0.58–0.65 vs Genz's 0.66–0.72 keV. Klein reproduces the measured widths only
after adding crystal defects (0.9 µm spacing). Genz also notes that type IIa
and synthetic diamond have narrower lines than type Ia. The measured excess is
therefore defect-dominated and not specific to the Einstein TDS model.
Klein's 3→2 line is wider in both the model and Klein's calculation. It is
dispersive, so its FWHM reflects the band shape (see `cr-line-shape`).

## Systematics

- **In-plane angular model:** for Genz 9 MeV 13 µm, the `none`/`amorphous`
  bracket moves the width ratio to 0.74/0.79, against 0.80 for `kernel`. For
  Azadegan 30/34 MeV it moves to 0.56/0.49 and 0.76/0.70. Even the widest
  bracket does not reach the measurement. The kernel σ_y at exit, 8.7/7.6 mrad
  at 14.6/17 MeV, is the same order as Azadegan's depth-averaged effective
  Θ_ms,ch of 6.0/5.3 mrad.
- **Divergence ×2** (Genz): yields ≤ +9%, widths ≤ ±2%.
- **Detector resolution:** Azadegan's stated FWHM is applied. Genz does not
  state a resolution, so its widths are compared unsmeared, which favours the
  model. Klein's widths are the paper's unfolded values.
- **Absorption:** Genz Table I is read as absorbed yields (an interpretation),
  and self-absorption is applied. Kephart's fits are absorption-corrected, so
  no absorption is applied.
- **Grid:** the convergence arms change Kephart 120 µm 1→0 to 0.687 (from
  0.648), Genz (110) 9 MeV 13 µm to 0.84/0.80 and Klein 30.5 MeV yields by
  +3%, with widths unchanged.
- **TDS scale ×1.3** (a stand-in for the Andersen correlation factor of
  1.25–1.4 on the TDS kernel and in-plane weight): the width ratios rise by
  0.03–0.06 (Genz 9 MeV 13 µm 0.80 → 0.84, 55 µm 0.69 → 0.72; Azadegan
  14.6/17 MeV 0.68/0.62 → 0.74/0.68), and the yields fall by 8–15%. Correlated
  TDS at that size closes less than a fifth of the measured width gap.

## Conclusions

1. **Absolute yield:** Kephart Si(110) at 16.92 MeV anchors the entry
   bound→bound yield. The 1→0 and 2→1 lines agree within 3% at 1 µm, and
   3→0 is 35% high. Thick-crystal yields are 25–40% low. That deficit belongs
   to the dechanneling or occupation history (`cr-density-matrix-split-operator`
   and the TDS kernel), not to the assembly.
2. **Widths:** none of the four measured-width datasets matches within a
   defensible tolerance. All are natural or type Ia/IIa diamond, where defect
   broadening dominates, and no Si planar widths are available (Kephart
   publishes yields only). `cr-line-shape` therefore stays `rederived`.
3. **Planar correlated TDS:** do not open an issue now. A ×1.3 kernel changes
   widths by only 3–6% of their value, and the measured excess is
   defect-dominated. Revisit as an extension of #1045 only if a defect-free
   planar width dataset turns up: synthetic diamond, Si (110), or
   Gouanère-style intrinsic widths with an electronic kernel for diamond.
