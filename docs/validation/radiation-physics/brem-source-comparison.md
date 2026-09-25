# Validation: `brem-source-comparison`

## Scope and question

Validation: `brem-source-comparison`. This is an implementation-context comparison, not a fresh-context re-derivation. It asks which bremsstrahlung cross-section source PyRITE should use in production: the packaged EEDL evaluation (current default) or the released BremsLib tables (#86, #84). Both are compared with an independent reference, the Seltzer–Berger tabulation, as PyRITE evaluates them in production (`montecarlo/spectrum/brem.py::_bremsstrahlung_dsigma_dk`). The angular model is not covered; no independent double-differential reference is used here.

## Reference

Seltzer and Berger, *At. Data Nucl. Data Tables* **35**, 345 (1986), tabulate the scaled cross section

$$
\chi(Z,T,\kappa)=\frac{\beta^2}{Z^2}\,k\,\frac{d\sigma}{dk}\quad[\mathrm{mb}],\qquad \kappa=k/T,
$$

for $Z=1$–100, $T=1$ keV–10 GeV (57 nodes) and 30 $\kappa$ nodes from 0 to 1. It includes electron–nucleus and electron–electron bremsstrahlung and underlies ESTAR's radiative stopping. PyRITE reads Seltzer's original `BREME.DAT` (NBS, 1984) from EGSnrc's copy, `HEN_HOUSE/data/nist_brems.data`, pinned by commit and SHA-256 and fetched on demand into the user data directory. It is not packaged.

Limiting check. With $\phi=\int_0^1\chi\,d\kappa$, radiative stopping is $S_\mathrm{rad}/\rho=(N_A/A)(Z^2/\beta^2)\,T\,\phi$. The table's trapezoid moment on its own nodes reproduces ESTAR within 0.5 % for C and W from 10 keV to 30 MeV. This confirms the layout (`kappa`-major blocks of `T` values per element), the units, and that electron–electron bremsstrahlung is included.

## Method

For each of the 24 BremsLib catalogue elements (Z = 5–83) and every table node from 1 keV to 30 MeV, `pyrite.validation.brem_sources.compare_sources` evaluates each model's $d\sigma/dk$ at the table's $\kappa$ nodes, with $\kappa=0$ taken at $10^{-3}$, and reduces it to $\chi$. It reports:

- the pointwise ratio model/reference;
- the hard-photon cross section $\sigma(\kappa>0.05)=(Z^2/\beta^2)\int\chi/\kappa\,d\kappa$;
- the first moment $\phi$.

The same trapezoid rule is applied on both sides. `checks/brem_source_comparison.py` runs the sweep and gates BremsLib. EEDL is reported but not gated.

## Results (2026-09-25)

BremsLib, pointwise over $0.05\le\kappa\le0.95$:

| Z | 1–8 keV | 10 keV–1 MeV | 1–30 MeV |
|---|---|---|---|
| 5–16 | 0.94–1.06 | 0.90–1.02 | 0.85–1.01 |
| 22–42 | 0.89–1.04 | 0.97–1.03 | 0.95–1.02 |
| 46–83 | 0.90–1.17 | 0.94–1.04 | 0.95–1.04 |

BremsLib first-moment ratios lie within 0.97–1.02 for Z ≥ 22 from 10 keV to 30 MeV. For Z ≤ 16 they fall to 0.94 at 1 MeV and 0.86 at 30 MeV.

- **Electron–electron share.** The low-Z shortfall is the electron–electron bremsstrahlung that Seltzer–Berger includes. BremsLib is an electron–atom partial-wave library and omits it. The share grows toward $1/Z$ of the nuclear term at high energy, and toward $\xi/Z$ with $\xi>1$ as $\kappa\to0$ under screening. That is why the $\kappa\to0$ node, outside the gate, reaches 0.70 for boron at 30 MeV.
- **Tip region.** $\kappa>0.95$ spans 0.69–1.12. It is not gated because the two grids resolve the tip differently.
- **Low energy.** 1–8 keV at high Z reaches 1.17 pointwise and 1.08 in moment. The gate starts at 10 keV.

EEDL, at its own incident panels (no interpolation): the first moment is 0.979–0.991 of Seltzer–Berger over 106 panels, so the packaged data agree and include electron–electron bremsstrahlung. Between panels, PyRITE's evaluation is not usable:

- pointwise ratios span 0.000–2.35, and moments 0.31–1.01;
- the File-26 spectra exist at only 8–10 decade-spaced incident energies per element (carbon: 14.1 keV, 251 keV, 1.19 MeV, 12.2 MeV between 10 keV and 30 MeV);
- the declared ENDF lin-lin (Cartesian) interpolation at fixed photon energy gives every $k$ above the lower panel's endpoint only the upper panel's share.

For carbon at 30 keV, $\chi$ is 0.09–0.15 of Seltzer–Berger for $\kappa\ge0.5$. This is an interpolation defect, not a data disagreement, and #174 owns the fix. A scratch unit-base ($\kappa$) interpolation brought the same nodes to within about 35 %, with the remaining error set by panel sparsity.

## Verdict

- **Claim:** `brem-source-comparison`. BremsLib reproduces Seltzer–Berger within the gated bands from 10 keV to 30 MeV. The lower band edge $1/(1+1/Z)$ is the omitted electron–electron share.
- **Filters:** units pass (ESTAR moment); limits pass (high-Z band closes to ±5 % pointwise and ±3 % in moment); conventions pass (same quadrature and node set on both sides).
- **Recommendation for #86/#84:** BremsLib is the more accurate production spectrum source throughout 10 keV–30 MeV, for all catalogue elements. At 30–300 keV, which covers the catalogue's beam energies, it agrees within 0.96–1.02 pointwise for every catalogue Z, and within 0.97–1.01 in moment for Z ≥ 14. Its only systematic deficit is electron–electron bremsstrahlung at low Z and MeV energies: 8–14 % of the moment at 30 MeV for Z ≤ 8. EEDL's accuracy is limited by its interpolation until #174 lands, and afterwards by panel sparsity.
- **Not covered:** the BremsLib angular shape; energies above 30 MeV; any full-track observable (#172).
- **Status:** `filtered`. Fresh-context verification is pending, and only a human may mark this signed off.
