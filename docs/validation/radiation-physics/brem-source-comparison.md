# Validation: `brem-source-comparison`

## Scope and question

Validation: `brem-source-comparison`. This is an implementation-context comparison, not a fresh-context re-derivation. It asks which bremsstrahlung cross-section source PyRITE should use in production: the packaged EEDL evaluation (current default) or the released BremsLib tables (#86, #84). Both are compared with an independent reference, the Seltzer–Berger tabulation, as PyRITE evaluates them in production (`montecarlo/spectrum/brem.py::_bremsstrahlung_dsigma_dk`). The angular model is not covered; no independent double-differential reference is used here.

## Reference

Seltzer and Berger, *At. Data Nucl. Data Tables* **35**, 345 (1986), tabulate the scaled cross section

$$
\chi(Z,T,\kappa)=\frac{\beta^2}{Z^2}\,k\,\frac{d\sigma}{dk}\quad[\mathrm{mb}],\qquad \kappa=k/T,
$$

for $Z=1$–100, $T=1$ keV–10 GeV (57 nodes) and 30 $\kappa$ nodes from 0 to 1. It includes electron–nucleus and electron–electron bremsstrahlung and underlies ESTAR's radiative stopping. PyRITE reads Seltzer's original `BREME.DAT` (NBS, 1984) from EGSnrc's copy, `HEN_HOUSE/data/nist_brems.data`, pinned by commit and SHA-256 and fetched on demand into the user data directory. It is not packaged.

Limiting check. With $\phi=\int_0^1\chi\,d\kappa$, radiative stopping is $S_\mathrm{rad}/\rho=(N_A/A)(10^{-27}\,\mathrm{cm^2/mb})(Z^2/\beta^2)\,T\,\phi$ in MeV cm²/g. The table's trapezoid moment on its own nodes reproduces ESTAR within 0.5 % for C and W from 10 keV to 30 MeV. This confirms the layout (`kappa`-major blocks of `T` values per element), the units, and that electron–electron bremsstrahlung is included.

## Fresh-context independent derivation (2026-09-25)

I derived the reduction from the cited Seltzer–Berger scaled table definition and the intended cross-section units before inspecting `brem_sources.py` or the production evaluator. The [Geant4 physics reference](https://geant4.web.cern.ch/documentation/pipelines/master/prm_html/PhysicsReferenceManual/electromagnetic/electron_incident/bremsstrahlung/ebrem.html) independently identifies the screened-nucleus and orbital-electron components of that reference cross section. [NIST ESTAR](https://pml.nist.gov/PhysRefData/Star/Text/method.html) reports mass radiative stopping in MeV cm²/g and uses the Seltzer–Berger cross sections. Neither source establishes the comparison's empirical tolerance bands; those remain measured gate criteria.

Let $T$ and $k$ be kinetic and photon energies in MeV, $m_ec^2=0.51099895$ MeV, and $\kappa=k/T$. Then $\beta^2=1-(1+T/(m_ec^2))^{-2}$. The table defines $\chi=(\beta^2/Z^2)k\,d\sigma/dk$ in millibarns. Solving for the differential cross section and changing variables with $dk=T\,d\kappa$ gives

$$
\frac{d\sigma}{dk}=\frac{Z^2}{\beta^2}\frac{\chi(Z,T,\kappa)}{k}
\quad[\mathrm{mb}/\mathrm{MeV}],
\qquad
\sigma_{>\kappa_c}=\frac{Z^2}{\beta^2}
\int_{\kappa_c}^{1}\frac{\chi(Z,T,\kappa)}{\kappa}\,d\kappa
\quad[\mathrm{mb}].
$$

The photon-energy-weighted first moment and mass radiative stopping are

$$
M_1=\int_0^T k\frac{d\sigma}{dk}\,dk
=\frac{Z^2}{\beta^2}T\phi,
\qquad
\phi=\int_0^1\chi(Z,T,\kappa)\,d\kappa,
$$

$$
\boxed{\frac{S_{\mathrm{rad}}}{\rho}
=\frac{N_A}{A}\,(10^{-27}\,\mathrm{cm^2/mb})
\frac{Z^2}{\beta^2}T\phi}
\quad[\mathrm{MeV}\,\mathrm{cm^2/g}].
$$

Here $A$ is the molar mass in g/mol. The $10^{-27}$ conversion is required because $\chi$ and $\phi$ are in mb. The equation in the ledger row omits this factor when it labels $S_{\mathrm{rad}}/\rho$ with ESTAR units; the corrected equation appears above. That is an exact dimensional discrepancy in the written equation, even though the implementation and its ESTAR test apply the factor.

At fixed positive $\kappa_c$, $\sigma_{>\kappa_c}$ is finite and nonnegative. A finite soft-photon limit of $\chi$ gives $d\sigma/dk\propto1/k$, so the cutoff-free zeroth moment diverges while $M_1$ remains finite. The photon endpoint is $k=T$; $\beta^2\simeq2T/(m_ec^2)$ for $T\ll m_ec^2$. The nuclear contribution scales roughly as $Z^2$ and the orbital-electron contribution roughly as $Z$. Consequently $1/(1+1/Z)$ is a useful nominal high-energy comparison floor, not a source identity or a universal bound: screening and $\kappa$ change its coefficient.

### Source-to-code comparison

Only after the derivation, I inspected `parse_seltzer_berger`, `beta_squared`, `model_chi`, `_hard_cross_section`, `compare_sources`, and the ESTAR anchor. The parser reshapes the table as element, $\kappa$, $T$; the comparison evaluates at the same table nodes. `model_chi` multiplies production $d\sigma/dk$ in cm²/eV by photon energy in eV and $10^{27}$ mb/cm², matching the table definition without an energy-unit factor. `beta_squared` matches the relativistic expression above. `compare_sources` uses the same trapezoid on both spectra for $\phi$ and for the hard $\chi/\kappa$ integral. Ratios cancel the common $Z^2/\beta^2$ factors and mb-to-cm² conversion; this makes the ratios valid but unable to catch the omitted factor in the written stopping-power equation.

The committed independent table excerpt pins six C/W incident-energy panels. Its ESTAR test explicitly multiplies the table moment by $10^{-27}$ and compares C at 0.1 MeV with 0.003414 MeV cm²/g and W at 10 MeV with 1.132 MeV cm²/g, among four other anchors. The historical full-catalogue ratio results below are preserved; this fresh-context audit did not reproduce that full sweep. The focused ESTAR test passed all six anchors when run with a writable Numba cache.

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

EEDL, at its own incident panels (no interpolation): the first moment is 0.979–0.991 of Seltzer–Berger over 106 panels, so the packaged data agree and include electron–electron bremsstrahlung. Before #174, PyRITE's between-panel evaluation was not usable:

- pointwise ratios span 0.000–2.35, and moments 0.31–1.01;
- the File-26 spectra exist at only 8–10 decade-spaced incident energies per element (carbon: 14.1 keV, 251 keV, 1.19 MeV, 12.2 MeV between 10 keV and 30 MeV);
- the declared ENDF lin-lin (Cartesian) interpolation at fixed photon energy gives every $k$ above the lower panel's endpoint only the upper panel's share.

For carbon at 30 keV, the former $\chi$ was 0.09–0.15 of Seltzer–Berger for $\kappa\ge0.5$. This was an interpolation defect, not a data disagreement. #174 subsequently replaced fixed-photon-energy interpolation with unit-base refinement; the post-fix catalogue comparison is 0.96–1.34 pointwise for $0.05\le\kappa\le0.95$ from 10 keV to 1 MeV, with 0.99–1.07 in first moment. The remaining spread reflects sparse EEDL panels.

## Implementation-context verdict before independent review

- **Claim:** `brem-source-comparison`. BremsLib reproduces Seltzer–Berger within the gated bands from 10 keV to 30 MeV. The lower band edge $1/(1+1/Z)$ is a nominal allowance for the omitted electron–electron contribution, not an exact universal share.
- **Filters:** units pass (ESTAR moment); limits pass (high-Z band closes to ±5 % pointwise and ±3 % in moment); conventions pass (same quadrature and node set on both sides).
- **Recommendation for #86/#84:** BremsLib is the more accurate production spectrum source throughout 10 keV–30 MeV, for all catalogue elements. At 30–300 keV, which covers the catalogue's beam energies, it agrees within 0.96–1.02 pointwise for every catalogue Z, and within 0.97–1.01 in moment for Z ≥ 14. Its only systematic deficit is electron–electron bremsstrahlung at low Z and MeV energies: 8–14 % of the moment at 30 MeV for Z ≤ 8. #174 removed EEDL's interpolation collapse; its remaining accuracy limit is panel sparsity.
- **Not covered:** the BremsLib angular shape; energies above 30 MeV; any full-track observable (#172).
- **Status at the original comparison:** `filtered`. The later independent verification and corrected ledger status are recorded below; only a human may mark this signed off.

## Fresh-context adjudication (2026-09-25)

- **Re-derivation:** The scaled cross-section reduction, hard zeroth moment, radiative first moment, and production comparison algebra match. The written mass-stopping equation differs by the exact factor $10^{-27}\,\mathrm{cm^2/mb}$.
- **Verdict:** `discrepancy` for the units filter of the ledgered stopping-power limiting check. This is a documentation equation error; the implementation and ESTAR anchor include the conversion. The historical `filtered` result above records the earlier comparison and remains unchanged here.
- **Suggested ledger edit:** Insert $10^{-27}\,\mathrm{cm^2/mb}$ in the row's $S_{\mathrm{rad}}/\rho$ equation and note that $1/(1+1/Z)$ is a nominal, $\kappa$-dependent electron–electron allowance rather than an exact physical share. A human should apply the ledger edit and reconsider status; only a human may mark `signed-off`.

## Post-correction recheck (2026-09-25)

The ledger now states $S_{\mathrm{rad}}/\rho=(N_A/A)(10^{-27}\,\mathrm{cm^2/mb})(Z^2/\beta^2)T\phi$ in MeV cm²/g and calls $1/(1+1/Z)$ a nominal allowance rather than an exact electron–electron share. With $\phi=\int_0^1\chi\,d\kappa$ in mb, $T$ in MeV, and $N_A/A$ in atoms/g, the units reduce to MeV cm²/g. This matches the independent derivation above and the six ESTAR anchors. The earlier missing-factor discrepancy remains recorded in the preceding section as history; the corrected row has no remaining unit discrepancy.

- **Filters:** units pass; limits pass for the stated gated domain and the finite first moment; signs/conventions pass. The $\kappa\to0$ hard-cross-section divergence and the non-universal electron–electron coefficient remain outside the gate or are described as allowances.
- **Re-derivation:** `matches` — the corrected ledger equation, hard zeroth moment, first moment, and production $d\sigma/dk$ reduction agree with the independent expressions and the inspected implementation.
- **Final verifier verdict:** `rederived` for `brem-source-comparison`. The historical full-catalogue numerical envelope is retained as implementation-context evidence; the focused ESTAR anchor passed six cases. The independent derivation does not extend to angular shape, energies above 30 MeV, or full-track observables.
- **Ledger status:** `rederived` after the corrected equation was independently rechecked. Only a human may mark `signed-off`.

## Source recommendation and domain for #86 (2026-09-25, implementation context)

Evidence: this page (Seltzer–Berger, spectrum and moments), `bremslib-angular-schiff` (angular shape), the boundary and fallback tests in `test_bremsstrahlung_bremslib.py` and `test_bremsstrahlung_eedl.py`. This is a recommendation from measurement; it changes no default, and only a human may sign off.

- **Recommendation.** BremsLib is the recommended production bremsstrahlung source for the 24 catalogue elements (Z = 5–83) from 10 keV to 30 MeV. It matches Seltzer–Berger within the gated bands, and it is the only source with an angular model that agrees with Schiff (5–30 MeV).
- **Tolerances.** Spectrum: pointwise $[1/(1+1/Z)-0.05,\,1.05]$ for $0.05\le k/T\le0.95$, first moment $[1/(1+1/Z)-0.03,\,1.03]$. Angle: $\theta_{50},\theta_{90}$ within ±6 % (Z < 46) and ±15 % (Z ≥ 46), 5–30 MeV, $0.1\le k/T\le0.8$.
- **Electron–electron gap.** EEDL includes it (first moment 0.979–0.991 of Seltzer–Berger at its own panels). BremsLib omits it: moment shortfall 8–14 % at 30 MeV for Z ≤ 8, 3 % or less for Z ≥ 22, and at $k/T\to0$ up to 0.70 for boron at 30 MeV. Disposition: a documented low-Z, MeV limitation, not a blocker, because catalogue beam energies (30–300 keV) agree within 0.96–1.02 pointwise. A correction term is optional and is not needed for the default; revisit it if MeV low-Z observables (#172) are affected.
- **EEDL fallback remains for:** elements outside the catalogue without a local checkout; incident energies above 30 MeV; missing or unfetched tables. `"eedl"` stays selectable. After #174 EEDL is 0.96–1.34 pointwise and 0.99–1.07 in first moment at 10 keV–1 MeV (panel sparsity), so it is a usable fallback but not the more accurate source.
- **Unvalidated.** The BremsLib angular shape below 5 MeV and against any tabulated or measured double-differential data; magnitudes above 30 MeV; full-track observables (#172); low-Z first moment at MeV energies beyond the stated shortfall.
- **Preconditions for the default flip (not this issue).** Public hosting of the catalogue archive; coordinated goldens and cache keys with #172. Characteristic production keeps the EEDL baseline (`eedl-shell-ionization-comparison`).
