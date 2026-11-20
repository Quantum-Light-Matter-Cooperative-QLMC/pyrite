# `brem-source-comparison`

## Scope and question

Validation: `brem-source-comparison`. This is an implementation-context comparison, not a fresh-context re-derivation. It asks which bremsstrahlung cross-section source PyRITE should use in production: the pinned EEDL evaluation (current default) or the released BremsLib tables (#86, #84). Both are compared with an independent reference, the Seltzer–Berger tabulation, as PyRITE evaluates them in production (`montecarlo/spectrum/brem.py::_bremsstrahlung_dsigma_dk`). The angular model is not covered; no independent double-differential reference is used here.

Current gate and evidence: [Gate domain and low-Z excess (#1068, 2026-11-10)](#gate-domain-and-low-z-excess-1068-2026-11-10), with Li's gate replaced by [Lithium attribution (#1073, 2026-11-20)](#lithium-attribution-1073-2026-11-20) and Li's pointwise reason re-cited in [Lithium after the sub-node fix (#1349, 2026-11-20)](#lithium-after-the-sub-node-fix-1349-2026-11-20). The sections dated 2026-09-25 measured 24 elements (Z = 5–83) with the electron–nucleus library alone; they are kept as history, and their electron–electron explanation of the low-Z deficit no longer applies to the staged tables.

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
- **Preconditions for the default flip (not this issue).** Coordinated goldens and cache keys with #172. Public archive hosting (Zenodo) is deferred and does not gate the flip; archives are distributed internally until then. Characteristic production keeps the EEDL baseline (`eedl-shell-ionization-comparison`).

## Update: electron–electron term added (#439, 2026-10-22)

The staged BremsLib tables now carry Seltzer and Berger's (1985) factor `1 + η/Z` (`bremslib-electron-electron`), so both sides of this comparison include electron–electron bremsstrahlung. The earlier sections describe the electron–nucleus library alone and are kept as history. `checks/brem_source_comparison.py` gates pointwise `χ` within `1 ± 0.13` for `0.05 ≤ k/T ≤ 0.95` and the first moment within `1 ± 0.06`. These bounds follow Seltzer–Berger's quoted uncertainty: ~10 % for the nuclear term below 2 MeV, 3–10 % at 2–50 MeV, and 5–10 % theory in the total.

Measured over the 24 catalogue elements, 10 keV–30 MeV:

- First moment: 0.974–1.057 (previously 0.86–1.03).
- Pointwise: 0.94–1.12.
- At `k/T → 0`: low Z is 0.97–1.11 (previously 0.70 for boron).

The largest values are B and C at `k/T ≤ 0.05` and 0.4–1 MeV. They persist at `k/T = 0`, where `η` is tabulated, so they are a BremsLib versus Seltzer–Berger electron–nucleus difference, not the fitted small-`k/T` shape. The electron–electron gap described above is closed; the remaining limitation is that residual.

## Gate domain and low-Z excess (#1068, 2026-11-10)

The catalogue (`xsgen/bremslib/release.py::catalogue_elements()`, every transport element) has grown to 42 elements, Z = 1–83. Under the previous gate (pointwise `1 ± 0.13`, moment `1 ± 0.06`) the exact recipe failed with 61 violations, all H and Li, and all excesses.

### Sources

- **SB86** — S. M. Seltzer and M. J. Berger, *At. Data Nucl. Data Tables* **35**, 345 (1986). p. 346: estimated `dσ/dk` uncertainty "3% to 5% in the high-energy region, T1 ≥ 50 MeV; 5% to 10% in the intermediate-energy region, 2 ≤ T1 ≤ 50 MeV; and about 10% … in the low-energy region, T1 < 2 MeV". p. 346: Pratt et al.'s tables cover Z between 2 and 92; p. 348: below 2 MeV the electron–nucleus values are those tables, and "linear extrapolation was used to extend their results to the cases Z = 1 and Z = 93–100"; 2–50 MeV is interpolated between those and the high-energy theory. Table II (p. 385 on) tabulates `η` and `η̄` for every Z = 1–100.
- **SB85** — *Nucl. Instrum. Methods B* **12**, 95 (1985). p. 96: the same linear extrapolation of Pratt et al. to Z = 1; Pratt et al. "conservatively estimate the overall uncertainty to be 10%". Table 4 gives `η` for Z = 6, 13, 29, 47, 74, 92 only; it is the source of PyRITE's staged factor.
- **Poškus 2025** — A. Poškus, *At. Data Nucl. Data Tables* **166**, 101734 (2025), the BremsLib v2.0 paper. p. 35: for Z = 1 and 2 with `T2 ≥ 0.5 MeV` and `k ≠ 0` the library uses the screened Born approximation (error about Z %). Elsewhere it quotes only computational uncertainties (Secs. 2.3–2.4) and contains no comparison with Seltzer–Berger; the library's own `RelErr` columns for Li are ≤ 0.2 % at 0.4–1 MeV, about 1.4 % at 2–5 MeV and 3–6 % at 10–30 MeV. Electron–electron bremsstrahlung is not in the library (Sec. 1).

### Root cause

Both sides include electron–electron bremsstrahlung with nearly the same `η`. PyRITE holds SB85's Z = 6 row below Z = 6; SB86's own rows are close to it:

| `η̄` | 0.5 MeV | 30 MeV |
|---|---|---|
| SB86 Z = 1 | 0.451 | 1.029 |
| SB86 Z = 3 | 0.445 | 1.060 |
| SB85 Z = 6 (held) | 0.435 | 1.056 |

Substituting SB86's rows moves H's total by +1.1 % at 0.5 MeV and −1.3 % at 30 MeV (at most 1.3 % up to 30 MeV, 1.6 % at 80–100 MeV), and Li's by at most 0.3 %. The low-Z excess is therefore in the electron–nucleus term: BremsLib exceeds SB86's by about 5 % for B–O (already recorded in the update above), more for Li, and most for H, largest at `k/T = 0.05`. Reference identity (pinned SHA), table coverage (no fallback warnings; H and Li digests `660fe488…`, `1f148244…`) and interpolation in `T` were ruled out.

### Gate

`checks/brem_source_comparison.py` now holds:

- `MOMENT_TOLERANCE_BANDS`: first moment within `1 ± 0.10` for 10 keV–30 MeV, the upper end of SB86 p. 346's quoted ranges below 50 MeV, and the existing `1 ± 0.06` for the extension rows, 30 < T ≤ 100 MeV (`bremslib-high-energy-extension`, #439). Pointwise stays `1 ± 0.13`.
- `UNGATED_ELEMENTS`, printed and written to the JSON report with a reason each:
  - **H (Z = 1):** the reference electron–nucleus value is extrapolated outside Pratt et al.'s Z = 2–92 (SB86 p. 348, Z = 2–92 range p. 346; SB85 p. 96), and BremsLib uses the screened Born approximation for Z ≤ 2 (Poškus 2025 p. 35).
  - **Li (Z = 3):** an explicit policy choice, not a sourced exclusion — a measured BremsLib-versus-SB86 electron–nucleus discrepancy with no known cause, tracked in #1073. Superseded by [#1073](#lithium-attribution-1073-2026-11-20): Li's band moment is now gated, and its pointwise and extension-moment exclusions cite a measured cause.

### Results (2026-11-10)

Exact recipe, `PYRITE_MC_BACKEND=cpu`, exit 0, `PASS`. BremsLib, `0.05 ≤ k/T ≤ 0.95`:

| Z | pointwise, 10 keV–30 MeV | moment, 10 keV–30 MeV | moment, 30 < T ≤ 100 MeV |
|---|---|---|---|
| 5–20 | 0.957–1.121 | 0.981–1.057 | 1.026–1.058 |
| 22–42 | 0.955–1.057 | 0.986–1.033 | 1.022–1.031 |
| 46–83 | 0.942–1.053 | 0.973–1.035 | 1.028–1.055 |
| H (reported) | 0.959–1.266 | 1.013–1.104 | 1.108–1.134 |
| Li (reported) | 0.959–1.139 | 0.989–1.063 | 1.059–1.072 |

Below 10 keV (reported): 0.78–1.27 pointwise, 0.95–1.08 in moment. EEDL at its own panels: 0.979–0.991 in moment over 185 panels.

### Fresh-context verification (2026-11-10)

An independent verifier read SB86, SB85 and Poškus 2025 directly (rendering SB86 Table II and SB85 p. 96), recomputed the `η̄` substitution, inspected `MOMENT_TOLERANCE_BANDS`, `UNGATED_ELEMENTS`, `moment_tolerance` and `gate_violations`, and reran the recipe (exit 0, `PASS`, 42 elements) and the module tests (17 passed).

- **Filters:** units pass (dimensionless ratios); limits pass (the electron–electron factor tends to 1 at high Z); signs and conventions pass (`1 + η/Z`; BremsLib over Seltzer–Berger).
- **Re-derivation:** `matches`. SB86 p. 346 uncertainties, the p. 348 Z = 1 extrapolation (Z = 2–92 range on p. 346), Table II `η̄` for Z = 1 and 3, SB85 Table 4 `η̄` for Z = 6 and Poškus p. 35 all read as stated. Substituting SB86's `η̄` moves H by at most 1.3 % up to 30 MeV (1.6 % at 80–100 MeV) and Li by at most 0.29 %, so the excess is electron–nucleus.
- **Boundaries:** 10 keV gated at 0.10 and lower energies never gated; 30 MeV takes the 0.10 band and keeps the pointwise gate; `30 < T ≤ 100 MeV` gates only the moment at 0.06; EEDL and ungated elements never fail and are reported with reasons.
- **Corrections applied:** SB85's Z = 1 extrapolation and Pratt et al.'s 10 % are on p. 96, not p. 97; the SB86 citation is p. 348 with the Z = 2–92 range on p. 346; the Li computational uncertainty comes from the library's `RelErr` columns, not the paper; the `η̄` shift is scoped to energy.
- **Verdict:** `rederived`; with the gate-domain tests in `tests/montecarlo/brem/test_brem_source_comparison.py` the row is `anchored`. Li's exclusion remains a policy choice (#1073). Only a human may mark this signed off.

## Lithium attribution (#1073, 2026-11-20)

#1068 left Li (Z = 3) ungated as a policy choice, because no source explained its excess. This section attributes the excess, splits it into a staging part and a library-versus-reference part, and replaces the policy choice with cited reasons. It changes no physics and no default numerical output; only the gate domain changes.

### Sources

- **SB85 pp. 96–97** (Pratt et al.'s method, as quoted by SB85). Pratt et al. (ADNDT 20, 175 (1977)) computed partial-wave results only for Z = 2, 8, 13, 47, 79 and 92, at `T1` typically 1, 5, 10, 50, 100, 500, 1000 and 2000 keV and at `k/T1` typically 0.95, 0.8, 0.6, 0.4 and 0.0. They filled in 2 ≤ Z ≤ 92 and 0 ≤ `k/T1` ≤ 1 by least-squares cubic-spline fits of the ratio to screened, Coulomb-corrected Born theory. Their overall uncertainty is 10 %. All of this is on p. 96. The footnote, which starts on p. 96, adds a low-energy exception from the energy-independent exchange potential. Its numbers are on p. 97: without exchange, the `k = 0` cross section at 1 keV is 16 % lower for Li, and the difference fades by about 100 keV for low Z. So in the reference, Li is an interpolation between computed Z = 2 and Z = 8, and `k/T1 = 0.05` is an interpolation between computed 0 and 0.4. The exchange caveat raises the reference, the opposite sign to Li's excess, and does not apply above about 100 keV.
- **SB86 pp. 347–348.** For `T1 ≥ 50 MeV` the electron–nucleus term is analytic high-energy theory: Bethe–Heitler, plus a screening correction using Hubbell et al.'s form factors (their 1975 tables for Z ≤ 6), plus a Coulomb correction (p. 347). From 2 to 50 MeV it is a spline interpolation between that theory and Pratt et al. (p. 348). **SB86 p. 346** gives the uncertainty: 3–5 % at `T1 ≥ 50 MeV` and 5–10 % at 2–50 MeV.
- **BremsLib v2.0 manual (library version 2.0.8), p. 3.** For `k/T1 ≥ 0.1` and `T1 ≥ 50 MeV`, the SDCS integrates an analytic Sommerfeld–Maue triple-differential cross section with next-to-leading-order and screening corrections. Its `RelErr` column is zero there.
- **Interpolate_DCS v1.0.7 user's manual, Sec. 2, p. 3** (shipped with BremsLib v2.0.8). Between the first two library nodes (`k/T1` = 0 and 0.1), "a simple spline fitting or interpolation typically causes relative errors greater than 10 % when k/T1 is of the order of 0.01 (and linear interpolation is even less accurate)". The manual prescribes cubic-spline fitting of `SDCS/SDCS_Born`, multiplied by the screened Born SDCS. The Born SDCS is tabulated on 20 `k/T1` nodes, including 10 below 0.1, in `SDCS/Born_appr/`, for `T1 ≥ 1 MeV`. Below 1 MeV, "simple spline fitting of the original grid values … is sufficiently accurate". The >10 % sentence is stated for the DDCS at large angles and `T1 ≳ 10 MeV`; the manual does not quantify linear-interpolation error at 0.4–1 MeV. It supports only that linear interpolation is less accurate than the prescribed spline scheme; the measured comparison below (Interpolate_DCS itself) is the evidence for the size of the effect.
- Pratt et al. (1977) is not in the project Zotero library: the local database has no item, creator or attachment that matches. The SB85 and SB86 quotations above are the secondary record of it. They are enough for the decision below, because both of Li's remaining exclusions lie outside what Pratt et al. cover.

### Method

These are local CPU probes, capped at 8 GiB. They use the pinned Seltzer–Berger table, the released Li/B/C/N/O tables and the raw BremsLib v2.0.8 files: `SDCS/SDCS_<Z>.txt`, `SDCS/Born_appr/SDCS_Born_<Z>.txt`.

- **Raw.** `beta**2` times the raw library SDCS, linear in `k/T1` on its own nodes. It is multiplied by the same `1 + eta/Z` that the staged tables carry, then divided by SB's `chi`, on the SB nodes. Both sides use the same `eta`, so the ratio is the electron–nucleus ratio up to the ≤ 0.3 % SB86-versus-SB85 `eta` difference for Li (#1068).
- **Staged.** `model_chi` on the released tables.
- **Interpolate_DCS.** The library's own interpolator, Interpolate_DCS v1.0.7, run with its default parameters on Li at `k/T = 0.05` (0.4–30 MeV). Below 1 MeV its default is a least-squares spline over the full `k/T1` range (knots 0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.9, 0.975, 1 from `k_T1_knots1.txt`). These values are quoted wherever a number at `k/T = 0.05` is given.
- **Interpolating spline.** An approximation of the prescription with an interpolating (not least-squares) not-a-knot cubic spline, used for the full-grid scans. Below 1 MeV it splines the SDCS over `k/T1` = 0–0.9; there the result depends on the end condition (a natural spline gives Li 1.112–1.115 at 0.4–0.8 MeV, not-a-knot 1.093–1.096, Interpolate_DCS 1.072–1.077), so below 1 MeV it only bounds the effect from below. At 1 MeV and above it splines `SDCS/SDCS_Born` up to `k/T1 = 0.7` and the SDCS over 0.4–0.975 above that, and matches Interpolate_DCS within 0.004. A second variant, linear in `SDCS/SDCS_Born`, cross-checks it.

### Results

**Staging is not the excess.** Raw and staged ratios agree within 0.5 % at every probed node and within 0.001 in moment, for Z = 3, 5, 6, 7 and 8 from 10 keV to 30 MeV. Reference identity, table digest and `T` interpolation were already excluded (#1068).

**Li continues the low-Z trend and is not an outlier.** Electron–nucleus ratios at 0.5 MeV are on library nodes for BremsLib and on table nodes for SB:

| Z | `k/T` = 0 | 0.1 | 0.2 | 0.4 | moment |
|---|---|---|---|---|---|
| 3 Li | 1.119 | 1.067 | 1.051 | 1.027 | 1.062 |
| 5 B | 1.114 | 1.062 | 1.048 | 1.029 | 1.056 |
| 6 C | 1.107 | 1.060 | 1.047 | 1.030 | 1.054 |
| 7 N | 1.096 | 1.057 | 1.046 | 1.029 | 1.050 |
| 8 O | 1.086 | 1.057 | 1.045 | 1.029 | 1.048 |

O is one of Pratt et al.'s computed atomic numbers, and `k/T = 0` is one of their computed nodes. O still shows the same excess (1.086 at `k/T = 0`). So the shared excess is a library-versus-reference difference, not an artefact of Pratt et al.'s interpolation in Z. Li is 0.5 % above B at the nodes, and every node is within the reference's 10 %, except `k/T = 0` (1.12), which is not gated.

**The pointwise failure comes from interpolation in `k/T`.** The gated maximum always falls at `k/T = 0.05`. BremsLib has no node there, and PyRITE interpolates linearly between `k/T` = 0 and 0.1, which is less accurate than the spline scheme the library manual prescribes for that interval. The odd-node zigzag (`k/T` = 0.15, 0.25, 0.35 above both neighbours) has the same cause.

| Li, `k/T = 0.05` | 0.4 MeV | 0.5 | 0.6 | 0.8 | 1.0 | 2.0 | 10 | 30 |
|---|---|---|---|---|---|---|---|---|
| linear (production) | 1.134 | 1.137 | 1.137 | 1.134 | 1.130 | 1.113 | 1.099 | 1.128 |
| Interpolate_DCS v1.0.7 (default parameters) | 1.077 | 1.076 | 1.075 | 1.072 | 1.074 | 1.069 | 1.081 | 1.125 |
| interpolating spline (not-a-knot) | 1.095 | 1.096 | 1.095 | 1.093 | 1.074 | 1.069 | 1.077 | 1.126 |

On the full SB grid from 10 keV to 30 MeV, over `0.05 ≤ k/T ≤ 0.95`, Li spans 0.952–1.137 linear and 0.964–1.126 under the interpolating spline; the Born-ratio-linear variant gives at most 1.123. Under the library's own interpolation (Interpolate_DCS ≤ 1.125 at the probed energies, and below the interpolating spline at 0.4–0.8 MeV), Li passes the ±13 % gate everywhere. Interpolate_DCS also removes the odd-node zigzag at `k/T = 0.15` (1.060–1.065 against 1.076–1.080 linear). The same artefact puts B (1.121) and C (1.112) near the edge; under the interpolating spline C drops to 1.097 and O from 1.094 to 1.084. Li's band moment falls by about 0.01 (1.062 → 1.053 at 0.5 MeV).

**Upstream data defect (B).** `SDCS_Born_5.txt` has a zero at `T1 = 3 MeV`, `k/T1 = 0.0046`. Any Born-ratio staging must guard against it: either variant gives B 1.185 at 3 MeV, `k/T = 0.15`.

**The extension moment is a smooth function of Z.** At 30–100 MeV the first moment does not depend on interpolation (linear and interpolating spline agree within 0.001 at 20–30 MeV). It rises monotonically as Z falls:

| first moment | 30 MeV | 50 MeV | 100 MeV |
|---|---|---|---|
| Al | 1.035 | 1.032 | 1.035 |
| O | 1.041 | 1.040 | 1.045 |
| C | 1.046 | 1.047 | 1.053 |
| B | 1.049 | 1.051 | 1.058 |
| Li | 1.058 | 1.063 | 1.072 |
| H | 1.103 | 1.116 | 1.134 |

The excess grows with `k/T` falling toward 0.1. At 100 MeV, Li is 1.038 at `k/T = 0.8` and 1.095 at `k/T = 0.1`. Above 50 MeV both sides are analytic theories, with different screening: Hubbell form factors in SB, a Kohn–Sham potential in BremsLib. That difference is a plausible cause, but no source quantifies it. At 30–50 MeV the reference is instead a spline interpolation between that theory and Pratt et al. (SB86 p. 348) with 5–10 % uncertainty (SB86 p. 346); Li's 40 MeV moment, 1.059, lies within it. Above 50 MeV the reference alone carries 3–5 %, and no source bounds the difference within the ±6 % `bremslib-high-energy-extension` gate.

### Classification and gate

- **Pointwise, 10 keV–30 MeV: PyRITE staging.** Linear interpolation in `k/T` below 0.1 raises Li by 0.056–0.062 at 0.4–1 MeV against Interpolate_DCS (1.130–1.137 against 1.072–1.077; the not-a-knot interpolating spline gives a smaller 0.039–0.056, depending on its end condition), on top of a library-versus-reference difference of 6–12 % at the library nodes. The upstream manual prescribes the more accurate scheme; Interpolate_DCS measures the difference. Correcting it changes every element's spectrum below `k/T = 0.1`. It needs a `BREMSSTRAHLUNG_BREMSLIB_MODEL` bump and identity-pin re-mint, not a tables tag, so it is out of scope here and proposed as a follow-up. Li's pointwise gate is reported, not gated, with this reason (`UNGATED_KINDS[3]["chi"]`). Superseded by [#1349](#lithium-after-the-sub-node-fix-1349-2026-11-20): the staging fix landed, and the remaining exclusion cites the library's screening model.
- **Band moment, 10 keV–30 MeV: gated again.** Li is 0.989–1.063, within ±0.10 (SB86 p. 346).
- **Extension moment, 30–100 MeV: library versus reference.** Li is 1.059–1.072. The cause is cited and shown to continue the low-Z trend: above 50 MeV, two analytic high-energy theories (SB86 p. 347 and the BremsLib manual p. 3); at 30–50 MeV, within SB's 5–10 % interpolation uncertainty (SB86 pp. 346, 348). It is reported, not gated (`UNGATED_KINDS[3]["extension_moment"]`), because no source bounds it within ±6 %. H stays fully ungated (#1068).

The exact recipe, `PYRITE_MC_BACKEND=cpu uv run python checks/brem_source_comparison.py --download`, exits 0 with `PASS`. All 42 catalogue elements are compared and Li is flagged `partly ungated`. Li reports pointwise 0.959–1.139, band moment 0.989–1.063 (gated) and extension moment 1.059–1.072.

The gate-domain change is pinned by `test_gate_domain_ungates_h_and_two_cited_li_kinds` and `test_gate_violations_gate_the_li_band_moment_only`. No physics equation changed.

### Fresh-context verification (2026-11-20)

An independent verifier read the Interpolate_DCS v1.0.7 manual (Sec. 2, p. 3), SB86 pp. 346–348, SB85 pp. 96–97 and the BremsLib v2.0 manual (p. 3, including its `RelErr` column) directly; reproduced the node ratios, the `k/T = 0.05` values (also with the Interpolate_DCS executable) and the extension moments from the raw library files and an independent parse of the SB table; inspected `UNGATED_KINDS` and the `gate_violations` boundaries (exactly 30 MeV is band, so its moment is gated); and reran the recipe (exit 0, `PASS`) and the module tests.

- **Verdict:** `rederived` for the gate-domain change. The row stays `anchored`.
- **Corrections applied:** library-scheme values below 1 MeV now quote Interpolate_DCS (1.072–1.077, so linear adds 0.056–0.062 to the ratio, not 3.5–4.0 %), with the interpolating spline's end-condition sensitivity noted; the manual is cited as prescribing a more accurate scheme, not as calling linear inaccurate at 0.4–1 MeV; "two analytic theories" is scoped to `T ≥ 50 MeV`, with 30–50 MeV covered by SB's 5–10 % interpolation uncertainty; the 2–50 MeV interpolation is SB86 p. 348 and the Li exchange figure SB85 p. 97; the interpolating-spline full-grid minimum is 0.964, not 0.953. The ≥ 1 MeV figures and the ≤ 1.126 bound stand.

Only a human may mark this signed off.

## Lithium after the sub-node fix (#1349, 2026-11-20)

#1349 replaced the linear `k/T1` sub-nodes with the library's own Interpolate_DCS scheme (`tables-bremslib-4`, `bremslib-electron-electron`). The recipe still exits 0 with `PASS`. Li's pointwise range is 0.959–1.130 and its band moment 0.989–1.058. At 0.4–10 MeV the `k/T = 0.05` maximum fell from 1.13–1.14 to ≤ 1.085, and B (1.121 → 1.104) and C (1.112 → 1.098) fell too. One gated point is left: 30 MeV, `kappa = 0.05`, 1.1301, which is 1e-4 outside ±0.13. This section finds its cause. It changes no physics and no default numerical output.

### Sources

- **Y.-S. Tsai, Rev. Mod. Phys. 46, 815 (1974), Eq. (3.66) and Table B.2.** In complete screening the electron–nucleus spectrum is `chi = 4 α r_e² [(4/3 − 4y/3 + y²)(L_rad − f(Z)) + (1 − y)/9]`, `y = k/E`. For Z ≤ 4, `L_rad` comes from Hartree–Fock form factors (H 5.31, He 4.79, Li 4.74, Be 4.71). For Z ≥ 5, `L_rad = ln(184.15 Z^−1/3)` (Thomas–Fermi–Moliere). The same `L_rad` already backs the radiation-length claim in the ledger.
- **Poškus, ADNDT 166, 101734 (2025).** The atom is a static charge distribution, and the continuum electron sees a Kohn–Sham potential (Sec. 2, and Sec. 2.1 for the screening function `Ṽ(r)`, the screened potential divided by the unscreened one). The library's screened Born DDCS is "calculated using the same screening function `Ṽ(r)` as the partial-wave DDCS" (Sec. 2.6). The Kohn–Sham references [14,15] were not read here, so the exchange–correlation functional is not known. The probe below brackets it with LDA exchange alone and with exchange plus correlation.
- **Slater's rules** (Phys. Rev. 36, 57 (1930)) give the analytic Slater-type-orbital densities used for Li, C and Al. H uses its exact 1s density, and He the hydrogenic `ζ = 27/16`. The LDA functional is Slater exchange plus Perdew–Zunger correlation (Phys. Rev. B 23, 5048 (1981)), spin-unpolarized.

### Measurement: BremsLib exceeds the complete-screening limit

At `k → 0` and 30 MeV, screening is complete: the minimum momentum transfer is far below the atomic scale. Here `beta²` times the raw library SDCS (`SDCS_<Z>.txt`, `Born_appr/SDCS_Born_<Z>.txt`, `k/T1 = 0`) is compared with Tsai's `chi(y = 0)`. SB86's electron–nucleus value is SB86's total divided by the staged `1 + eta/Z`:

| Z | 1 H | 2 He | 3 Li | 4 Be | 5 B | 6 C | 8 O | 13 Al | 29 Cu |
|---|---|---|---|---|---|---|---|---|---|
| BremsLib PW / Tsai | 1.195 | 1.114 | 1.110 | 1.093 | 1.079 | 1.071 | 1.051 | 1.031 | 1.020 |
| BremsLib Born / Tsai | 1.195 | 1.114 | 1.110 | 1.092 | 1.078 | 1.070 | 1.051 | 1.031 | 1.026 |
| SB86 e–n / Tsai | 0.942 | 0.990 | 1.001 | 1.001 | 1.000 | 1.000 | 0.996 | 0.992 | 0.993 |

SB86 reproduces Tsai within 0.1 % for Z = 3–6, so the reference agrees with Hartree–Fock and Thomas–Fermi complete screening. BremsLib's PW and Born agree with each other within 0.1 % (Cu 0.6 %) and both exceed Tsai. The excess grows as Z falls. It is not a partial-wave or interpolation effect: it is present in the library's own Born values at a library node. At 30 MeV, BremsLib-Born divided by Bethe's intermediate-screening formula falls from 1.086 (Li) at `k → 0` to ≈ 1.0 at `k/T ≈ 0.8–0.9`, where screening stops mattering. So the excess comes from small momentum transfer, that is, from screening. At Li's failing point (30 MeV, `k/T = 0.05`) the minimum momentum transfer is ≈ 0.06 a.u., which is still inside the screening region.

### Cause: exchange–correlation in the Kohn–Sham screening function

In Born approximation with complete screening, `L_rad = 1 + ∫ (dq/q) [Z_eff(q)/Z]²` from 0 to `m c`, where `Z_eff(q)/Z = 1 + ∫₀^∞ Ṽ′(r) cos(qr) dr`. With an electrostatic screening function (nucleus plus electron cloud) and the densities above, this gives H 5.310, He 4.787 and Li 4.729, against Tsai's 5.31, 4.79 and 4.74. It gives C 4.598 and Al 4.284, against the Thomas–Fermi 4.618 and 4.361. A Kohn–Sham potential adds the exchange–correlation potential, `Ṽ_KS = Ṽ_es − r v_xc[ρ]/Z`. That potential is attractive and decays as `ρ^{1/3}`, so its range is longer than the charge cloud's. It undoes part of the screening. For one-electron H, the exact `v_xc` cancels the Hartree self-repulsion completely. On the same densities the predicted ratio `chi_KS/chi_es` at `y = 0` is:

| Z | LDA x only | LDA x + c | BremsLib / Tsai |
|---|---|---|---|
| 1 H | 1.171 | 1.209 | 1.195 |
| 2 He | 1.103 | 1.115 | 1.114 |
| 3 Li | 1.103 | 1.115 | 1.110 |
| 6 C | 1.068 | 1.072 | 1.071 |
| 13 Al | 1.045 | 1.046 | 1.031 |

The prediction reproduces both the size and the Z dependence of the excess. H through C fall within the LDA x / x + c bracket. Al's Slater-rule density is the crudest one, and its electrostatic `L_rad` is already 1.8 % below Thomas–Fermi. So the low-Z BremsLib excess over SB86 is the library's screening model, an exchange–correlation term in the potential seen by the projectile. That term is not a PyRITE staging or interpolation error, and it is not an SB86 defect. Whether a local exchange–correlation potential belongs in the elastic screening of a fast projectile is a modelling choice of the library. Tsai and SB86 use electrostatic form factors. No source bounds the difference within ±13 %.

Limits of the probe: densities are analytic rather than self-consistent, the functional is bracketed rather than known, the calculation is non-relativistic, and only `y = 0` is covered. An earlier self-consistent LDA attempt was abandoned because of tail-precision artefacts. The probe identifies the mechanism and its size; it does not reproduce BremsLib to better than a few per cent.

### Gate

Li's pointwise `chi` stays reported, not gated (`UNGATED_KINDS[3]["chi"]`), now with this cause, and #1073's staging reason is retired. The band moment stays gated. The extension-moment reason stands. Its "different screening" remark (Hubbell form factors against a Kohn–Sham potential) is the same mechanism. H stays fully ungated (#1068). The re-cited reason is pinned by `test_gate_domain_ungates_h_and_two_cited_li_kinds`. The probe script and its output are recorded on [#1349](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1349#issuecomment-7052289582).

