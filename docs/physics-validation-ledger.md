# Physics validation ledger

The single source of truth for **what physics `cxr-mc` claims and whether it has been verified.** One row per atomic physics claim. Goal: every load-bearing equation reaches `signed-off` before publication. See [`docs/validation/README.md`](validation/README.md) for the method, the status lifecycle, and the re-derivation workflow.

> [!note]
> **Seeded, not complete.** The rows below are the core physics, extracted from `docs/repo_map.md` + the in-code citations. Remaining formulas (detector internals, geometry helpers, edge corrections) still need ledgering — grep the physics modules for un-annotated `def`s. Anchor on `file::symbol`, never a line number.

**Status:** `unverified` → `filtered` (units+limits+signs) → `rederived` (independent derivation matches) → `anchored` (regression test green) → `signed-off` (human-certified). `discrepancy` = a check failed.

Progress: **0 / 22 signed-off** · 3 rederived · 1 filtered · 1 blocked.

## Core coherent physics (highest risk — verify first)

| id | claim | code | source | status | checks | anchor | notes |
|----|-------|------|--------|--------|--------|--------|-------|
| `coherent-line-spectrum` | `\|A_PXR + A_CBS\|²` segment-sum line spectrum, exact mosaic average | `montecarlo/spectrum.py::mc_spectrum` | Feranchuk–Spence 2000 Eq.(10),(12); Zhai 2025 | unverified | — | `checks/anchor_figures.py::single_segment_anchor` | interference is non-separable; highest priority |
| `finite-time-lineshape` | `\|Q\|² = t_L²·sinc²(P·t_L)` (replaces absorption-limited δ) | `montecarlo/spectrum.py::mc_spectrum` | Feranchuk 2000 (finite interaction length) | unverified | — | _t→∞ → δ limit test (to add)_ | |
| `pxr-amplitude` | `χ_g` PXR susceptibility amplitude | `crystallography.py::chi_g` | Feranchuk 2000 | unverified | — | — | |
| `cbs-amplitude` | `U_g` CBS potential amplitude + relativistic 1/γ braced terms | `crystallography.py::U_g` (+ amplitude assembly in `montecarlo/spectrum.py`) | Feranchuk 2000 | unverified | — | — | 1/γ matters ≳100 keV |
| `line-energy-dispersion` | `ω = v·g / (1 − v·n̂)` tunable line energy | `montecarlo/geometry.py::tilted_geometry` / `checks/anchor_figures.py::line_energy_eV` | Zhai 2025 Eq.(10) | unverified | — | `checks/anchor_figures.py::theory_line_energies` | |
| `closed-form-flux` | Eq.(12) closed-form line flux (single-segment reference) | `checks/anchor_figures.py::feranchuk_line_flux` | Feranchuk 2000 Eq.(12) | unverified | — | `checks/anchor_figures.py::single_segment_anchor` (ratio≈1) | reference, not pipeline |
| `enhancement-bulk-film` | bulk-vs-film line enhancement | `checks/anchor_figures.py::figure_enhancement` | Zhai 2025 | unverified | — | `checks/anchor_figures.py::figure_enhancement` | |

## Crystallography & atomic data

| id | claim | code | source | status | checks | anchor | notes |
|----|-------|------|--------|--------|--------|--------|-------|
| `structure-factor` | structure factor `F(g)` + Debye–Waller | `crystallography.py::structure_factor`, `::debye_waller` | standard crystallography | unverified | — | — | |
| `atomic-form-factor` | `F(g,E) = f0(g) + f'(E) + i·f''(E)` | `atomic_form_factors.py::atomic_form_factor` | Waasmaier–Kirfel f0 + Chantler/FFAST (xraydb) | filtered | provenance re-validated | — | see `docs/atomic-data-sources.md` |
| `mote2-bulk-structure` | 2H-MoTe2 bulk lattice + basis (a=3.517 Å, c=13.96 Å) | `data/crystal_structures.toml::mote2` | literature / Materials Project | unverified | cell volume + stoichiometry check | `tests/test_crystallography.py::test_mote2_structure_sane` | bulk material, used in bare MoTe2 scans |
| `mote2-product-structure` | 2H-MoTe2 product-page lattice + basis used for few-layer MoTe2-on-sapphire scans | `data/crystal_structures.toml::mote2_product` | 2D Semiconductors product page | unverified | unit conversion nm→Å + cell-volume check | `tests/test_crystallography.py::test_mote2_product_structure_sane` | product-page lattice (a=3.50 Å, c=13.41 Å) differs from bulk for thin films |
| `sapphire-corundum-structure` | α-Al2O3/sapphire corundum lattice + explicit conventional-cell basis, B_ang2=0.25 | `data/crystal_structures.toml::sapphire` | Newnham & de Haan 1962; B_ang2 literature ~0.25 | unverified | cell volume + stoichiometry check | `tests/test_crystallography.py::test_sapphire_structure_sane` | expanded from R-3c Wyckoff sites because the loader does not apply symmetry; Debye-Waller B_ang2 updated from 0.5 Ų to 0.25 Ų (literature range 0.20–0.30) |
| `hbn-structure` | h-BN P6_3/mmc layered/eclipsed lattice + explicit four-atom conventional-cell basis | `data/crystal_structures.toml::hbn` | Pease, Acta Cryst 5, 356 (1952) | rederived | V=36.17 Å³ ✓; 2B+2N ✓; basis ≡ Pease Wyckoff under shift (2/3,1/3,3/4) ✓; AA′ registry ✓ (anchor green) | `tests/test_crystallography.py::test_hbn_structure_sane` | a=2.504 A, c=6.661 A; B/N sites swap across the half-cell so B lies above N; write-up `docs/validation/hbn-structure.md` |
| `absorption-length` | X-ray absorption length / μ | `crystallography.py::absorption_length_ang` | Henke f2 / Beer–Lambert | unverified | — | — | |
| `grazing-optical-constants` | complex refractive index `n = 1 − δ − iβ` (δ, β from f1=Z+f′, f2) | `crystallography.py::optical_constants` | Als-Nielsen & McMorrow *Elements of Modern X-ray Physics* 2nd ed. Ch. 3; equiv. Attwood & Sakdinawat Ch. 3 | rederived | units (δ,β dimensionless); sign δ,β>0 off-edge; independent θ_c=√(2δ) spot-check at Cu Kα matches textbook Si (0.223° vs ≈0.22°) and Au (0.554° vs ≈0.55°) to 4 sig figs | `tests/test_crystallography.py::test_optical_constants_beta_matches_absorption_length`, `::test_optical_constants_delta_positive_off_edge` | feeds `grating.py::Grating.reflectivity`; write-up `docs/validation/grazing-reflectivity.md`; note the β↔`absorption_length_ang` test is a construction tautology (same formula reused), not independent corroboration — see write-up |
| `self-absorption` | per-segment Beer–Lambert path-to-surface, cross-stack | `montecarlo/spectrum.py::mc_spectrum` | Beer–Lambert | unverified | — | — | reduces across multilayer |

## Transport & background

| id | claim | code | source | status | checks | anchor | notes |
|----|-------|------|--------|--------|--------|--------|-------|
| `electron-transport` | Joy–Luo slowing-down + Mott/screened-Rutherford elastic scattering → radiating segments | `montecarlo/transport.py::simulate_trajectories` | Joy–Luo; NIST SRD 64 Mott; Browning free paths | unverified | — | — | CASINO-style single-scattering MC; upstream of all spectra |
| `brem-spectrum` | bremsstrahlung background, Born + Elwert | `montecarlo/spectrum.py::mc_brem_spectrum` | Born + Elwert | unverified | — | — | benign 0-eV divide-by-zero clamped |

## Mosaicity & multilayer (code-cross-checked; need sign-off + measured data)

| id | claim | code | source | status | checks | anchor | notes |
|----|-------|------|--------|--------|--------|--------|-------|
| `mosaic-analytic` | analytic broadening `FWHM = E·\|tan ψ\|·η` | `montecarlo/detector.py::mosaic_fwhm_eV` | `docs/crystal-mosaicity.md` | unverified | — | `checks/mosaic_mc_check.py` | energy-shift only; `tan ψ` capped near grazing |
| `mosaic-mc` | exact per-orientation incoherent average (2-D Gauss–Hermite) | `montecarlo/spectrum.py::mc_spectrum` (`mosaic_route="mc"`) | `docs/crystal-mosaicity.md` | unverified | η→0 bit-for-bit; small-η→analytic (in check) | `checks/mosaic_mc_check.py` | broadens PXR+CBS; no grazing divergence |
| `multilayer-stack` | film-on-substrate transport + absorption | `montecarlo/transport.py::simulate_trajectories` (`layers=`) | `docs/multilayer-materials.md` | unverified | — | `checks/multilayer_validation_check.py` | substrate-dominance prediction lives here |

## Detector forward models (downstream — lower risk)

| id | claim | code | source | status | checks | anchor | notes |
|----|-------|------|--------|--------|--------|--------|-------|
| `detector-eaglexo` | `solid_angle(Ω) × QE(E)` CCD operator | `eaglexo_response.py::EagleResponse` | `eaglexo_qe.csv` | unverified | — | — | |
| `detector-timepix` | Si charge model, diffusion, ~1.9 keV counting threshold | `timepix_response.py::TimepixResponse` | Henke f2 (Si) | blocked | — | — | **hardware params are placeholders** — can't sign off until real quad values land |

## Grazing-incidence grating spectrometer

| id | claim | code | source | status | checks | anchor | notes |
|----|-------|------|--------|--------|--------|--------|-------|
| `grazing-reflectivity` | small-angle Fresnel reflectivity `R(θ) = \|r(θ)\|²`, `r(θ) = (θ − √(θ²−2δ−2iβ)) / (θ + √(θ²−2δ−2iβ))`, polarization-independent (grazing-incidence approximation) | `grating.py::Grating.reflectivity` | Als-Nielsen & McMorrow *Elements of Modern X-ray Physics* 2nd ed. Ch. 3; equiv. Attwood & Sakdinawat Ch. 3 | rederived | bounded in [0,1]; θ≪θc → R→1 (total external reflection); θ≫θc → R→(θc/2θ)⁴·(1+(β/δ)²) independently re-derived (Taylor expansion), test's 60° choice confirmed necessary (Au β/δ≈0.25 at 1500 eV, needs θ≳20° for <10% convergence to the β→0 idealization); monotonic decrease with grazing angle; independent Maxwell/Snell re-derivation matches code term-for-term | `tests/test_grating.py::test_reflectivity_total_external_reflection_below_critical_angle`, `::test_reflectivity_asymptotic_falloff_above_critical_angle`, `::test_reflectivity_bounded_and_decreases_away_from_critical_angle` | coating element/density confirmed gold for McPherson 251MX, density/molar-mass values not independently re-verified here (`Grating.coating`); depends on `grazing-optical-constants`; write-up `docs/validation/grazing-reflectivity.md` — the 60°-grazing test checks the formula's own asymptotic self-consistency, not physical accuracy outside θ≪1 (real operation stays at few-degree grazing) |
