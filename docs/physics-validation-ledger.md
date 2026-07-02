# Physics validation ledger

The single source of truth for **what physics `cxr-mc` claims and whether it has been verified.** One row per atomic physics claim. Goal: every load-bearing equation reaches `signed-off` before publication. See [`docs/validation/README.md`](validation/README.md) for the method, the status lifecycle, and the re-derivation workflow.

> [!note]
> **Seeded, not complete.** The rows below are the core physics, extracted from `docs/repo_map.md` + the in-code citations. Remaining formulas (detector internals, geometry helpers, edge corrections) still need ledgering — grep the physics modules for un-annotated `def`s. Anchor on `file::symbol`, never a line number.

**Status:** `unverified` → `filtered` (units+limits+signs) → `rederived` (independent derivation matches) → `anchored` (regression test green) → `signed-off` (human-certified). `discrepancy` = a check failed.

Progress: **0 / 20 signed-off** · 1 filtered · 1 blocked.

## Core coherent physics (highest risk — verify first)

| id | claim | code | source | status | checks | anchor | notes |
|----|-------|------|--------|--------|--------|--------|-------|
| `coherent-line-spectrum` | `\|A_PXR + A_CBS\|²` segment-sum line spectrum, exact mosaic average | `montecarlo.py::mc_spectrum` | Feranchuk–Spence 2000 Eq.(10),(12); Zhai 2025 | unverified | — | `checks/anchor_figures.py::single_segment_anchor` | interference is non-separable; highest priority |
| `finite-time-lineshape` | `\|Q\|² = t_L²·sinc²(P·t_L)` (replaces absorption-limited δ) | `montecarlo.py::mc_spectrum` | Feranchuk 2000 (finite interaction length) | unverified | — | _t→∞ → δ limit test (to add)_ | |
| `pxr-amplitude` | `χ_g` PXR susceptibility amplitude | `crystallography.py::chi_g` | Feranchuk 2000 | unverified | — | — | |
| `cbs-amplitude` | `U_g` CBS potential amplitude + relativistic 1/γ braced terms | `crystallography.py::U_g` (+ amplitude assembly in `montecarlo.py`) | Feranchuk 2000 | unverified | — | — | 1/γ matters ≳100 keV |
| `line-energy-dispersion` | `ω = v·g / (1 − v·n̂)` tunable line energy | `montecarlo.py::tilted_geometry` / `checks/anchor_figures.py::line_energy_eV` | Zhai 2025 Eq.(10) | unverified | — | `checks/anchor_figures.py::theory_line_energies` | |
| `closed-form-flux` | Eq.(12) closed-form line flux (single-segment reference) | `checks/anchor_figures.py::feranchuk_line_flux` | Feranchuk 2000 Eq.(12) | unverified | — | `checks/anchor_figures.py::single_segment_anchor` (ratio≈1) | reference, not pipeline |
| `enhancement-bulk-film` | bulk-vs-film line enhancement | `checks/anchor_figures.py::figure_enhancement` | Zhai 2025 | unverified | — | `checks/anchor_figures.py::figure_enhancement` | |

## Crystallography & atomic data

| id | claim | code | source | status | checks | anchor | notes |
|----|-------|------|--------|--------|--------|--------|-------|
| `structure-factor` | structure factor `F(g)` + Debye–Waller | `crystallography.py::structure_factor`, `::debye_waller` | standard crystallography | unverified | — | — | |
| `atomic-form-factor` | `F(g,E) = f0(g) + f'(E) + i·f''(E)` | `atomic_form_factors.py::atomic_form_factor` | Waasmaier–Kirfel f0 + Chantler/FFAST (xraydb) | filtered | provenance re-validated | — | see `docs/atomic-data-sources.md` |
| `mote2-product-structure` | 2H-MoTe2 product lattice + basis used for few-layer MoTe2-on-sapphire scans | `data/crystal_structures.toml::mote2` | 2D Semiconductors product page | unverified | unit conversion nm→Å + cell-volume check | `tests/test_crystallography.py::test_mote2_structure_sane` | product-page lattice constants replace older bulk literature defaults for this material key |
| `sapphire-corundum-structure` | α-Al2O3/sapphire corundum lattice + explicit conventional-cell basis | `data/crystal_structures.toml::sapphire` | Newnham & de Haan 1962 | unverified | cell volume + stoichiometry check | `tests/test_crystallography.py::test_sapphire_structure_sane` | expanded from R-3c Wyckoff sites because the loader does not apply symmetry |
| `absorption-length` | X-ray absorption length / μ | `crystallography.py::absorption_length_ang` | Henke f2 / Beer–Lambert | unverified | — | — | |
| `self-absorption` | per-segment Beer–Lambert path-to-surface, cross-stack | `montecarlo.py::mc_spectrum` | Beer–Lambert | unverified | — | — | reduces across multilayer |

## Transport & background

| id | claim | code | source | status | checks | anchor | notes |
|----|-------|------|--------|--------|--------|--------|-------|
| `electron-transport` | Joy–Luo slowing-down + Mott/screened-Rutherford elastic scattering → radiating segments | `montecarlo.py::simulate_trajectories` | Joy–Luo; NIST SRD 64 Mott; Browning free paths | unverified | — | — | CASINO-style single-scattering MC; upstream of all spectra |
| `brem-spectrum` | bremsstrahlung background, Born + Elwert | `montecarlo.py::mc_brem_spectrum` | Born + Elwert | unverified | — | — | benign 0-eV divide-by-zero clamped |

## Mosaicity & multilayer (code-cross-checked; need sign-off + measured data)

| id | claim | code | source | status | checks | anchor | notes |
|----|-------|------|--------|--------|--------|--------|-------|
| `mosaic-analytic` | analytic broadening `FWHM = E·\|tan ψ\|·η` | `montecarlo.py::mosaic_fwhm_eV` | `docs/crystal-mosaicity.md` | unverified | — | `checks/mosaic_mc_check.py` | energy-shift only; `tan ψ` capped near grazing |
| `mosaic-mc` | exact per-orientation incoherent average (2-D Gauss–Hermite) | `montecarlo.py::mc_spectrum` (`mosaic_route="mc"`) | `docs/crystal-mosaicity.md` | unverified | η→0 bit-for-bit; small-η→analytic (in check) | `checks/mosaic_mc_check.py` | broadens PXR+CBS; no grazing divergence |
| `multilayer-stack` | film-on-substrate transport + absorption | `montecarlo.py::simulate_trajectories` (`layers=`) | `docs/multilayer-materials.md` | unverified | — | `checks/multilayer_validation_check.py` | substrate-dominance prediction lives here |

## Detector forward models (downstream — lower risk)

| id | claim | code | source | status | checks | anchor | notes |
|----|-------|------|--------|--------|--------|--------|-------|
| `detector-eaglexo` | `solid_angle(Ω) × QE(E)` CCD operator | `eaglexo_response.py::EagleResponse` | `eaglexo_qe.csv` | unverified | — | — | |
| `detector-timepix` | Si charge model, diffusion, ~1.9 keV counting threshold | `timepix_response.py::TimepixResponse` | Henke f2 (Si) | blocked | — | — | **hardware params are placeholders** — can't sign off until real quad values land |
