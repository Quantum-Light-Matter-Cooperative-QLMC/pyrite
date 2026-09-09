# Detector forward models (downstream — lower risk)

Part of the [physics validation ledger](physics-validation-ledger.md). See the [validation methodology](methodology.md) for the status lifecycle and the [domain inventories](domain-inventories.md) for a claim-by-claim index.

## `positioned-filter-attenuation`

- **Claim:** primary photons reaching pixel centre `p` through finite plates have factor `T_p(E) = exp[-Σ_j μ_j(E)ℓ_pj]`, where each `ℓ_pj` is the exact source-to-pixel ray length inside plate `j`; pixel flux is `F_p(E) = I_q(p)(E) ΔΩ_p T_p(E)`
- **Code:** `materials/attenuation.py::linear_attenuation_inv_mm`; `instrument/geometry.py::ray_box_path_lengths`; `instrument/attenuation.py::primary_transmission`
- **Source:** Bouguer--Beer exponential attenuation; Henke et al. 1993, *Atomic Data and Nuclear Data Tables* 54, 181--342, DOI 10.1006/adnd.1993.1013 (elemental `f₂`, through `absorption-length`)
- **Status:** unverified
- **Checks:** implementation-side units, positive optical depth, zero-filter identity, uncovered-ray identity, normal-incidence thickness, compound additivity, plate-order invariance; fresh-context derivation and external numerical oracle pending
- **Anchor:** `tests/materials/test_attenuation.py`; `tests/instrument/test_attenuation.py`; `tests/instrument/test_geometry.py`
- **Notes:** homogeneous passive primary attenuation only; no scatter, fluorescence, diffraction, or secondaries; centre rays do not integrate finite pixel area. Author-prepared verification packet: [positioned filter attenuation](detectors/positioned-filter-attenuation.md). Independent physics review must update this row before status advances.

## `detector-eaglexo`

- **Claim:** `solid_angle(Ω) × QE(E)` CCD operator
- **Code:** `detectors/eaglexo_response.py::EagleResponse`
- **Source:** `eaglexo_qe.csv`
- **Status:** filtered
- **Checks:** units, zero/far-field limits, exact rectangular solid angle, QE bounds/tail continuity, and single ownership of `Ω` checked
- **Anchor:** —
- **Notes:** operator matches; digitized manufacturer QE ordinates still require source-datasheet comparison; [validation write-up](detectors/detector-eaglexo.md)

## `detector-timepix`

- **Claim:** Si charge model, diffusion, ~1.9 keV counting threshold
- **Code:** `detectors/timepix_response.py::TimepixResponse`
- **Source:** Henke f2 (Si)
- **Status:** blocked
- **Checks:** —
- **Anchor:** —
- **Notes:** **hardware params are placeholders** — can't sign off until real quad values land

## `detector-line-broadening`

- **Claim:** EDS polar-aperture line broadening `FWHM = (2√(2ln2)/3)·(∂Ep/∂θobs)·Δθobs`
- **Code:** `montecarlo/detector.py::aperture_fwhm_eV`
- **Source:** Zhai et al. 2025 SI Eq. (14)
- **Status:** rederived
- **Checks:** units (eV); limiting case FWHM→0 as Δθobs→0 or β→0; prefactor pinned against an independent `2*sqrt(2*log(2))/3` computation; fresh-context re-derivation: Eq. (14) azimuthal reduction for v∥z confirmed term-for-term (rtol 1e-14 spot check), Zhai's 0.785 prefactor reproduced exactly by the span=3σ Gaussian-equivalence reading
- **Anchor:** `tests/detectors/test_response.py`
- **Notes:** **2026-07-11 fix**: prefactor was `2·sqrt(2ln2/3)` (√3× too large, misplaced parenthesis) prior to this fix; corrected to `2·sqrt(2ln2)/3` per Eq. (14); the 0.785 factor is Zhai's own convention (~15% wider than a uniform-span variance match, ≈0.68) — we follow the paper; write-up `detectors/detector-line-broadening.md`
