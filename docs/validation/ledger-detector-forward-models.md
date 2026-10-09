# Detector forward models (downstream — lower risk)

Part of the [physics validation ledger](physics-validation-ledger.md). See the [validation methodology](methodology.md) for the status lifecycle and the [domain inventories](domain-inventories.md) for a claim-by-claim index.

## `positioned-filter-attenuation`

- **Claim:** primary photons reaching pixel centre `p` through finite plates have factor `T_p(E) = exp[-Σ_j μ_j(E)ℓ_pj]`, where each `ℓ_pj` is the exact source-to-pixel ray length inside plate `j`; pixel flux is `F_p(E) = I_q(p)(E) ΔΩ_p T_p(E)`
- **Code:** `materials/attenuation.py::linear_attenuation_inv_mm`; `_planar_geometry.py::planar_rays`; `instrument/geometry.py::ray_box_path_lengths`; `instrument/attenuation.py::primary_transmission`
- **Source:** Bouguer--Beer exponential attenuation; `μ_j` is the narrow-beam total of `narrow-beam-total-attenuation`, sourced from EPDL2025 since issue #274 (Henke et al. 1993 `f₂` convention / Chantler tabulation before it)
- **Status:** rederived
- **Checks:** implementation-side units, positive optical depth, zero-filter identity, uncovered-ray identity, normal-incidence thickness, compound additivity, plate-order invariance; fresh-context derivation and external numerical oracle pending
- **Anchor:** `tests/materials/test_attenuation.py`; `tests/instrument/test_attenuation.py`; `tests/instrument/test_geometry.py`
- **Notes:** **Issue #274:** `μ_j` now comes from EPDL2025 (photoelectric + coherent + incoherent + pair), defined to 100 GeV; the geometry claim and its re-derivation are unchanged, while the `f₂`-based numeric cross-checks quoted below refer to the retired Chantler + Elam coefficient. homogeneous passive primary attenuation only; no fluorescence, diffraction, or secondaries; centre rays do not integrate finite pixel area. **2026-09-13 fresh-context re-derivation matches** with no divergent factor, sign, exponent, unit, or convention; largest code-traceable divergence is `1.28e-08`, fully attributed to two truncated literals (`HC_EV_ANG`, `3.5e-09`; `R_E_ANG`, `9.3e-09`). Reference side used CODATA constants plus `xraydb.f2_chantler` and the independent Elam compilation — no PyRITE helper; geometry checked against an 8e6-sample arc-length quadrature of the box indicator over normal, oblique, tilted-plate, side-escape (`10.008 mm`, **not** `t/cos θ = 20.016`), miss, parallel-outside, pixel-interior, beyond-pixel, and corner-grazing cases. Sole ownership of `ΔΩ_p` was confirmed rather than assumed: `SpatialResult.spectra(measured=True)` calls `detector.score(..., scale=1.0)`, and `scale` is the only flux-normalisation hook in `detectors/spec.py`; the Eagle `Ω×QE` reading belongs to the legacy scalar path and never composes with `ray_map.solid_angle_sr`. Unpixelated `PlanarDetector` faces now use exact finite-rectangle acceptance, including oblique poses; pixel grids retain the centre-ray approximation, with convergence anchored by `tests/instrument/test_geometry.py`. **The quantified `f₂` bias recorded here is fixed** ([#104](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/104)): `μ_j` is now the narrow-beam total, photoabsorption plus coherent plus incoherent, owned by `narrow-beam-total-attenuation`. The 2026-09-13 re-derivation's arithmetic is unaffected — it was a scope gap in the cross-section, not an error in `T_p = exp[-Σⱼ μⱼ ℓ_pj]` — but the numbers it quoted for the missing fraction (`1.6%` Al 8 keV, `9.9%` Al 20 keV, `7.3%` C 8 keV, `50.8%` C 20 keV) are now inside `μ`, not outside it. What remains is the geometry condition the row never stated: the narrow-beam form still assumes scattered photons leave the collection solid angle, and no build-up factor is applied, so a plate close to the detector over-counts small-angle coherent scatter as removal. Plate additivity assumes disjoint plate interiors, which `instrument/model.py::validate_downstream_scene` does not enforce ([#105](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/105)). **2026-09-26 implementation extension, not covered by the re-derivation above** (#23): a grid node at exactly `E = 0` (the standard profile's default bremsstrahlung grid starts there) takes the limit `μ_j → +∞` as `E → 0⁺`, so `T_p = 0` wherever `ℓ_pj > 0` and `T_p = 1` is kept exactly for `ℓ_pj = 0`; finite `μ` keeps the unchanged closed form bit-for-bit. Needs a fresh-context check of that branch. Author-prepared verification packet plus the independent Part I–III derivation: [positioned filter attenuation](detectors/positioned-filter-attenuation.md). Human sign-off pending.

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
- **Source:** Henke `f₂` convention, Chantler/FFAST tabulation via xraydb (Si)
- **Status:** unverified
- **Checks:** native measured-bin batch application matches one-spectrum application; output bins are explicit and nonnegative; detected event mass cannot exceed incident event mass in the focused regression; source cells split by overlap onto the same fixed input channels in both scoring paths; an exactly represented line has partition-independent native mass, and Gaussian native counts on uniform and resonance-local axes agree within 1e-5
- **Anchor:** `tests/instrument/test_acquisition_core.py`; `tests/detectors/test_timepix_native_rebin.py`
- **Notes:** **hardware params are placeholders** — can't sign off until real quad values land. The native-bin checks cover operator bookkeeping only, not hardware accuracy. Issue #219 replaces native node assignment with the same piecewise-constant overlap integral as `apply`; interpolation of measured density back onto source nodes has its own quadrature/truncation error. Transport/checkpoint and observation identity payloads are unchanged: stored observations contain true-spatial factors and rescore native counts at read time, so rescoring updates expected counts and fixed-seed Poisson draws without rerunning transport. No native-count golden is stored. Fresh-context overlap derivation and independent numeric references match; the bookkeeping subclaim is rederived, while the full hardware claim remains unverified. [Resampling verification](detectors/detector-timepix.md).

## `pixel-acquisition-counting`

- **Claim:** accepted event mass per incident electron scales to expected counts with $N_e=t_{exp}f_{rep}Q_{bunch}10^{-12}/e$; reporting bins plus disjoint underflow, overflow, and below-cut channels conservatively partition native measured event mass
- **Code:** `instrument/acquisition.py::electron_count`; `instrument/acquisition.py::score_acquisition`
- **Source:** SI definitions of coulomb, picocoulomb, hertz, and the exact elementary charge; `scipy.constants.elementary_charge`
- **Status:** unverified
- **Checks:** units; analytic one-electron-per-bunch normalization; exact zero-charge limit; hand-computed fractional-overlap partition; component-additive totals; fixed-seed selection/chunk invariance
- **Anchor:** `tests/instrument/test_acquisition_core.py`; `tests/results/test_spatial_model.py`
- **Notes:** implementation-context review only. Uniform density within each native measured bin is the conservative-rebin assumption. Fresh-context validation remains pending; no hardware calibration claim is made.

## `pixel-angular-interpolation`

- **Claim:** under `PixelScorer(reconstruction="bilinear_tile")` pixel `p` takes intrinsic density `I_p(E) = Σ_k w_pk I_k(E)` over at most four angular tiles, with separable weights `w_pk = (1-t_y)(1-t_x), (1-t_y)t_x, t_y(1-t_x), t_y t_x`; `t_x = clip((x_p - X_j)/(X_{j+1} - X_j), 0, 1)` (and `t_y` likewise), where `x_p` is pixel `p`'s detector-local centre coordinate and knot `X_j` is the mean over tile rows of the local `x` of tile column `j`'s detector-plane intersections `t d_k`, `t = (c·n)/(d_k·n)`, of the tile representative directions `d_k`; pixel flux is then `F_p(E) = I_p(E) ΔΩ_p T_p(E)` as in `positioned-filter-attenuation`
- **Code:** `instrument/geometry.py::angular_tile_weights`; `results/model.py::SpatialResult._materialize`
- **Source:** tensor-product (bilinear) linear interpolation on a rectilinear grid, Press et al., *Numerical Recipes* 3rd ed. (2007) §3.6.1, Eq. 3.6.5; ray–plane intersection; tile representatives from `instrument/geometry.py::angular_tiles`
- **Status:** anchored
- **Checks:** implementation-side: weights non-negative and summing to 1 (oblique, rolled pose, uneven grids); knots strictly increasing; exact one-hot weight at a pixel on a tile centre; bit-identical to `nearest_tile` when `angular_shape` equals the pixel shape and for 1×1; one-tile axis constant; linear field in knot coordinates reproduced to 1e-13 inside the knot hull and clamped outside; store round-trip preserves the mode; default `nearest_tile` identity digests unchanged; **2026-10-05 fresh-context re-derivation matches** (verdict `rederived`): each projected representative is the $r^{-4}$-weighted convex combination of its own pixel centres, so knots are strictly increasing; weights match an independent loop/closed-form numpy reference to ≤2.2e-14 across 6 oblique/rolled/offset cases; write-up [pixel angular interpolation](detectors/pixel-angular-interpolation.md)
- **Anchor:** `tests/instrument/test_geometry.py`; `tests/results/test_spatial_model.py`; `tests/observations/test_store.py`
- **Notes:** issue #212. Assumes the intrinsic density varies smoothly, close to linearly, in projected direction between tile representatives. Blending mixes spectra at fixed energy: an angle-dependent line (coherent/Bragg) is broadened or doubled, not shifted, so accuracy is claimed only for smooth components (characteristic, bremsstrahlung). Clamping keeps edge pixels at the edge-knot value (no extrapolation), so a gradient beyond the outermost tile centres is flattened there. Knots are row/column means of projected representatives, so the representative points themselves need not lie on the knot grid; the reconstruction is exact only at knots, not at each representative. Convergence against a direct per-pixel evaluation is measured by `checks/pixel_reconstruction_oracle.py --geometry tpx-test` (decision evidence, not an anchor). **2026-10-05 local smoke** (5 electrons, seed 1; tile and pixel spectra share one transported population, so the error is angular rather than MC noise): at `angular_shape=[5, 5]` the interior-pixel characteristic-flux error falls from `+1.30%` (nearest) to `-0.03%` (bilinear), continuum `+0.23%` to `-0.01%`; edge and corner pixels lie outside the knot hull, where clamping equals nearest-tile (`≤1.8%` characteristic, `≤0.34%` continuum). Higher-statistics multi-seed evidence is pending a remote run. Knot snapping within `1e-9` of an interval absorbs plane-projection round-off up to a distance-to-pitch ratio of about `1e8`; `nearest_tile` bit-identity at full angular resolution also requires finite neighbour spectra (`0·∞` is NaN). Human sign-off pending (#277).

## `detector-line-broadening`

- **Claim:** EDS polar-aperture line broadening `FWHM = (2√(2ln2)/3)·(∂Ep/∂θobs)·Δθobs`
- **Code:** `montecarlo/detector.py::aperture_fwhm_eV`
- **Source:** Zhai et al. 2025 SI Eq. (14)
- **Status:** rederived
- **Checks:** units (eV); limiting case FWHM→0 as Δθobs→0 or β→0; prefactor pinned against an independent `2*sqrt(2*log(2))/3` computation; fresh-context re-derivation: Eq. (14) azimuthal reduction for v∥z confirmed term-for-term (rtol 1e-14 spot check), Zhai's 0.785 prefactor reproduced exactly by the span=3σ Gaussian-equivalence reading
- **Anchor:** `tests/detectors/test_response.py`
- **Notes:** **2026-07-11 fix**: prefactor was `2·sqrt(2ln2/3)` (√3× too large, misplaced parenthesis) prior to this fix; corrected to `2·sqrt(2ln2)/3` per Eq. (14); the 0.785 factor is Zhai's own convention (~15% wider than a uniform-span variance match, ≈0.68) — we follow the paper; write-up `detectors/detector-line-broadening.md`
