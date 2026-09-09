# Crystallography & atomic data

Part of the [physics validation ledger](physics-validation-ledger.md). See the [validation methodology](methodology.md) for the status lifecycle and the [domain inventories](domain-inventories.md) for a claim-by-claim index.

## `structure-factor`

- **Claim:** structure factor `F(g)` + Debye–Waller
- **Code:** `materials/crystal.py::structure_factor`, `::debye_waller`
- **Source:** standard crystallography
- **Status:** anchored
- **Checks:** units, forward/`g→0`/`B→0` limits, phase sign, DW exponent `16π²` normalization; diamond extinct/allowed + pinned DW value anchored green
- **Anchor:** `tests/materials/test_crystallography.py::test_diamond_structure_factor_extinct`, `::test_diamond_structure_factor_allowed`, `::test_debye_waller_pinned_value`
- **Notes:** independent derivation matches (three concurring fresh-context passes); phase `exp(2π i hkl·R)`, single amplitude DW `exp(−B g²/16π²)` with `B=8π²⟨u_g²⟩` under `g=2π/d` (factor-4 trap ruled out); diamond `(222)/(200)` extinct, `(111)/(220)/(311)/(400)` allowed pinned by regression test; in-code `Validation:` markers now present on both symbols. Remaining caveat: single scalar `B` per crystal (isotropic approx); [write-up](atomic-physics/structure-factor.md)

## `debye-waller-catalog-provenance`

- **Claim:** every production `B_ang2` is tied to a phase- and temperature-specific primary refinement, with scalar approximation scope stated
- **Code:** `data/materials.toml::crystals.*.B_ang2`; `materials/crystal.py::{structure_factor,U_g}`
- **Source:** per-material primary refinements collected in [audit](materials/debye-waller-audit.md)
- **Status:** discrepancy
- **Checks:** baseline inventory; `B=8π²Uiso`; tensor limit `exp[-gᵀU_jg/2]`; phase, temperature, site, convention, and reflection-scope checks
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden` (serialization only)
- **Notes:** 2026-07-25 baseline: 48 entries; h-BN, GeP, GeS, GeSe, PdTe2, and 2H-TaS2 now have primary-refinement temperature/conversion notes, sapphire has only an approximate range, four distinct legacy values lack value-level provenance, and 37 chemically diverse entries reuse `0.6 Å²`. Current crystal-wide scalar is insufficient when sites differ or nonparallel reflections probe anisotropic ADPs. No schema extension until source coverage establishes required site/tensor representation.

## `surface-hkl-orientation`

- **Claim:** reciprocal cleavage-plane normal `g_hkl = h b1 + k b2 + l b3` is mapped to sample `+z` by a proper minimal rotation, followed by the configured right-handed azimuth about `+z`
- **Code:** `montecarlo/geometry.py::_orientation_R`; plumbing through `sweep.py`, `montecarlo/spectrum/lines.py`, `montecarlo/detector.py`, and `montecarlo/runner/__init__.py`
- **Source:** standard reciprocal-lattice geometry and Rodrigues rotation
- **Status:** unverified
- **Checks:** reciprocal/direct equivalence for orthogonal one-axis cuts; nonorthogonal reciprocal-normal alignment; determinant/azimuth handedness; mutually exclusive parser/API inputs; legacy direct-axis matrix frozen bit-for-bit
- **Anchor:** `tests/montecarlo/test_surface_orientation.py`; `tests/materials/test_material_catalog.py::test_crystal_requires_exactly_one_orientation`, `::test_crystal_accepts_reciprocal_surface_orientation`
- **Notes:** Implementation-side record only. Exact reciprocal orientation removes the direct-axis approximation for nonorthogonal cleavage planes; an explicit Sweep/Layer `beam_uvw` override clears the catalog surface orientation.

## `crystals-cif-adapter`

- **Claim:** `crystals.Crystal`/CIF lattice + symmetry-expanded fractional basis conversion into the internal `CRYSTALS` entry shape
- **Code:** `materials/_cif.py::crystals_crystal_to_crystal_info`, `::load_crystal_from_cif`
- **Source:** crystals 1.7.0 API / CIF parser
- **Status:** anchored
- **Checks:** lattice lengths/angles + basis + volume round trip; non-P1 symmetry expansion; partial-occupancy rejection
- **Anchor:** `tests/materials/test_crystallography.py::test_load_crystal_from_cif_returns_compatible_deterministic_info`, `::test_load_crystal_from_cif_expands_non_p1_symmetry`, `::test_load_crystal_from_cif_rejects_partial_occupancy`
- **Notes:** structural-data importer only; X-ray form factors, structure factors, reflection selection, attenuation, and transport remain in PyRITE

## `cod-lattice-catalog-geometry`

- **Claim:** six lattice parameters of every COD-pinned catalog crystal agree with the pinned external COD record
- **Code:** `data/materials.toml::crystals.*.cod_id`; `data/cifs/*.cif`
- **Source:** Crystallography Open Database records pinned by `cod_id`
- **Status:** anchored
- **Checks:** all 35 COD ids re-fetch; local and external `a,b,c` agree within `0.01 Å`, and `α,β,γ` within `0.1°`; committed cache coverage is exact
- **Anchor:** `tests/materials/test_crystal_external_db.py::test_cached_reference_covers_every_cod_entry`, `::test_local_lattice_matches_cached_cod`, `::test_local_lattice_matches_live_external`
- **Notes:** Geometry-only claim covering 35/48 catalog crystals and 34/47 crystals used by active simulation presets (LiF is catalog-only). It does not validate fractional basis, occupancy, surface orientation, Debye–Waller factors, couplings, transport, or spectra. The four MP-only and nine unpinned crystals remain outside this anchor; GeSe2's MP geometry comparison explicitly fails and remains deferred.

## `atomic-form-factor`

- **Claim:** `F(g,E) = f0(g) + f'(E) + i·f''(E)`
- **Code:** `materials/atomic.py::atomic_form_factor`
- **Source:** Waasmaier–Kirfel f0 + Chantler/FFAST (xraydb)
- **Status:** rederived
- **Checks:** units, `q=g/(4π)`, `+i f''` convention, forward limit, and three direct xraydb points checked exactly
- **Anchor:** —
- **Notes:** provenance in `docs/physics/atomic-physics/atomic-data-sources.md`; independent derivation matches; [validation write-up](atomic-physics/atomic-form-factor.md)

## `dans-diffraction-oracle`

- **Claim:** optional independent `Dans_Diffraction` lattice, reciprocal-geometry, and `\|F_hkl\|²` comparison harness
- **Code:** `validation_oracles.py::validate_dans_crystal`, `::compare_lattice`, `::compare_reflection_geometry`, `::compare_structure_factor_magnitudes`
- **Source:** `Dans_Diffraction` 3.4.0 generated API/docs; Waasmaier–Kirfel (1995); independent Henke/CXRO dispersion tables; local pyrite conventions
- **Status:** unverified
- **Checks:** fail-closed threshold evaluator; fake-oracle unit tests; pinned real-backend test at 1, 2, 3, and 8 keV
- **Anchor:** `tests/materials/test_validation_oracles.py`; `uv run --group oracle python checks/dans_diffraction_oracle.py`
- **Notes:** 2026-07-25 implementation run passed: max relative `Δ\|g\|=2.22e-16`; non-resonant Waasmaier–Kirfel `Δ\|F\|²=2.65e-15`; Chantler-vs-Henke dispersive max `Δ\|F\|²=6.85%` under the cross-database 10% bound. Validation-only backend; no production import. Status remains unverified pending fresh-context review; run is not human sign-off.

## `absorption-length`

- **Claim:** X-ray absorption length / μ
- **Code:** `materials/crystal.py::absorption_length_ang`
- **Source:** Henke f2 / Beer–Lambert
- **Status:** anchored
- **Checks:** units (`μ` in Å⁻¹, length in Å); passive-medium sign; field-to-intensity factor two; zero-density/zero-`f₂` and mixture limits
- **Anchor:** `tests/materials/test_crystallography.py::test_absorption_length_matches_henke_f2_coefficient`
- **Notes:** independent derivation gives `μ=2rₑλnf₂` and matches every `2π` factor in production; [validation write-up](radiation-physics/absorption-length.md)

## `grazing-optical-constants`

- **Claim:** complex refractive index `n = 1 − δ − iβ` (δ, β from f1=Z+f′, f2)
- **Code:** `materials/crystal.py::optical_constants`
- **Source:** Als-Nielsen & McMorrow *Elements of Modern X-ray Physics* 2nd ed. Ch. 3; equiv. Attwood & Sakdinawat Ch. 3
- **Status:** rederived
- **Checks:** units (δ,β dimensionless); sign δ,β>0 off-edge; independent θ_c=√(2δ) spot-check at Cu Kα matches textbook Si (0.223° vs ≈0.22°) and Au (0.554° vs ≈0.55°) to 4 sig figs
- **Anchor:** `tests/materials/test_crystallography.py::test_optical_constants_beta_matches_absorption_length`, `::test_optical_constants_delta_positive_off_edge`
- **Notes:** feeds `detectors/grating.py::Grating.reflectivity`; write-up `detectors/grazing-reflectivity.md`; note the β↔`absorption_length_ang` test is a construction tautology (same formula reused), not independent corroboration — see write-up

## `xray-chi-zero`

- **Claim:** g=0 unit-cell susceptibility `χ₀ = −rₑλ²/(πV_cell) · Σᵢ(f1ᵢ + i f2ᵢ)`, `f1 = Z + f′`, `f2 = f″`
- **Code:** `materials/crystal.py::chi_0`
- **Source:** `chi_g` (Feranchuk–Spence 2000 Eq. (3)) evaluated at g=0, where Debye–Waller and every basis phase reduce to 1; forward factors in the Henke convention as in `grazing-optical-constants`
- **Status:** rederived
- **Checks:** units (dimensionless); sign (Re χ₀ < 0, Im χ₀ < 0 for a passive medium); limiting cases — `use_henke=False` collapses to the real Thomson `−rₑλ²Z_cell/(πV_cell)` in closed form, and `\|χ₀\| ∝ λ²` verified over a decade in E; exact cross-check that `−Re χ₀/2` and `−Im χ₀/2` reproduce `optical_constants` δ, β to 1e-12 rel (shared normalization, not an independent derivation)
- **Anchor:** `tests/materials/test_crystallography.py::test_chi_0_linearization_is_optical_constants_exactly`, `::test_chi_0_without_henke_is_real_thomson_limit`, `::test_chi_0_vanishes_and_index_tends_to_vacuum_at_high_energy`
- **Notes:** Deliberately uses `Z_TABLE + f′` rather than `cromer_mann_f0(el, 0)` so it shares one normalization with `absorption-length`/`grazing-optical-constants` exactly instead of to within the Cromer-Mann `f0(0) ≈ Z` fit residual. Bulk response only — no interface/Fresnel term. Consumed by the spectrum kernels through `xray-refractive-index`. Independent re-derivation: [write-up](radiation-physics/xray-chi-zero.md)

## `xray-refractive-index`

- **Claim:** complex crystal refractive index `n(E) = √(1 + χ₀(E)) ≈ 1 − δ − iβ`
- **Code:** `materials/crystal.py::refractive_index`
- **Source:** Maxwell dispersion relation in a homogeneous dielectric, `k² = (1 + χ₀)ω²`
- **Status:** rederived
- **Checks:** units (dimensionless); convention — `n = 1 − δ − iβ` with time factor `exp(+iωt)`, matching both `grazing-optical-constants` and the coherent propagation phase `exp{i[ωt − k·r]}` in `lines.py`; sign (δ > 0, β > 0, δ ≫ β off-edge); limiting cases — `χ₀ → 0` gives `n → 1` (checked to 1e-5 at 20 keV), and the exact √ agrees with the linearized `1 − δ − iβ` to O(χ₀²) (measured residuals δ/2 and δ, i.e. 9.2e-5 and 1.8e-4 rel at 1.5 keV in Si)
- **Anchor:** `tests/materials/test_crystallography.py::test_refractive_index_matches_linearization_to_second_order`, `::test_refractive_index_delta_beta_positive_off_edge`
- **Notes:** Square root taken exactly rather than linearized because the in-medium wavevector is defined from it. δ ~ 1e-5..1e-3 per Å is negligible pointwise but is expected to accumulate to order-unity phase over micron trajectories — that accumulation is now quantified under `xray-in-medium-propagation-phase` (≈π per µm of depth separation). Both production consumers — the in-medium wavevector and the coherent propagation phase — take the exact √, never the linearized form. Builds on `xray-chi-zero`. Independent re-derivation: [write-up](radiation-physics/xray-refractive-index.md)

## `xray-in-medium-resonance`

- **Claim:** CXR line kinematics on the in-medium photon dispersion `k = Re n(ω) ω n̂`: resonance `ω_res = v·g / (1 − Re n (v·n̂))`, `k·v = ω(1 − denom)`, `k·g = Re n ω (n̂·g)`, PXR detuning `\|k+g\|² − k² = g² + 2k·g` and PXR numerator `k² = (Re n ω)²`
- **Code:** `montecarlo/spectrum/lines.py::_in_medium_kinematics`, `::mc_spectrum` (unconditional — there is no vacuum-dispersion switch); CUDA port of the same fixed point in `montecarlo/spectrum/coherent_stream_jit_kernel.py::_coherent_prologue_kernel`
- **Source:** energy–momentum conservation `ω = v·(k+g)` closed with the Maxwell dispersion relation in a homogeneous dielectric, `k² = (1 + χ₀)ω²` (`xray-refractive-index`); Feranchuk–Spence 2000 Eq. (10)/(13) with `k² → εω²` rather than an ad hoc `n` inserted into the vacuum result
- **Status:** rederived
- **Checks:** limiting case — the production switch that forced `k = ω` is retired, so the vacuum limit is now checked where it is physical (`χ₀ → 0` at high energy, under `xray-chi-zero`) and, at kernel level, by feeding `Re n = 1` to the CUDA prologue; absolute root — the measured peak lands on the closed-form in-medium fixed point to inside one grid step (2.5e-5 eV), which pins the root itself rather than an increment between two code paths; sign — with the detector upstream (`v·n̂ < 0`) the in-medium denominator exceeds the vacuum one and the line moves UP in energy; magnitude — the measured fractional line shift matches the closed form `−δ (v·n̂)/(1 − v·n̂)` to 4e-4 rel (hopg 002, 100 keV, θ_obs = 119°: +6.382e-2 eV on a 1600.32 eV line, δ = 1.900e-4), identically on BOTH the batched and the per-hkl accumulation paths; fixed-point solve of the implicit resonance contracts at rate ~δ ~ 1e-5 per pass, 3 passes taken — but **only in the X-ray regime**, so convergence is now VERIFIED per sample rather than assumed: the last pass must move `denom` by less than `_RESONANCE_ROOT_RTOL = 1e-3` (a genuine contraction moves it by ~δ³ ~ 1e-15 in float64, floored by float32 rounding ~1e-7 on the device twin — five orders of margin either side), and failures carry NaN out of `denom` onto the caller's existing finite mask
- **Anchor:** `tests/montecarlo/test_xray_dispersion.py::test_line_sits_on_the_in_medium_resonance_not_the_vacuum_one`; `tests/montecarlo/test_xray_dispersion_cuda.py::test_prologue_solves_the_in_medium_resonance`, `::test_prologue_unit_index_reproduces_the_vacuum_kinematics` (CUDA-gated: the streaming prologue's device fixed point matches the same root in float64, and `Re n = 1` collapses it back onto the pre-existing vacuum kernel bit-for-bit); convergence guard: `tests/montecarlo/test_xray_dispersion.py::test_the_two_cycle_root_is_rejected_rather_than_returned`, `::test_the_rejected_root_would_otherwise_have_passed_the_energy_window`, `::test_a_converged_root_is_untouched_by_the_guard`, `::test_the_guard_is_inert_across_the_xray_regime`
- **Notes:** Real part only: `Im n` is the same absorption already carried as the Beer-Lambert `μ(E)` escape factor, so folding it in here would double-count it. Bulk response only — no interface/Fresnel term, so grazing observation geometry is out of scope. The coherent path now takes its propagation phase from the same dispersion relation — see `xray-in-medium-propagation-phase`. Builds on `xray-refractive-index`. **2026-08-20: the contraction premise was falsified and the code now guards against it.** The `Re n = 1 − δ`, `δ ~ 1e-5–1e-3` premise behind "three passes, the third is margin" fails once the vacuum root lands in the optical/UV, where the tabulations honestly carry `Re n > 1` (carbon: 6.24–285 eV, peak 4.766 at 6.40 eV). A segment nearly perpendicular to `g` then gives `Re n (v·n̂) = 0.99931`, `denom = 6.9e-4`, and an expansive 2-cycle instead of a contraction: three passes returned whichever half pass 3 landed on (`E_res = 4942.7 eV` on the traced sample), which the CBS `1/(γ (v·g)²)` turned into a characteristic-line total ten orders too large — finite, so nothing downstream flagged it. `_in_medium_kinematics` and the CUDA prologue now check convergence and reject failures (see **Checks**); such samples violate `cbs-amplitude`'s own perturbative validity condition by 15.6×, so rejection is the correct handling rather than a workaround. Traced seed 7602 → 8.03e-07, inside the healthy population (6.6e-07–1.05e-06); healthy seeds bit-identical; zero pairs rejected at the catalog's 1000 Å production thickness, 0.233% at 1e6 Å. Found and fixed on `feature/relativistic-bethe-stopping`; see `docs/validation/ledger-transport-background.md#relativistic-bethe-stopping` and the branch task doc. **Owed to this row:** the [write-up](radiation-physics/xray-in-medium-resonance.md)'s §2 contraction argument still states the unconditional rate; its 2026-08-20 addendum records the falsification, but the section itself wants rewriting by whoever re-runs this validation, and the `rederived` status should be re-confirmed at the same time. Independent re-derivation: [write-up](radiation-physics/xray-in-medium-resonance.md)

## `xray-in-medium-propagation-phase`

- **Claim:** coherent segment-to-segment propagation phase on the in-medium wavevector: segment `j` accumulates `−δ(E) ω(E) L_esc,j` on top of the vacuum `ω d_j`, with `d_j = t_j − n̂·r_j` and `L_esc,j` the in-crystal escape path
- **Code:** `montecarlo/spectrum/lines.py::mc_spectrum` (`coherent=True`; the in-medium leg is unconditional)
- **Source:** observation-time phase `ω(t_j + n_med L_esc,j + L_vac,j)` (Jackson 14.65 kernel `exp{iω t_obs}`) with the first-order far-field path split `L_esc + L_vac = R − n̂·r_j`; the medium leg contributes `ω(Re n − 1) L_esc = −δ ω L_esc`, and `exp(i n ω L) = exp(iωL)·exp(−iδωL)·exp(−βωL)` shows it is the real partner of the Beer-Lambert amplitude `exp(−βωL) = √(exp(−μL))` already applied over the SAME path
- **Status:** rederived
- **Checks:** limiting cases — a SINGLE segment reproduces the incoherent result to 1e-10 rel, so the new factor is pure phase and does not leak into intensity; normal exit (`n̂` along the face normal) reduces to the naive `k(E) n̂·r_j` form up to a segment-independent global phase. Magnitude/sign — the relative phase between two segments at depths 5000 Å and 10000 Å (hopg 002, 100 keV, θ_obs = 119°, δ = 1.900e-4) matches the closed form `−δ(E) ω(E) (z₁ − z₂)/(−n̂_z)` = +1.588643 rad to 5.7e-13 rad (float64 rounding), identically on BOTH the batched and the per-hkl accumulation paths. Accumulation scale — ~1 µm of depth separation gives ≈π, i.e. δ ~ 1e-4 does reach order-unity phase over micron trajectories, which was the open question left by `xray-refractive-index`
- **Anchor:** `tests/montecarlo/test_xray_dispersion.py::test_interference_phase_matches_the_in_medium_closed_form`, `::test_single_segment_coherent_is_pure_phase_under_refraction`, `::test_micron_scale_depth_separation_inverts_the_interference`, `tests/montecarlo/test_xray_dispersion_cuda.py` (CUDA-gated: the reduction and stream field kernels reproduce a NumPy reference of the in-medium phase, and are bit-for-bit identical to the pre-change launch when the in-medium arguments are omitted)
- **Notes:** Deliberately NOT `ω t_j − k(E)(n̂·r_j)`: that form charges the medium's index for the whole flight to the detector, and differs from this one by more than a global phase whenever the exit direction is off the face normal. Real part only, for the same non-double-counting reason as `xray-in-medium-resonance`. Refused for LAYERED absorbers — the per-layer δ summed along the escape path (the real partner of `_stack_tau`'s per-layer μ) is not modelled. The CUDA coherent reduction kernels carry the second (per-segment scalar `L_esc,j`)×(per-energy table `δ(E) ω(E)`) product directly, and the streaming route now runs under `refractive` too — `montecarlo/spectrum/coherent_stream_jit_kernel.py::run_coherent_prologue_kernel` solves the in-medium root on device — so both GPU coherent routes carry this phase, and both inherit that solve's 2026-08-20 convergence guard (`xray-in-medium-resonance`): segment/reflection pairs whose root does not converge are dropped rather than phased. Builds on `xray-refractive-index`, `xray-in-medium-resonance`. Independent re-derivation: [write-up](radiation-physics/xray-in-medium-propagation-phase.md)

## `self-absorption`

- **Claim:** per-segment Beer–Lambert path-to-surface, cross-stack
- **Code:** `montecarlo/spectrum/lines.py::mc_spectrum`; `materials/attenuation.py::_stack_tau`
- **Source:** Beer–Lambert
- **Status:** rederived
- **Checks:** units, zero/single-layer/subdivision limits, front/back sign, lateral finite-prism path, and two-layer closed form checked
- **Anchor:** `tests/montecarlo/test_multilayer.py`; `checks/multilayer_validation_check.py`
- **Notes:** independent piecewise path-integral derivation matches; [validation write-up](radiation-physics/self-absorption.md)
