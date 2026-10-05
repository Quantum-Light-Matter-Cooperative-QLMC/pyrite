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
- **Code:** `data/catalog/crystals/*.toml::B_ang2`; `materials/crystal.py::{structure_factor,U_g}`
- **Source:** per-material primary refinements collected in [audit](materials/debye-waller-audit.md)
- **Status:** discrepancy
- **Checks:** baseline inventory; `B=8π²Uiso`; tensor limit `exp[-gᵀU_jg/2]`; phase, temperature, site, convention, and reflection-scope checks
- **Anchor:** `tests/materials/test_material_catalog.py::test_packaged_catalog_matches_independent_serialized_golden` (serialization only)
- **Notes:** 2026-07-25 baseline: 48 entries; h-BN, GeP, GeS, GeSe, PdTe2, and 2H-TaS2 now have primary-refinement temperature/conversion notes, sapphire has only an approximate range, four distinct legacy values lack value-level provenance, and 37 chemically diverse entries reuse `0.6 Å²`. Current crystal-wide scalar is insufficient when sites differ or nonparallel reflections probe anisotropic ADPs. No schema extension until source coverage establishes required site/tensor representation. **2026-09-13, scoped pass over HOPG/h-BN/MoS2/MoSe2/MoTe2 only** (rest of catalog untouched): HOPG `0.8 → 1.30 Å²`, the multiplicity-weighted `B33 = 8π²·mean(U33)` of Trucano & Chen's neutron refinement — COD 9011577, the record the entry's own `cod_id` already cites, so the value now shares a source with the lattice and basis. The earlier specimen-state objection is withdrawn: an ADP is lattice-dynamical, and HOPG's distinguishing mosaic spread is modelled separately by `mosaic_fwhm_deg`. MoS2 `0.6 → 0.47 Å²` from Schönfeld, Huang & Moss (1983), whose 2H deposit (COD 9007660) omits the ADP loop but whose 3R deposit (COD 9007661) carries it; 2H/3R MoS2 share identical trigonal-prismatic layers and differ only in stacking, so the transfer is admitted with explicit scope — unlike the rejected 1T→2H NbS2 case, which changes coordination. The prior pass recorded "no usable ADP" for MoS2 after checking COD 1010993/1531960 and did not check the entry's own 9007660/9007661 pair. h-BN unchanged at `3.45 Å²`. MoSe2 and MoTe2 remain at the `0.6 Å²` placeholder and are recorded **not reported**: every matching-phase COD deposit (2310945, 2310465, 9009147) carries zeroed `U_iso` or is a secondary compilation, and the correct MoSe2 target — Bronsema, De Boer & Jellinek (1986), which the entry's cell matches — is paywalled and was not retrieved. Sensitivity measured at all five before changing values (audit acceptance item 7): the Mo dichalcogenides' long `c` axes put the pinned `(00l)` at `|g| ≈ 0.90–1.02 Å⁻¹`, so over a realistic `B = 0.3–1.5 Å²` band the placeholder costs ≤1.6% on `(002)` and ≤6.2% on `(004)` — the two open entries are not on the accuracy critical path. HOPG/h-BN are the sensitive pair (≤5.3% and ≤19.5% over the same band). HOPG's change is production-visible: `|F002|²` −2.20%, `|F004|²` −8.50%, the serialized catalog golden moved, and `case_content_key` moved once, orphaning CAS records minted under `0.8`. Write-ups: [HOPG](materials/hopg-debye-waller-00l.md), [MoS2](materials/mos2-debye-waller-00l.md), [audit](materials/debye-waller-audit.md).

## `surface-hkl-orientation`

- **Claim:** reciprocal cleavage-plane normal `g_hkl = h b1 + k b2 + l b3` is mapped to sample `+z` by a proper minimal rotation, followed by the configured right-handed azimuth about `+z`
- **Code:** `montecarlo/geometry.py::_orientation_R`; plumbing through `campaign/sweep.py`, `montecarlo/spectrum/lines/`, `montecarlo/detector.py`, and `montecarlo/runner/__init__.py`
- **Source:** standard reciprocal-lattice geometry and Rodrigues rotation
- **Status:** rederived
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
- **Code:** `data/catalog/crystals/*.toml::cod_id`; `data/cifs/*.cif`
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
- **Code:** `validation/validation_oracles.py::validate_dans_crystal`, `::compare_lattice`, `::compare_reflection_geometry`, `::compare_structure_factor_magnitudes`
- **Source:** `Dans_Diffraction` 3.4.0 generated API/docs; Waasmaier–Kirfel (1995); independent Henke/CXRO dispersion tables; local pyrite conventions
- **Status:** unverified
- **Checks:** fail-closed threshold evaluator; fake-oracle unit tests; pinned real-backend test at 1, 2, 3, and 8 keV
- **Anchor:** `tests/materials/test_validation_oracles.py`; `uv run --group oracle python checks/dans_diffraction_oracle.py`
- **Notes:** 2026-07-25 implementation run passed: max relative `Δ\|g\|=2.22e-16`; non-resonant Waasmaier–Kirfel `Δ\|F\|²=2.65e-15`; Chantler-vs-Henke dispersive max `Δ\|F\|²=6.85%` under the cross-database 10% bound. Validation-only backend; no production import. Status remains unverified pending fresh-context review; run is not human sign-off.

## `absorption-length`

- **Claim:** X-ray absorption length / μ
- **Code:** `materials/crystal.py::absorption_length_ang`
- **Source:** Henke `f₂` convention / Beer–Lambert; `f₂` tabulation is Chantler/FFAST via `xraydb.f2_chantler` (legacy `henke_dispersion` name — see `docs/physics/atomic-physics/atomic-data-sources.md`)
- **Status:** anchored
- **Checks:** units (`μ` in Å⁻¹, length in Å); passive-medium sign; field-to-intensity factor two; zero-density/zero-`f₂` and mixture limits
- **Anchor:** `tests/materials/test_crystallography.py::test_absorption_length_matches_henke_f2_coefficient`
- **Notes:** independent derivation gives `μ=2rₑλnf₂` and matches every `2π` factor in production; [validation write-up](radiation-physics/absorption-length.md). **Photoabsorption only, by design:** `f₂` is the photoabsorption tabulation and this is the coefficient the refractive index needs (`grazing-optical-constants`, `xray-chi-zero`). A narrow-beam transmission must add coherent+incoherent removal on top — see `narrow-beam-total-attenuation`, which owns that sum; do not fold scattering in here.

## `narrow-beam-total-attenuation`

- **Claim:** the linear attenuation coefficient in a narrow-beam (good-geometry) Bouguer–Beer exponent is the **total** removal rate `μ_tot(E) = Σ_i n_i σ_tot,i(E)`, `σ_tot = σ_photo + σ_coh + σ_incoh + σ_pair,nuc + σ_pair,el`, each per-atom term from EPDL2025 (Z = 1–100, 1 eV – 100 GeV), evaluated lin-lin (the upstream ENDF law) and right-continuous at photoionization edges; NaN outside 1 eV – 100 GeV
- **Code:** `materials/photon_cross_sections.py::photon_cross_sections_ang2`, `::total_photon_cross_section_ang2`, `::_interpolate`, `::photoelectric_edges`; `materials/attenuation.py::_mu_total_inv_ang` (the sum, and the single helper behind filter plates, crystal-source self-absorption, the PXR/CBS line escape, the brem/characteristic/hard-event escape factors, the per-line tabulation, and the detector window), `::_finite_mu_or_raise` (continuum out-of-domain policy); provenance (not a physics symbol): generator `scripts/release_epdl_table.py`
- **Source:** D. E. Cullen et al., *EPDL2025: Evaluated Photon Data Library*, EPICS2025 (IAEA NDS, `NDS-IAEA-225`, evaluated Aug 2023, distributed Jan 2025; the EPDL97 evaluation, Cullen, Hubbell & Kissel, UCRL-50400 Vol. 6 Rev. 5, 1997, carried forward), ENDF-6 File 23 MT 522/502/504/517/515, `https://nuclear.llnl.gov/EPICS/ENDF2025/EPDL2025.ALL` pinned by SHA-256 `59bbd8c5…c43fd`. Additive decomposition as in Hubbell & Seltzer / Berger et al., NIST XCOM (SRD 8), used as the independent reference
- **Status:** rederived
- **Checks:** packaged table reproduces upstream MT=501 total at every upstream node and interval midpoint for Z = 1–100 to `5.0e-4` (the stored thinning tolerance); NIST XCOM total `μ/ρ` away from edges (> 2% in energy from any XCOM or EPDL edge), 1 keV – 100 GeV: ≤ 0.35% for C, N, Al, Si, ≤ 0.5% for Se, ≤ 3% for W and Pb (W 2.83% at 2 keV, Pb 2.57% at 80 keV, just below M/K edges); pair channels ≤ 0.1% above 1.5 MeV; incoherent ≤ 1.3% 10 keV – 50 MeV; units (barn → Å² per atom, pinned end-to-end by the Klein–Nishina anchor `σ_incoh/Z → σ_KN` at 500 keV); thresholds (`σ_pair,nuc = 0` below `2mₑc²`, `σ_pair,el = 0` below `4mₑc²`); `n_a → 0` limit; isolation of the refractive-index path (`μ = 2kβ` identity preserved exactly); edge right-continuity
- **Anchor:** `tests/materials/test_photon_cross_sections.py` (`::test_total_mass_attenuation_matches_xcom`, `::test_pair_production_matches_xcom`, `::test_incoherent_matches_xcom_below_50_MeV`, `::test_photoelectric_edges_are_right_continuous`, `::test_packaged_table_is_pinned_and_compact`, `::test_generator_thinning_keeps_edges_zeros_and_tolerance`, `::test_generator_npz_is_byte_reproducible`); `tests/materials/test_attenuation.py` (`::test_incoherent_term_approaches_klein_nishina_at_500_keV`, `::test_pair_production_vanishes_below_its_thresholds`, `::test_scattering_dominates_carbon_attenuation_at_20_keV`, `::test_refractive_index_path_stays_photoabsorption_only`, `::test_continuum_guard_refuses_undefined_attenuation_above_the_floor`); `tests/montecarlo/test_multilayer.py::test_stack_tau_carries_scattering_for_a_crystalline_source`; reference data `tests/data/xcom/`
- **Notes:** **Issue #274 (2026-10-01) replaced the source.** Until then photoabsorption came from Chantler/FFAST `f₂` and scattering from Elam (`xraydb.mu_elam`), NaN above 800 keV and with no pair production. One evaluated library now supplies every channel to 100 GeV; Chantler/FFAST stays the source of `f₂` for the refractive index, `χ_g`/`U_g`, and Si sensor absorption (`absorption-length`), so `μ_photo ≠ 2kβ` by design (different compilations). **Change against the old μ below 800 keV** (26 catalog elements, 100 eV – 790 keV): median 1–3%; up to 10–15% away from edges at 1–2 keV, where XCOM sides with EPDL (C/Si within 0.3%) and Chantler's photoelectric cross section departs from Scofield's; large local ratios within a few eV of edges, because EPDL's edges sit at their own energies (C K 288 eV vs Chantler 283.8 eV). The edge locator brackets EPDL edges for the escape `μ` (`line-window-seeding`, `continuum-node-refinement`). **Out-of-domain policy:** NaN outside 1 eV – 100 GeV; the continuum scorers (wide brem grid, hard-event scorer, characteristic escape) raise on undefined `μ` at `E ≥ 1 eV`, and keep the historical `μ = 0` only for sub-eV nodes of an unfloored grid; the line routes drop NaN samples. **Packaging:** derived table, lin-lin knot-thinned to `5e-4` relative (1.5 MB, deterministic `.npz`, SHA-pinned); ADR-0014 class (b) candidate like EEDL. **Good geometry is assumed and not enforced** — no build-up factor; Bragg coherent removal at a reflection condition is not the free-atom average used here (open condition unchanged from the Elam era). Photonuclear absorption (sub-percent, 10–30 MeV) is not in EPDL. Identity marker `attenuation_model` orphans earlier records once. **2026-10-01 fresh-context re-derivation matches (EPDL2025):** units, `n → 0`, Klein–Nishina (σ_incoh/(Zσ_KN) = 0.9992 C … 0.978 Pb at 500 keV, → 1 from below as Z falls), pair thresholds and edge right-continuity (all 1519 MT 522 edges) pass; an independent TAB1 reader matches the packaged table against upstream MT 501 to `4.9999e-4` at nodes and midpoints for all 100 elements, and a regeneration from the pinned tape reproduces the packaged SHA. Human sign-off pending. [validation write-up](radiation-physics/narrow-beam-total-attenuation.md)

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

- **Claim:** first-order Snell CXR kinematics: `h = ∇L_esc = −e/(e·n̂)`, `k_eff = ω(n̂ + δh)`, `D(E) = 1 − v·n̂ − δ(E) v·h`, `E_res D(E_res) = ħc v·g`; vector `k·v`, `k·g`, `g²+2k·g`, `k²`, and the PXR numerator’s `(k+g)·e_s` including `k_eff·e_s`
- **Code:** `montecarlo/spectrum/lines/_kernels.py::_in_medium_kinematics`, `montecarlo/spectrum/segment_escape.py::segment_escape_gradient`, `montecarlo/spectrum/lines/_spectrum.py::mc_spectrum`; CUDA twin `montecarlo/spectrum/coherent_stream_jit_kernel.py::_coherent_prologue_kernel`
- **Source:** tangential continuity and Maxwell dispersion, closed with `ω = v·(k+g)`; Zhai SI Eqs. (7)–(9), Eq. (3) Maxwell inverse; Feranchuk–Spence 2000 Eqs. (10)/(13)/(14)
- **Status:** rederived
- **Checks:** units, signs, vacuum and normal-exit limits pass; both NumPy routes place the HOPG 002, 100 keV, 119° peak at 1600.5865 eV within one fp64 grid step; box side exits use their own normal. Normal exit exactly reproduces the bulk root. Vector invariants and fused amplitudes match independent dot products. Three fixed-point passes retain `_RESONANCE_ROOT_RTOL = 1e-3`; critical-angle pairs `2 |δ| h² >= 1` are refused. Contraction requires `|E δ′ v·h / D| < 1`.
- **Anchor:** `tests/montecarlo/test_xray_dispersion.py::test_line_sits_on_the_in_medium_resonance_not_the_vacuum_one`, `::test_side_face_line_uses_its_own_snell_root`, `::test_normal_exit_snell_root_is_exactly_the_bulk_root`, `::test_critical_angle_snell_root_is_rejected`, `::test_snell_block_and_fused_amplitudes_match_full_vectors`, `::test_dispersive_formation_integral_has_the_derivative_jacobian`; convergence controls in the same module; CUDA-gated `tests/montecarlo/test_xray_dispersion_cuda.py::test_prologue_solves_the_in_medium_resonance`, `::test_prologue_snell_amplitudes_match_full_vector_numpy`, `::test_prologue_unit_index_reproduces_the_vacuum_kinematics`
- **Notes:** Issue #187 replaces the bulk-direction root. χ_g, U_g and μ sample the new root. Geometric pieces preserve the incoherent parent duration and multiply mean transmission by their length fraction. The coherent factor retains its vacuum centre plus full escape phase, avoiding double-counting. User chose full δ(E): local Jacobian `J = D − E δ′ v·h`, narrow-line coherent/incoherent yield ratio `D/J`; exact equality is a constant-index limit. First-order Snell; no Fresnel/interface mode normalization or polarization matching. `LINE_ESCAPE_MODEL = "segment-mean-v3-snell-resonance"` forks identities. Fresh-context verification (2026-10-05): `rederived` for root, vector invariants, general-vector PXR/CBS algebra and piece weighting. Current CUDA execution is gated. Human sign-off remains separate. [derivation record](radiation-physics/xray-in-medium-resonance.md)

## `xray-in-medium-propagation-phase`

- **Claim:** coherent segment-to-segment propagation phase on the in-medium wavevector: segment `j` accumulates `−δ(E) ω(E) L_esc,j` on top of the vacuum `ω d_j`, with `d_j = t_j − n̂·r_j` and `L_esc,j` the in-crystal escape path
- **Code:** `montecarlo/spectrum/lines/_spectrum.py::mc_spectrum` (`coherent=True`; the in-medium leg is unconditional)
- **Source:** observation-time phase `ω(t_j + n_med L_esc,j + L_vac,j)` (Jackson 14.65 kernel `exp{iω t_obs}`) with the first-order far-field path split `L_esc + L_vac = R − n̂·r_j`; the medium leg contributes `ω(Re n − 1) L_esc = −δ ω L_esc`, and `exp(i n ω L) = exp(iωL)·exp(−iδωL)·exp(−βωL)` shows it is the real partner of the Beer-Lambert amplitude `exp(−βωL) = √(exp(−μL))` already applied over the SAME path
- **Status:** rederived
- **Checks:** limiting cases — a SINGLE segment reproduced the incoherent result to 1e-10 rel, so the new factor was pure phase and did not leak into intensity (superseded by issue #181: the leg's within-segment slope now enters each piece's formation factor, which shifts a single segment's line to the escape-path root — see Notes); normal exit (`n̂` along the face normal) reduces to the naive `k(E) n̂·r_j` form up to a segment-independent global phase. Magnitude/sign — the relative phase between two segments at depths 5000 Å and 10000 Å (hopg 002, 100 keV, θ_obs = 119°, δ = 1.900e-4) matches the closed form `−δ(E) ω(E) (z₁ − z₂)/(−n̂_z)` = +1.588643 rad to 5.7e-13 rad (float64 rounding), identically on BOTH the batched and the per-hkl accumulation paths. Accumulation scale — ~1 µm of depth separation gives ≈π, i.e. δ ~ 1e-4 does reach order-unity phase over micron trajectories, which was the open question left by `xray-refractive-index`
- **Anchor:** `tests/montecarlo/test_xray_dispersion.py::test_interference_phase_matches_the_in_medium_closed_form`, `::test_micron_scale_depth_separation_inverts_the_interference`, `::test_single_segment_coherent_line_sits_on_the_escape_path_root` (since issue #181), `tests/montecarlo/test_xray_dispersion_cuda.py` (CUDA-gated: the reduction and stream field kernels reproduce a NumPy reference of the in-medium phase, and are bit-for-bit identical to the pre-change launch when the in-medium arguments are omitted)
- **Notes:** Deliberately NOT `ω t_j − k(E)(n̂·r_j)`: that form charges the medium's index for the whole flight to the detector, and differs from this one by more than a global phase whenever the exit direction is off the face normal. Real part only, for the same non-double-counting reason as `xray-in-medium-resonance`. Refused for LAYERED absorbers — the per-layer δ summed along the escape path (the real partner of `_stack_tau`'s per-layer μ) is not modelled. The CUDA coherent reduction kernels carry the second (per-segment scalar `L_esc,j`)×(per-energy table `δ(E) ω(E)`) product directly, and the streaming route now runs under `refractive` too — `montecarlo/spectrum/coherent_stream_jit_kernel.py::run_coherent_prologue_kernel` solves the in-medium root on device — so both GPU coherent routes carry this phase, and both inherit that solve's 2026-08-20 convergence guard (`xray-in-medium-resonance`): segment/reflection pairs whose root does not converge are dropped rather than phased. Builds on `xray-refractive-index`, `xray-in-medium-resonance`. Independent re-derivation: [write-up](radiation-physics/xray-in-medium-propagation-phase.md) **2026-10-05 (#187):** frozen amplitudes and the incoherent route now share the phase’s Snell stationary root. Full δ(E) is unchanged; its derivative Jacobian is derived and anchored under `xray-in-medium-resonance`. **2026-09-26 (issue #181):** the leg is now integrated along each linear escape piece rather than frozen at the midpoint (`coherent-formation-absorption`): the relative phase between segments is unchanged, but its within-piece slope makes the coherent route exactly split invariant and places a single segment's coherent line at the escape-path root.

## `self-absorption`

- **Claim:** per-segment Beer–Lambert path-to-surface, cross-stack
- **Code:** `montecarlo/spectrum/lines/_spectrum.py::mc_spectrum`; `materials/attenuation.py::_stack_tau`
- **Source:** Beer–Lambert
- **Status:** rederived
- **Checks:** units, zero/single-layer/subdivision limits, front/back sign, lateral finite-prism path, and two-layer closed form checked
- **Anchor:** `tests/montecarlo/test_multilayer.py`; `checks/multilayer_validation_check.py`
- **Notes:** independent piecewise path-integral derivation matches; [validation write-up](radiation-physics/self-absorption.md)

## `photon-continuum-floor`

- **Claim:** a photon-continuum grid's strictly positive lowest node is the larger of the medium's bulk free-electron plasma energy `ħω_p = ħ√(n_e e²/ε₀mₑ)`, with `n_e = Σᵢ nᵢ Zᵢ` from the medium's own catalog number densities, and the lowest energy at which every table the continuum pipeline evaluates has real support
- **Code:** `materials/attenuation.py::plasma_energy_eV`; `_photon_continuum_floor.py::photon_continuum_floor_eV`; `energy_grid/floor.py::geometric_continuum_grid`
- **Source:** Drude free-electron dielectric function `ε(ω) = 1 − ω_p²/ω²`, Jackson, *Classical Electrodynamics* 3rd ed., Sec. 7.5; the same quantity as `ħω_p = 28.816√(ρ⟨Z/A⟩)` eV in the PDG "Passage of particles through matter" presentation of the Sternheimer density effect (Sternheimer, Berger & Seltzer, *Atomic Data and Nuclear Data Tables* **30**, 261 (1984)). Ties to this repository's own optics through `grazing-optical-constants`: `δ = (rₑλ²/2π) n_a f₁` is identically `ω_p²/(2ω²)` in the free-electron limit `f₁ → Z`
- **Status:** rederived
- **Checks:** units (eV); `√n_e` scaling; dilute limit `n_e → 0 ⇒ ħω_p → 0`, where the floor falls back to table support; independent oracle — inverting the packaged PDG/Sternheimer `C̄ = 2 ln(I/ħω_p) + 1` for silicon gives `31.0482 eV` against `31.0498 eV` computed from the catalog number density, `5.2e-5` relative, with no shared code path; mass conservation across the separate zero-energy detector channel
- **Anchor:** `tests/energy-grid/test_continuum_floor.py`
- **Notes:** the floor is a **bandwidth policy backed by a validity bound**, not a new emission or escape model — no transport or radiation kernel evaluates `ħω_p`. Its role is to refuse placing continuum nodes where `δ = ω_p²/(2ω²)` exceeds `1/2` and the weakly-refracting transparent-medium expansion behind photon escape and self-absorption has collapsed. All-`Z` free response is assumed, exact only for `ω` above every binding energy; near `ω_p` the true response is collective and band-structure dependent, which is precisely why the model is not claimed to hold below it. No Drude damping term, so a real plasmon resonance shifts by order `(1/τ)/ω_p`. For every catalog medium the plasma energy binds (smallest is `sio2` at `30.201 eV`, largest `ptbi2` at `66.248 eV`) against a `12 eV` data-support floor set by the digitized Eagle XO QE curve; EEDL photon spectra (`0.1 eV`), EPDL2025 attenuation (`1 eV`) and Chantler/FFAST (`1.01 eV`) never bind. Measured table limits are recorded in `_photon_continuum_floor.py::DATA_SUPPORT_LIMITS_EV`. Human sign-off pending.

## `continuum-node-refinement`

- **Claim:** a photon-continuum grid's nodes are placed geometrically, because a grid uniform in `u = ln E` equidistributes the relative midpoint-quadrature error `ε_i ≈ (h_i²/24)|n''/n|` for a locally power-law integrand, and are refined only where the modelled integrand is not smooth on the scale of its own step: at absorption edges located in the table each element is read from — the medium's EPDL2025 photoionization edges (exact discontinuities of its escape `μ`), anchored as a pair straddling the edge so a bin boundary falls on it, and the Si detection path's Chantler `f2` jumps — and at the kinematic endpoint above which the bremsstrahlung source term is identically zero
- **Code:** `energy_grid/refine.py::refined_continuum_grid`; `energy_grid/refine.py::continuum_refinement_marks`; `montecarlo/spectrum/line_seeds.py::absorption_edge_brackets`
- **Source:** midpoint-rule truncation error (standard quadrature; e.g. Press et al., *Numerical Recipes* 3rd ed., Sec. 4.1) applied to the midpoint bin masses of `eq-grid-midpoint-edges`, with the two non-smooth features of `n(E) = S(E) exp(−μ(E)ℓ)`: the Chantler/FFAST `f2` absorption jump carried by `materials/crystal.py::absorption_length_ang`, and the EEDL MF=26/MT=527 photon-spectrum cutoff at `k = T`. Derivation, assumptions, and limiting case in [energy-grid semantics](../physics/radiation-physics/energy-grid-semantics.md) (`continuum-node-refinement`, `eq-grid-quadrature-error`)
- **Status:** rederived
- **Checks:** limiting case — a band with no located edge and no interior endpoint returns the geometric baseline bit-identically (`np.array_equal`), and an edge whose jump lies outside the band is reported and dropped rather than refused; endpoint placement verified to put a midpoint bin **edge** at the cutoff to `1e-9 eV` where the plain baseline's nearest edge was `38.77 eV` away inside a `105 eV` bin; edge anchors verified to be adjacent grid nodes equal to the located native Chantler bracket; edge positions verified material-derived (`wse2` seeds `W`/`Se` marks, `hopg` does not); refined grids on `hopg`/`silicon`/`wse2`/`diamond` pass `_grid_semantics.validate_backend_coordinates` at float32, so no merge leaves a near-degenerate interval
- **Anchor:** `tests/energy-grid/test_continuum_refinement.py`
- **Notes:** a **node-placement policy backed by a quadrature-error argument**, not a new emission or escape model — no transport or radiation kernel changes, and the same densities are evaluated on a different set of coordinates. Measured on the fixed-trajectory HOPG convergence case (`seed=0`, 3 electrons, 30 keV, `Timepix3(n_mc=32, seed=7)`), refinement moves every gated continuum observable far inside its allocated budget: at 2049 baseline nodes, continuum yield `2.6e-6` and centroid `2.5e-6` against a `1e-3` `intrinsic_source` budget, Timepix3 `9.4e-6` and Eagle XO `4.9e-6` against a `1e-2` `detected_counts` budget; with the endpoint inside the band the centroid shift rises to `2.5e-5`, still `40×` inside budget, and every delta shrinks as the baseline refines. Cost is `+19`/`+16`/`+10` nodes on 2049/4097/8193 (`0.93%`/`0.39%`/`0.12%`) with the 30 keV endpoint inside a 40 keV band, and `+18`/`+15`/`+9` with the endpoint outside a 29 keV one, falling with node count because refinement displaces baseline nodes as well as adding them (4/7/13 displaced against a fixed 23 mark nodes); grid construction costs `1.9`-`2.3 ms` of edge lookup against `0.13`-`0.23 ms` for the plain baseline, and continuum evaluation is unchanged within noise. Refinement adds only interior nodes and never moves the band endpoints, so Timepix input channels — and the seeded MC response matrix — are bit-identical between a refined mesh and its baseline. Narrow line windows and adaptive line-axis refinement are a different axis and are **not** claimed here. **Issue #274:** the medium's edges now come from EPDL2025 (`absorption_edge_brackets(..., tables=("epdl",))`, labelled `(EPDL)`), the Si path's still from Chantler; the EPDL pair borrows the same shell's Chantler native spacing (else 0.1% of the edge energy). The measured node costs above predate this and were not re-measured. Human sign-off pending.
