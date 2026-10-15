# Channeling

Part of the [physics validation ledger](physics-validation-ledger.md). See the [validation methodology](methodology.md) for the status lifecycle and the [domain inventories](domain-inventories.md) for a claim-by-claim index. Design: [ADR-0018](../adr/0018-quantum-channeling-zone.md) and the [channeling radiation walkthrough](../research/physics/channeling-radiation-physics.md). These rows cover the static, scattering-free planar profile (#503). Angles are plane-local; `cr-plane-incidence` maps the laboratory tilt and #25 `CrystalCut` orientation onto them.

## `cr-debye-b-of-t`

- **Claim:** site Debye–Waller `B(T) = 8π²⟨u_x²⟩` is the linear interpolation of phonon-DOS tabulated rows inside their range. Outside the range it is the Debye model `B = (6h²/(m_u A k_B))(T/Θ²)[φ(Θ/T) + Θ/(4T)]`, with `φ(y) = (1/y)∫₀ʸ x/(eˣ−1) dx` and `Θ` fitted to the nearest tabulated end.
- **Code:** `materials/channeling/thermal.py::{debye_integral,debye_b_factor,fit_debye_temperature,ThermalSite.b_factor,thermal_b_factor}`; `data/channeling_thermal.toml`
- **Source:** Willis & Pryor (1975); Peng, Ren, Dudarev & Whelan, Acta Cryst. A52, 456 (1996), Eq. 8 and Table 1 ([doi:10.1107/S010876739600089X](https://doi.org/10.1107/S010876739600089X))
- **Status:** anchored
- **Checks:** Units: Å² from the CODATA prefactor 6h²/(m_u k_B) = 11 490 Å²·K. Peng quotes 11 492; `φ(y→0) = 1`; `φ(y) → π²/(6y)`; `φ(10)` equals 0.16444 by independent quadrature; Classical `B ∝ T`; T-independent zero-point limit; The fitted Θ_D reproduces its anchor exactly. Si Θ_D = 520 K, in line with Peng's high-temperature ≈500 K; Tabulated rows are reproduced exactly, and the model is continuous at both range ends.
- **Anchor:** `tests/materials/test_channeling_profile.py::{test_debye_integral_limits,test_debye_b_factor_classical_and_zero_point_limits,test_fit_debye_temperature_round_trips,test_thermal_rows_reproduced_and_extrapolation_continuous}`
- **Notes:** **Data.** Rows 170–280 K are transcribed from Peng Table 1. Rows below 170 K were ambiguous in the published layout and were not used; **Why interpolation.** A one-parameter Debye fit at 280 K is 11% low for Si at 180 K but only −1.6% for diamond, matching Peng's warning that the Debye model fails for Si. The model is therefore used only outside the tabulated range. Cryogenic Si values carry that model error; **Catalog discrepancy.** The PDOS diamond value (0.144 Å² at 293 K) disagrees with the catalog `B_ang2 = 0.21` used by PXR/CBS. That is a `debye-waller-catalog-provenance` follow-up, not changed here; **2026-10-15 fresh-context verdict `rederived`** ([write-up](channeling/cr-debye-b-of-t.md)): CODATA prefactor 11 490.29 Å²·K vs Peng's rounded 11 492 (0.015%); B(T) identical to an independent quad/brentq reference to ~16 digits over 10–1000 K; all 20 TOML rows (170–280 K) match Peng Table 1; rows below 170 K unconfirmable in the extract (slope break at 160 K supports excluding them). `φ(10) = 0.164443` by independent quadrature; Peng's Table 3 is not legible in the extract, so the earlier "matches Peng's tabulated value" wording is withdrawn. Human sign-off pending.

## `cr-mott-bethe-factor`

- **Claim:** the electron–atom potential Fourier component uses the Mott–Bethe numerator `Z − f0(g)` over `g²` (Poisson's equation), with only the energy-independent X-ray form factor `f0` (Waasmaier–Kirfel, valid to `s ≤ 6 Å⁻¹`). The anomalous `f′, f″` are excluded.
- **Code:** `materials/channeling/potential.py::planar_profile`
- **Source:** Mott–Bethe formula, `f^e(s) = (Z − f_x(s))/(8π² a₀ s²)` (International Tables Vol. C, §4.3); Andersen et al., Phys. Scr. 28, 308 (1983)
- **Status:** anchored
- **Checks:** Harmonics are kept up to `s = g/(4π) ≤ 6 Å⁻¹`, and `g = 0` is excluded, so the `s → 0` limit is never evaluated; The diamond (220) coefficient equals `−U_g` exactly, since `U_g` uses bare `f0` for non-edge-prone elements.
- **Anchor:** `tests/materials/test_channeling_profile.py::test_planar_coefficient_is_negative_cbs_coupling`
- **Notes:** **Finding.** `materials.crystal.U_g` adds `f′` for `_EDGE_PRONE` elements (Si, P, Fe, …) even with `use_henke=False`. For Si (220) that moves the CBS potential coupling 3.7% from the electrostatic `Z − f0` value. That belongs to the CBS model (`cbs-amplitude`) and is reported for review there; No Doyle–Turner cross-check yet; the Si(110) line-energy anchor (`cr-transverse-eigenproblem`) agrees with Azadegan's Doyle–Turner potential to 0.6%; **2026-10-15 fresh-context verdict `rederived`** ([write-up](channeling/cr-mott-bethe-factor.md)): Poisson + Born gives `u = −4πe²(Z−f0)/q² = −(2πħ²/m) f^e`; `cromer_mann_f0` equals `xraydb.f0` (no `f′/f″`) at s = 0.26–5.9 Å⁻¹; independent Si (220) −4.31575 eV vs code −4.31589 eV (3×10⁻⁵, lattice-constant rounding). Human sign-off pending.

## `cr-planar-continuum-potential`

- **Claim:** the plane-averaged electron potential energy is `U(x) = Σ_{n≠0} U_n e^{i n g x}`, with `U_n = −(4π e²/(V g_n²)) Σ_j (Z_j − f0_j(g_n)) e^{i2π n hkl·R_j} e^{−B_j s_n²}`. The mean inner potential `U_0` is set to 0. The true period comes from the gcd of non-extinct harmonics. The nuclear and electron density coefficients use the same thermal smearing.
- **Code:** `materials/channeling/potential.py::{planar_profile,PlanarProfile.potential}`
- **Source:** Andersen, Bonderup, Lægsgaard & Sørensen, Phys. Scr. 28, 308 (1983); Azadegan, Comput. Phys. Commun. 184, 1064 (2013), Eqs. 4–5
- **Status:** anchored
- **Checks:** Units: eV from e²/(V g²); Electron wells sit on the atomic planes, consistent with the electron charge being −e; Si (110)/(100)/(111) periods equal d₂₂₀, d₄₀₀ and d₁₁₁. Doubled indices give the same profile; The centrosymmetric (110) coefficients are real; Mean densities are 8/V atoms and 112/V electrons; Heating monotonically reduces the well depth.
- **Anchor:** `tests/materials/test_channeling_profile.py::{test_extinctions_set_true_planar_period,test_si110_wells_sit_on_planes_with_literature_depth,test_density_means_and_heating}`
- **Notes:** Si(110) depth is 21.0 eV at 293.15 K. Literature reports ~22 eV, and Azadegan 2013 Fig. 1 shows ≈21 eV; Diamond(110) depth is 23.6 eV. The walkthrough's 25–30 eV estimate predates the PDOS `B`; **2026-10-15 fresh-context verdict `rederived`** ([write-up](channeling/cr-planar-continuum-potential.md)): independent 8-atom Si(110) rebuild with `|n| ≤ 46` matches `U_220/U_440/U_660` to ~1e-4 eV, depth 21.039 vs 21.040 eV, `U(x)` to 4e-4 eV; truncating at `|n| ≤ 5` gives only 17.7 eV, so high harmonics carry ~3 eV of the depth. Caveat recorded by the verifier: the period gcd keys on `|U_n|`, not the geometric structure factor, which is safe while `Z − f0 > 0` but would misread an accidental zero. Human sign-off pending.

## `cr-transverse-eigenproblem`

- **Claim:** transverse states solve `[−(ħ²/2γm) d²/dx² + U(x)] ψ = E_⊥ ψ`, expanded in Bloch plane waves, with `H_{mm′} = (ħ²/2γm)(k + m g)² δ_{mm′} + U_{m−m′}`. A state is bound when it lies below `max U`.
- **Code:** `materials/channeling/transverse.py::{transverse_states,TransverseStates.wavefunction}`
- **Source:** Andersen et al., Phys. Scr. 28, 308 (1983); Azadegan (2013), Eqs. 2–6
- **Status:** anchored
- **Checks:** Free limit: the plane-wave spectrum, with no bound states; Cosine (Mathieu) limit: spacings `ħω − (a g²/4)(n+1)` to 2×10⁻⁴; Parity alternates at `k = 0`; Basis convergence between M = 16 and 48; Bound count grows as √γ: 6 → 12 → 24 at 40 → 160 → 640 MeV.
- **Anchor:** `tests/materials/test_channeling_profile.py::{test_free_limit_has_no_bound_states,test_harmonic_limit_gives_equal_spacing,test_parity_alternates_and_selection_rule_emerges,test_bound_state_count_scales_as_sqrt_gamma,test_si110_40mev_matches_azadegan}`
- **Notes:** **Azadegan anchor (Si(110), 40 MeV).** Six bound states, matching the published six. `2γ²(ε₁−ε₀)` = 76.69 keV against the published 76.2071 keV at `k = 1.3089 Å⁻¹`, 0.6% apart. The remaining difference is consistent with Doyle–Turner factors and a different thermal amplitude; **Walkthrough correction.** Computed Si(110) bound-state counts are 1/2/3/5/9 at 1/3/10/30/100 MeV, below its WKB estimates (~2/4/7); Watson 1–3 MeV line anchors not yet pinned; **2026-10-15 fresh-context verdict `rederived`** ([write-up](channeling/cr-transverse-eigenproblem.md)): an independent plane-wave solver agrees to 0.0 eV at four `k`; Mathieu residual 7×10⁻⁴ eV on a 4.04 eV spacing; per-period norm 1 to 1e-14; the 0.64% offset from Azadegan is attributed to the potential model. Human sign-off pending.

## `cr-dipole-matrix-element`

- **Claim:** the transverse dipole matrix element between Bloch states at the same `k` is `x_fi = −i (ħ²/(γm)) Σ_m c_f*(k + m g) c_i / (E_f − E_i)`, from `[H, x] = −iħp/(γm)`. Parity selection emerges rather than being imposed.
- **Code:** `materials/channeling/transverse.py::TransverseStates.dipole_ang`
- **Source:** standard commutator identity; Andersen et al. (1983)
- **Status:** anchored
- **Checks:** Units: Å; Same-parity elements are below 10⁻¹² at `k = 0`; Agrees with direct `⟨ψ_f|x|ψ_i⟩` quadrature over one centred period to 2×10⁻³ for deep states.
- **Anchor:** `tests/materials/test_channeling_profile.py::{test_parity_alternates_and_selection_rule_emerges,test_dipole_matches_position_operator_for_deep_states}`
- **Notes:** Consumed by the `cr-dipole-rate` emission row (#505).; **2026-10-15 fresh-context verdict `rederived`** ([write-up](channeling/cr-dipole-matrix-element.md)): `|x_fi|` agrees with an independent implementation to 6×10⁻¹⁷ Å; commutator vs direct quadrature 1e-8 for (1,0),(2,1),(3,0). The phase is eigenvector-gauge dependent; consumers use `|x_fi|²`. Human sign-off pending.

## `cr-initial-occupation`

- **Claim:** an electron entering at angle `ψ` to the plane is the plane wave with `k_x = pψ/ħ`. Its populations are `P_n = |c_n(m₀)|²`, where `k + m₀ g = k_x`, which sums to 1. A Gaussian divergence is averaged by Gauss–Hermite quadrature. Surface refraction is not applied.
- **Code:** `materials/channeling/transverse.py::entry_populations`
- **Source:** Andersen et al. (1983); Azadegan (2013), Eq. 17
- **Status:** anchored
- **Checks:** Unitarity to 10⁻¹²; Even states only at `ψ = 0`; `P(ψ) = P(−ψ)`; The divergence average stays normalized and mixes parities.
- **Anchor:** `tests/materials/test_channeling_profile.py::{test_entry_populations_unitary_and_parity_selective,test_divergence_average_is_normalized_and_mixes_parity}`
- **Notes:** At 40 MeV Si(110), `P₀(0) = 0.24`, consistent with Azadegan 2013 Fig. 3; Surface refraction by the mean inner potential depends on the face/lattice geometry. It is deferred to the orientation coupling (#25), as is the laboratory-frame incidence angle; **2026-10-15 fresh-context verdict `rederived`** ([write-up](channeling/cr-initial-occupation.md)): independent version agrees to 1e-14 at 0 and 120 µrad tilt and with 100 µrad divergence, including the `k_x = g/2` fold edge; probabilists' Gauss–Hermite with normalized weights is equivalent. Orders beyond `M` contribute nothing (documented). Refraction and laboratory incidence are now supplied by `cr-plane-incidence`. Human sign-off pending.

## `cr-doppler-line-energy`

- **Claim:** a transverse transition emits at `ħω = ΔE_⊥/(1 − β cos θ)`, evaluated as `1/(γ²(1+β)) + 2β sin²(θ/2)`, where `θ` is the photon angle to the electron velocity. Photon recoil `O(ħω/E)` is neglected.
- **Code:** `materials/channeling/transverse.py::{doppler_denominator,line_energy_eV}`
- **Source:** relativistic kinematics; Andersen et al. (1983)
- **Status:** anchored
- **Checks:** Forward limit is `γ²(1+β)ΔE`; Agrees with the naive form at moderate angles; Cross-module: equals the PXR/CBS vacuum resonance denominator of `materials/emission.py::photon_energies` to 10⁻¹², at 5 MeV/3°, 5 MeV/20° and 30 keV/120°.
- **Anchor:** `tests/materials/test_channeling_profile.py::{test_forward_line_energy_is_doppler_upshift,test_doppler_denominator_matches_pxr_resonance}`
- **Notes:** Shares the `line-energy-dispersion` convention: `θ` relative to `v`, positive frequency.; **2026-10-15 fresh-context verdict `rederived`** ([write-up](channeling/cr-doppler-line-energy.md)): the cancellation-free identity holds to ~1e-32 at 1 GeV/1e-4 rad in 50-digit arithmetic; the `photon_energies` equality rests on the existing anchor test. Human sign-off pending.

## `cr-mean-inner-potential`

- **Claim:** the electron's mean potential energy in the crystal is the `g → 0` limit of the planar sum, `U_0 = −(4π e²/V) Σ_j Z_j⟨r²⟩_j/6`, using `f0(g) = Z − Z⟨r²⟩g²/6 + O(g⁴)`. It is the neutral independent-atom value and is negative, so electrons gain `−U_0` on entry.
- **Code:** `materials/channeling/entry.py::mean_inner_potential_eV`
- **Source:** Ibers, Acta Cryst. 11, 178 (1958); Spence & Zuo, *Electron Microdiffraction* (1992)
- **Status:** rederived
- **Checks:** Units are eV from e²/V. The sign is attractive. The two-node `g²` slope at `g = 0.05, 0.1 Å⁻¹` has `O(g²⟨r⁴⟩)` error of about 2e-3 relative (0.03 eV for Si). Si gives −13.80 eV. Measured Si mean inner potential is ≈12.6 V; independent-atom values run ~10% high because of bonding. Diamond is deeper.
- **Anchor:** `tests/materials/test_channeling_entry.py::test_mean_inner_potential_is_attractive_and_plausible`
- **Notes:** Used only for entrance-face refraction (`cr-plane-incidence`). Transverse levels are referenced to `U_0 = 0` (`cr-planar-continuum-potential`). **2026-10-15 fresh-context verdict `rederived`** ([write-up](channeling/cr-mean-inner-potential.md)): the exact Waasmaier–Kirfel slope `Σ a_i b_i/16π²` gives Si −13.830 eV and diamond −20.816 eV, against the code's −13.799 and −20.790. The whole gap is the finite-difference error. Independence caveat: the verifier saw the function body after writing its formula. The anchor test is a plausibility range rather than a reference value, so the row stays `rederived` until a measured mean inner potential is pinned. Human sign-off pending.

## `cr-plane-incidence`

- **Claim:** the channeling plane normal in the sample frame is `n = R g_hkl/|g_hkl|`, with `R` the `CrystalCut` lattice rotation. The vacuum beam direction follows the `tilted_geometry` tilt convention. At the entrance face (`+z`) the tangential momentum is conserved and the normal component becomes `sqrt(p_in² − p_∥²)`, with `p_in` at kinetic energy `T − U_0`. The transverse wavevector is `k_x = p_in·n`, and the effective incidence passed to `entry_populations` is `k_x/p`.
- **Code:** `materials/channeling/entry.py::plane_incidence`
- **Source:** reciprocal-lattice geometry (`surface-hkl-orientation`); electron refraction at a crystal surface by the mean inner potential (Spence & Zuo 1992)
- **Status:** anchored
- **Checks:** The beam direction equals `montecarlo.geometry.tilted_geometry` at four tilts; Tilting across perpendicular (110) planes gives `ψ = tilt`; tilting along them gives 0; For planes perpendicular to the face, refraction leaves `k_x` unchanged; Tangential momentum is conserved and `|p_in|` follows `T − U_0`; With the vacuum option, `ψ_eff = sin ψ`; A 1° `CrystalCut` miscut tilts (100) planes by exactly 1°; The catalog (440) face puts (110) planes at 90°; The beam must enter through the face; Lab-mapped populations equal plane-local populations.
- **Anchor:** `tests/materials/test_channeling_entry.py::{test_beam_direction_matches_transport_tilt_convention,test_tilt_across_versus_along_a_perpendicular_plane,test_refraction_conserves_tangential_momentum_and_raises_normal_momentum,test_miscut_tilts_channeling_planes_off_the_face_normal,test_catalog_face_orientation_is_default,test_cut_crystal_entry_feeds_populations}`
- **Notes:** Issue #503, consuming #25's `CrystalCut`/`ResolvedCut.rotation`. At MeV energies refraction is negligible: `δk_x ≈ n_z(−U_0)/(ħcβ)`, about 6×10⁻⁹ rad at 40 MeV for planes nearly perpendicular to a 1°-miscut face, and 3.4×10⁻⁷ rad for (004) planes (`n_z = sin 1°`), against 2.4×10⁻⁴ rad at 30 keV. It matters at keV energies and for planes inclined to the face. The γ change inside the crystal (`U_0/E ~ 1e-5`) is neglected in the states. **2026-10-15 fresh-context verdict `rederived`** ([write-up](channeling/cr-plane-incidence.md)): an independent construction over planes (110), (004), (220), (111) and (1-10), 0° and 1° miscut, 40 MeV and 30 keV, and six tilt/azimuth pairs agrees to 4e-16. Human sign-off pending.
