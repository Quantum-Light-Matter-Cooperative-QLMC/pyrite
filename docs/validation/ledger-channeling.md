# Channeling

Part of the [physics validation ledger](physics-validation-ledger.md). See the [validation methodology](methodology.md) for the status lifecycle and the [domain inventories](domain-inventories.md) for a claim-by-claim index. Design: [ADR-0018](../adr/0018-quantum-channeling-zone.md) and the [channeling radiation walkthrough](../research/physics/channeling-radiation-physics.md). These rows cover the static, scattering-free planar profile (#503). Its angles are plane-local; the laboratory orientation mapping waits on #25.

## `cr-debye-b-of-t`

- **Claim:** site Debye–Waller `B(T) = 8π²⟨u_x²⟩` is the linear interpolation of phonon-DOS tabulated rows inside their range. Outside the range it is the Debye model `B = (6h²/(m_u A k_B))(T/Θ²)[φ(Θ/T) + Θ/(4T)]`, with `φ(y) = (1/y)∫₀ʸ x/(eˣ−1) dx` and `Θ` fitted to the nearest tabulated end.
- **Code:** `materials/channeling/thermal.py::{debye_integral,debye_b_factor,fit_debye_temperature,ThermalSite.b_factor,thermal_b_factor}`; `data/channeling_thermal.toml`
- **Source:** Willis & Pryor (1975); Peng, Ren, Dudarev & Whelan, Acta Cryst. A52, 456 (1996), Eq. 8 and Table 1 ([doi:10.1107/S010876739600089X](https://doi.org/10.1107/S010876739600089X))
- **Status:** filtered
- **Checks:** Units: Å² from the CODATA prefactor 6h²/(m_u k_B) = 11 490 Å²·K. Peng quotes 11 492; `φ(y→0) = 1`; `φ(y) → π²/(6y)`; `φ(10)` matches Peng's tabulated 0.16444; Classical `B ∝ T`; T-independent zero-point limit; The fitted Θ_D reproduces its anchor exactly. Si Θ_D = 520 K, in line with Peng's high-temperature ≈500 K; Tabulated rows are reproduced exactly, and the model is continuous at both range ends.
- **Anchor:** `tests/materials/test_channeling_profile.py::{test_debye_integral_limits,test_debye_b_factor_classical_and_zero_point_limits,test_fit_debye_temperature_round_trips,test_thermal_rows_reproduced_and_extrapolation_continuous}`
- **Notes:** **Data.** Rows 170–280 K are transcribed from Peng Table 1. Rows below 170 K were ambiguous in the published layout and were not used; **Why interpolation.** A one-parameter Debye fit at 280 K is 11% low for Si at 180 K but only −1.6% for diamond, matching Peng's warning that the Debye model fails for Si. The model is therefore used only outside the tabulated range. Cryogenic Si values carry that model error; **Catalog discrepancy.** The PDOS diamond value (0.144 Å² at 293 K) disagrees with the catalog `B_ang2 = 0.21` used by PXR/CBS. That is a `debye-waller-catalog-provenance` follow-up, not changed here; Fresh-context rederivation pending.

## `cr-mott-bethe-factor`

- **Claim:** the electron–atom potential Fourier component uses the Mott–Bethe numerator `Z − f0(g)` over `g²` (Poisson's equation), with only the energy-independent X-ray form factor `f0` (Waasmaier–Kirfel, valid to `s ≤ 6 Å⁻¹`). The anomalous `f′, f″` are excluded.
- **Code:** `materials/channeling/potential.py::planar_profile`
- **Source:** Mott–Bethe formula, `f^e(s) = (Z − f_x(s))/(8π² a₀ s²)` (International Tables Vol. C, §4.3); Andersen et al., Phys. Scr. 28, 308 (1983)
- **Status:** filtered
- **Checks:** Harmonics are kept up to `s = g/(4π) ≤ 6 Å⁻¹`, and `g = 0` is excluded, so the `s → 0` limit is never evaluated; The diamond (220) coefficient equals `−U_g` exactly, since `U_g` uses bare `f0` for non-edge-prone elements.
- **Anchor:** `tests/materials/test_channeling_profile.py::test_planar_coefficient_is_negative_cbs_coupling`
- **Notes:** **Finding.** `materials.crystal.U_g` adds `f′` for `_EDGE_PRONE` elements (Si, P, Fe, …) even with `use_henke=False`. For Si (220) that moves the CBS potential coupling 3.7% from the electrostatic `Z − f0` value. That belongs to the CBS model (`cbs-amplitude`) and is reported for review there; No Doyle–Turner cross-check yet; the Si(110) line-energy anchor (`cr-transverse-eigenproblem`) agrees with Azadegan's Doyle–Turner potential to 0.6%; Fresh-context rederivation pending.

## `cr-planar-continuum-potential`

- **Claim:** the plane-averaged electron potential energy is `U(x) = Σ_{n≠0} U_n e^{i n g x}`, with `U_n = −(4π e²/(V g_n²)) Σ_j (Z_j − f0_j(g_n)) e^{i2π n hkl·R_j} e^{−B_j s_n²}`. The mean inner potential `U_0` is set to 0. The true period comes from the gcd of non-extinct harmonics. The nuclear and electron density coefficients use the same thermal smearing.
- **Code:** `materials/channeling/potential.py::{planar_profile,PlanarProfile.potential}`
- **Source:** Andersen, Bonderup, Lægsgaard & Sørensen, Phys. Scr. 28, 308 (1983); Azadegan, Comput. Phys. Commun. 184, 1064 (2013), Eqs. 4–5
- **Status:** filtered
- **Checks:** Units: eV from e²/(V g²); Electron wells sit on the atomic planes, consistent with the electron charge being −e; Si (110)/(100)/(111) periods equal d₂₂₀, d₄₀₀ and d₁₁₁. Doubled indices give the same profile; The centrosymmetric (110) coefficients are real; Mean densities are 8/V atoms and 112/V electrons; Heating monotonically reduces the well depth.
- **Anchor:** `tests/materials/test_channeling_profile.py::{test_extinctions_set_true_planar_period,test_si110_wells_sit_on_planes_with_literature_depth,test_density_means_and_heating}`
- **Notes:** Si(110) depth is 21.0 eV at 293.15 K. Literature reports ~22 eV, and Azadegan 2013 Fig. 1 shows ≈21 eV; Diamond(110) depth is 23.6 eV. The walkthrough's 25–30 eV estimate predates the PDOS `B`; Fresh-context rederivation pending.

## `cr-transverse-eigenproblem`

- **Claim:** transverse states solve `[−(ħ²/2γm) d²/dx² + U(x)] ψ = E_⊥ ψ`, expanded in Bloch plane waves, with `H_{mm′} = (ħ²/2γm)(k + m g)² δ_{mm′} + U_{m−m′}`. A state is bound when it lies below `max U`.
- **Code:** `materials/channeling/transverse.py::{transverse_states,TransverseStates.wavefunction}`
- **Source:** Andersen et al., Phys. Scr. 28, 308 (1983); Azadegan (2013), Eqs. 2–6
- **Status:** filtered
- **Checks:** Free limit: the plane-wave spectrum, with no bound states; Cosine (Mathieu) limit: spacings `ħω − (a g²/4)(n+1)` to 2×10⁻⁴; Parity alternates at `k = 0`; Basis convergence between M = 16 and 48; Bound count grows as √γ: 6 → 12 → 24 at 40 → 160 → 640 MeV.
- **Anchor:** `tests/materials/test_channeling_profile.py::{test_free_limit_has_no_bound_states,test_harmonic_limit_gives_equal_spacing,test_parity_alternates_and_selection_rule_emerges,test_bound_state_count_scales_as_sqrt_gamma,test_si110_40mev_matches_azadegan}`
- **Notes:** **Azadegan anchor (Si(110), 40 MeV).** Six bound states, matching the published six. `2γ²(ε₁−ε₀)` = 76.69 keV against the published 76.2071 keV at `k = 1.3089 Å⁻¹`, 0.6% apart. The remaining difference is consistent with Doyle–Turner factors and a different thermal amplitude; **Walkthrough correction.** Computed Si(110) bound-state counts are 1/2/3/5/9 at 1/3/10/30/100 MeV, below its WKB estimates (~2/4/7); Watson 1–3 MeV line anchors not yet pinned; Fresh-context rederivation pending.

## `cr-dipole-matrix-element`

- **Claim:** the transverse dipole matrix element between Bloch states at the same `k` is `x_fi = −i (ħ²/(γm)) Σ_m c_f*(k + m g) c_i / (E_f − E_i)`, from `[H, x] = −iħp/(γm)`. Parity selection emerges rather than being imposed.
- **Code:** `materials/channeling/transverse.py::TransverseStates.dipole_ang`
- **Source:** standard commutator identity; Andersen et al. (1983)
- **Status:** filtered
- **Checks:** Units: Å; Same-parity elements are below 10⁻¹² at `k = 0`; Agrees with direct `⟨ψ_f|x|ψ_i⟩` quadrature over one centred period to 2×10⁻³ for deep states.
- **Anchor:** `tests/materials/test_channeling_profile.py::{test_parity_alternates_and_selection_rule_emerges,test_dipole_matches_position_operator_for_deep_states}`
- **Notes:** Consumed by the `cr-dipole-rate` emission row (#505). Fresh-context rederivation pending.

## `cr-initial-occupation`

- **Claim:** an electron entering at angle `ψ` to the plane is the plane wave with `k_x = pψ/ħ`. Its populations are `P_n = |c_n(m₀)|²`, where `k + m₀ g = k_x`, which sums to 1. A Gaussian divergence is averaged by Gauss–Hermite quadrature. Surface refraction is not applied.
- **Code:** `materials/channeling/transverse.py::entry_populations`
- **Source:** Andersen et al. (1983); Azadegan (2013), Eq. 17
- **Status:** filtered
- **Checks:** Unitarity to 10⁻¹²; Even states only at `ψ = 0`; `P(ψ) = P(−ψ)`; The divergence average stays normalized and mixes parities.
- **Anchor:** `tests/materials/test_channeling_profile.py::{test_entry_populations_unitary_and_parity_selective,test_divergence_average_is_normalized_and_mixes_parity}`
- **Notes:** At 40 MeV Si(110), `P₀(0) = 0.24`, consistent with Azadegan 2013 Fig. 3; Surface refraction by the mean inner potential depends on the face/lattice geometry. It is deferred to the orientation coupling (#25), as is the laboratory-frame incidence angle; Fresh-context rederivation pending.

## `cr-doppler-line-energy`

- **Claim:** a transverse transition emits at `ħω = ΔE_⊥/(1 − β cos θ)`, evaluated as `1/(γ²(1+β)) + 2β sin²(θ/2)`, where `θ` is the photon angle to the electron velocity. Photon recoil `O(ħω/E)` is neglected.
- **Code:** `materials/channeling/transverse.py::{doppler_denominator,line_energy_eV}`
- **Source:** relativistic kinematics; Andersen et al. (1983)
- **Status:** filtered
- **Checks:** Forward limit is `γ²(1+β)ΔE`; Agrees with the naive form at moderate angles; Cross-module: equals the PXR/CBS vacuum resonance denominator of `materials/emission.py::photon_energies` to 10⁻¹², at 5 MeV/3°, 5 MeV/20° and 30 keV/120°.
- **Anchor:** `tests/materials/test_channeling_profile.py::{test_forward_line_energy_is_doppler_upshift,test_doppler_denominator_matches_pxr_resonance}`
- **Notes:** Shares the `line-energy-dispersion` convention: `θ` relative to `v`, positive frequency. Fresh-context rederivation pending.
