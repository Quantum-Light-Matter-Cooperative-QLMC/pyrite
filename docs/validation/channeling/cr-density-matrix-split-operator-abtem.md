# `cr-density-matrix-split-operator`: abTEM frozen-phonon oracle

Evidence for issue #504. This page compares band populations from an independent wave-optics simulation with the density-matrix solver running the TDS kernel alone. abTEM has no electronic excitation.

- **Harness:** `checks/abtem_channeling_oracle.py`.
- **Case:** Si(110), 16.92 MeV, normal incidence, 293.15 K, 1 µm depth.
- **Record:** `docs/validation/check-records/abtem_channeling_oracle.jsonl` (verdict `pass`, revision `be673d11`).

## Oracle

abTEM 1.0.10 runs multislice with the Lobato potential on a GPU via CuPy, with populations within 3e-4 of the CPU run.

- **Crystal:** diamond-structure Si oriented so the (110) planes contain the beam direction `[1, −1, n]`.
- **Thermal vibrations:** each frozen configuration displaces every atom by an isotropic Gaussian with `a² = B/8π²` per axis (the Einstein model).
- **Supercell:** 8 true periods in `x`, so its plane-wave momenta are the solver's Bloch grid.
- **Projection:** at each exit plane the wave is projected onto PyRITE's Bloch states, `P_n(z) = ⟨Σ_r |⟨r n|ψ(·, y)⟩|²⟩_y`, averaged over 16 configurations.
- **Interaction constant:** abTEM's `σ = 2π m λ e/h²` equals `e/(ħv)` to 8e-9 at 0.2–54.5 MeV.
- **Potential:** Lobato and Waasmaier–Kirfel Mott–Bethe `f^e` agree to 1e-3 for Si.

The solver runs with `tds_kernel` (`n_cells = 8`), with no sink (margin ∞) and `dz = 25 Å`.

## Model and convergence checks

Max |ΔP_n| over bands 0–3 and depths 0–1 µm:

| Variant | vs PyRITE | Notes |
|---|---|---|
| Random-string planes (atoms uniform in `y`, `z`) | 0.109 | Amorphous in-plane: static-disorder scattering at `q_y ≠ 0` that a crystal diffracts coherently. Rejected as an oracle. |
| Crystal `n = 7` | 0.179 | `[1 −1 7]` is close enough to a zone axis that the atom strings along the beam act coherently, as in axial channeling. Not planar. |
| Crystal `n = 11`, sampling 0.02 Å | 0.0057 | Planar regime. |
| Crystal `n = 15`, sampling 0.02 Å | 0.0064 | Within 0.0017 of `n = 11`, so converged in beam direction. |
| Crystal `n = 11`, slice 0.5 Å (vs 1 Å, paired seeds) | — | Changes populations by ≤ 0.0017. |
| **Crystal `n = 11`, sampling 0.0125 Å** | **0.0021** | **Anchor** (8 configurations). |
| PyRITE band `H` = 24 / 32 / 40 | — | Identical to 1e-4 over 1 µm. |

The residual is abTEM's grid. At 0.02 Å sampling its potential holds no Fourier components above ~157 Å⁻¹, and the antialias aperture removes scattering beyond ~105 Å⁻¹. Most of the large-`q` TDS tail acts on the ground band, which is centred on the nuclei. The ground-band gap at 1 µm falls from 0.0054 to 0.0019 when sampling goes from 0.02 to 0.0125 Å. The ratio, 0.35, matches the `(0.0125/0.02)² = 0.39` expected for truncating a `q⁻²` tail.

## Anchor

| z [Å] | abTEM bands 0–3 (± SEM) | PyRITE bands 0–3 |
|---|---|---|
| 0 | 0.3211 / 0 / 0.4462 / 0 | 0.3211 / 0 / 0.4462 / 0 |
| 5000 | 0.1813 / 0.0512 / 0.3719 / 0.0150 (± 0.0005 / 0.0002 / 0.0002 / 0.0001) | 0.1808 / 0.0515 / 0.3708 / 0.0150 |
| 10000 | 0.1161 / 0.0675 / 0.3142 / 0.0250 (± 0.0005 / 0.0006 / 0.0004 / 0.0002) | 0.1142 / 0.0675 / 0.3134 / 0.0251 |

The tolerance is 0.004 absolute over 1 µm, the sum of the oracle's systematic budget: grid ≈ 0.002, slicing ≤ 0.0017, statistics ≈ 0.0006. The measured value is 0.0021, so the check passes.

## Scope

- **Thickness:** the 1 µm thickness is far below the energy-loss scale (`ΔE/E ≈ 2e-5` at 17 MeV).
- **Not tested here:** electronic excitation, the dechanneling sink and energy loss. They are covered by `cr-dechanneling-collapse` and the solver's own tests.
- **Untested regimes:** other planes, other energies and diamond.
