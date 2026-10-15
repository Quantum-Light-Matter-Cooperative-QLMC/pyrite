# `cr-dechanneling-collapse`: Kephart occupation lengths and cutoff sensitivity

Evidence for issue #504. It is not a verdict: the ledger row stays `unverified` until the comparison basis is decided (see [Open decisions](#open-decisions)).

- **Harness:** `checks/kephart_occupation.py` at revisions `ebea431c` (remote sweep on `qlmc`, SLURM CPU) and `42b93283` (local `H ≥ 40` runs). The solver code is the same in both.
- **Reference:** Kephart et al., Phys. Rev. B 40, 4249 (1989). Table I gives 1/e occupation lengths ±20%; Table II gives the inferred divergences of 2.2 mrad (17 MeV) and 1.0 mrad (54 MeV).
- **Assumed temperature:** 293.15 K, which the paper does not state.

## Method

1. Run `DensityMatrixSolver` with the TDS and electronic kernels from normal incidence with Gaussian divergence.
2. Weight the band populations as the measured line (17 MeV) or as the dipole-weighted window (54 MeV).
3. Integrate them into a yield `Y(Z)`.
4. Fit `A(1 − e^{−Z/L})` on 24 log-spaced thicknesses over 2.8–120 µm (17 MeV) or 2.8–500 µm (54 MeV), the range where the paper's exponential fit holds.

The collapse margin sets the subspace cutoff `barrier + margin`; a margin of ∞ leaves the plane-wave band edge as the only sink.

## Cutoff sensitivity

Occupation length `L` in µm, with the ratio to Kephart in parentheses. Settings: `H = 32` (17 MeV) or `H = 40` (54 MeV), `n_cells = 16`, `dz = 31.25 Å`.

| margin [eV] | Si(110) 17 MeV n=1 (20) | n=2 (17) | n=3 (17) | Si(100) 17 MeV (16) | Si(100) 54 MeV (24) | Si(110) 54 MeV (36) |
|---|---|---|---|---|---|---|
| 0 | 1.8 (0.09) | 2.0 (0.12) | 2.0 (0.12) | 1.6 (0.10) | — | — |
| 30 | 5.2 (0.26) | 5.4 (0.32) | 5.6 (0.33) | 5.4 (0.34) | — | — |
| 100 | 9.9 (0.50) | 10.1 (0.59) | 10.2 (0.60) | 10.1 (0.63) | 21.0 (0.88) | 20.8 (0.58) |
| 300 | 19.0 (0.95) | 18.8 (1.11) | 18.7 (1.10) | 18.9 (1.18) | 42.3 (1.76) | 42.3 (1.18) |
| ∞ | 37.1 (1.85) | 36.0 (2.12) | 35.2 (2.07) | 41.6 (2.60) | — | 64.4 (1.79), `H = 40`, not converged in `H` |

The relative RMS residual of the integrated-exponential fit is 0.1–1% for finite margins and 2–3% for ∞. With no near-barrier sink the population decay is visibly non-exponential. The paper reports the same for electrons beyond 1 mm and also fits a multiple-scattering `Z^{1/2}` form.

## Convergence (Si(110), 17 MeV, margin ∞, n=1)

- **Depth step** (`H = 24`): `dz` = 250 / 125 / 62.5 / 31.25 Å gives `L` = 24.5 / 27.1 / 28.4 / 29.1 µm. This converges at second order with a large constant; the balance defect at 31.25 Å is −9e-3.
- **Bloch grid:** `n_cells = 32` reproduces `n_cells = 16` to 0.1 µm (`H = 32`, `dz = 31.25 Å`).
- **Plane-wave band** (`dz = 62.5 Å`): `H` = 24 / 32 / 40 / 48 / 56 / 64 gives `L` = 28.4 / 36.2 / 40.9 / 43.6 / 44.9 / 45.6 µm. The increments shrink from 7.8 to 0.7 µm, so the no-sink limit is about 46–47 µm, about 2.3× Kephart.

Entry populations of bands 1–3 at 17 MeV (110) are 6.40 / 6.40 / 6.44%, against Kephart's Table II theory of 6.5% each and measured 5.5 / 5.6 / 6.3%.

## Detector observable (maintainer-chosen comparison basis)

Kephart's detector sits on the beam axis. `checks/kephart_occupation.py` weights each band signal by `W(z)`, the detector's acceptance of the electrons' `y`-angle spread:

- The boosted-dipole photon density is `1/(1 + γ²θ²)²` at photon angle `θ` perpendicular to the `x` oscillation.
- `θ` starts with the beam divergence. Its characteristic function then decays as `exp(−z Λ(s))`, with `Λ(s) = Γ(Δ = s/p)`, the validated decoherence rate of a planar-uniform density.
- The scattering rate is taken at the mean density (×1) and bracketed at ×4, the bound bands' TDS rate on the planes.
- At 17 MeV, `W` is 0.94 at 120 µm (×1). The incoherent `y`-angle RMS is about 5 mrad, against the Highland 21 mrad of random-direction electrons. That fits Kephart's observation that multiple scattering has no perceptible influence on electron CR spectra.

`L` in µm with the detector observable (×1 / ×4). Settings: `H = 56`, `n_cells = 16`, `dz = 62.5 Å`; 54 MeV pending.

| margin [eV] | Si(110) 17 MeV n=1 (20) | n=2 (17) | n=3 (17) | Si(100) 17 MeV (16) |
|---|---|---|---|---|
| 300 | 17.0 / 14.6 | 16.9 / 14.5 | 16.8 / 14.5 | 17.0 / 14.6 |
| ∞ | 35.5 / 26.7 | 34.3 / 25.8 | 33.4 / 25.2 | 34.1 / 25.7 |

Populations alone at the same settings give 18.7 / 18.6 / 18.5 / 18.6 µm (300 eV) and 44.9 / 43.2 / 41.9 / 43.2 µm (∞). With the detector observable and a 300 eV cutoff, every 17 MeV value is within ±20% at ×1. At ×4 all are within ±20% except Si(110) n=1 (0.73). Without a near-barrier cutoff, the detector observable still leaves `L` 1.3–2.1× too long.

## Interpretation

- **The occupation length is not set by the kernel alone.** It depends on where electrons stop counting as channeled. Planar rechanneling is a one-dimensional random walk in transverse angle, which keeps returning to the bound band. With a band-edge sink only, `L` converges in `H`, but to roughly twice the measured value.
- **A fixed 300 eV cutoff matches all four 17 MeV measurements within 20%,** at ratios 0.95–1.18. It overshoots at 54 MeV (ratios 1.18 and 1.76). A 100 eV cutoff matches Si(100) at 54 MeV (0.88) but not Si(110) (0.58). No single energy cutoff fits both energies. In angle, the 17 MeV match corresponds to about 5.9 mrad (about 3.8 critical angles).
- **The solver traces over `q_y`, but the experiment does not.** Kephart detected forward photons, so electrons scattered along the planes beyond about `1/γ` (29 mrad at 17 MeV, 9 mrad at 54 MeV) radiate outside the acceptance. That reduces the measured yield with depth in a way the transverse-state population alone does not.

## Open decisions

- **Decided.** On 2026-10-15 the maintainer chose the detector observable as the comparison basis, over a pure dechanneling criterion or a bare calibration factor.
- **Still open.** The detector observable does not by itself reproduce Kephart without a near-barrier dechanneling cutoff. A 300 eV cutoff matches at 17 MeV. Whether one cutoff can be declared as a calibration across both energies depends on the pending 54 MeV runs. Without the cutoff, 54 MeV at `H = 64` gives 2–3× Kephart, and those runs are converged in neither `dz` nor `H`. A declared cutoff would enter model identity and provenance.
