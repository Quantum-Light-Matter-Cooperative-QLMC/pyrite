# Line recoil inside a crystal: the #527 gate

**Status:** Research note for [#527](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/527), 11 November 2026, branched from `563d17906`. It answers the gate #249 placed before any A/B/C work. It adds pinned data, a regression anchor and a check script, and changes no production model or default output. The claim `line-recoil-in-crystal` is ledgered as `anchored` after fresh-context verification (`rederived`, with corrections applied).

#249 added the exact vacuum PXR/CBS root (`line-recoil-resonance`). Its shift from the classical root is percent-level for 10–100 keV lines (`line-recoil-validity-screen`). #249 left two questions open:

1. Is the free-electron vacuum shift observed in a crystal?
2. Does it survive for a Bloch electron in a finite slab, with mean inner potential and refraction?

**Answer: yes to both, at the precision that matters for the main band.**

- Huang et al. (2023) measured graphite and hBN (002)/(004) lines from 10.7–14.5 keV electrons. Their fit-independent harmonic defect $2E_2-E_4$ is $18.19\pm0.66$ eV. The straight exact vacuum root predicts $19.10$ eV, and the classical root predicts exactly $0$.
- The in-medium corrections the vacuum root omits move that defect by at most 0.6 eV and single lines by at most 0.7 eV, against shifts of 10–45 eV. Bloch-branch shifts broaden rather than move the line (§2).

Option A (recoil-aware production line energies) is therefore justified independently of B and C.

## 1. Evidence

### 1.1 Observable

The classical root $\omega_c=\beta c\,\hat u\cdot\mathbf g/(1-\beta\hat u\cdot\hat n)$ is linear in $\mathbf g$. For collinear harmonics $\mathbf g_m=m\mathbf g_1$, the defect $2\omega_2-\omega_4$ is therefore zero for every tilt, energy and observation angle. The exact root,

$$
\hbar\omega=\frac{\beta\,\hat u\cdot\mathbf Q-Q^2/2E}{1-\beta\,\hat u\cdot\hat n+\mathbf Q\cdot\hat n/E},\qquad \mathbf Q=\hbar c\,\mathbf g
$$

(PyRITE sign, `line-recoil-resonance`), carries the $Q^2$ term that makes the defect nonzero.

An energy-dispersive (EDS) calibration $E_{\rm meas}=a+bE$ maps the defect to $a+b(2E_2-E_4)$. A gain error therefore cannot fake the defect, and an offset enters only additively. Huang's calibration against C–Si Kα lines bounds the offset to ±0.5 eV (SI §S5). The tilts in Table S4 were fitted with the paper's quantum model, so absolute line energies are not an independent test; the defect is.

### 1.2 Comparison

Source: Huang et al., Nat. Photon. 17, 224 (2023), DOI `10.1038/s41566-022-01132-6`, SI Table S4. Measurement angles are $\theta_{\rm obs}=130^\circ$ (JXA-8530F) and $119^\circ$ (JSM-7800F Prime), from the Methods. The 13 rows are pinned in `tests/data/recoil/huang-2023-table-s4.json`, transcribed and checked against the PDF page.

| Set | $T$ (keV) | $2E_2-E_4$ measured (eV) | exact vacuum root (eV) |
|---|---|---|---|
| 85 nm graphite, 130° | 10.7–13.0 | 16.3–20.8 | 18.4–19.7 |
| 55 nm graphite, 119° | 13.5–14.5 | 16.9–19.7 | 17.7–18.7 |
| 95 nm hBN, 119° | 13.5–14.5 | 14.9–19.6 | 19.6–20.2 |
| **All 13** | | **18.19** (s.d. 1.66) | **19.10** |

- The mean residual is $-0.91\pm0.66$ eV. The uncertainty combines the standard error with the calibration offset in quadrature, so the residual sits at 1.4σ.
- The classical defect of zero is 27σ away.
- PyRITE's straight, single-electron root reproduces the defect without Huang's Monte Carlo transport. Huang's own scattering Monte Carlo, read from their Fig. 1d, predicts ≈15.7 eV (quantum) and ≈−2.5 eV (classical), so it moves the defect by about 3 eV. The straight-path agreement therefore does not test their transport or peak-fit bias; the classical exclusion survives either way.
- The ±0.66 eV uncertainty assumes one calibration offset common to both lines. If the ±0.5 eV error is uncorrelated between the lines, classical exclusion drops to ≥ 11σ.
- The detector azimuth is not stated. The pin uses $\phi_{\rm obs}=0$ (Fig. 1c inset); 180° (implied by Fig. S15) predicts 18.04 eV. Both support the conclusion.
- The absolute lines from the root sit 2–6 eV above the measured lines at the fitted tilts. The classical lines sit 12–46 eV above.

Run `uv run python checks/recoil_in_crystal_527.py` for the full table. The anchor test is `tests/materials/test_recoil.py::test_huang_2023_harmonic_defect_matches_in_crystal_lines`.

**Other data.**
- Shentcis et al., Nat. Photon. 14, 686 (2020), used 60–300 keV electrons on WSe₂ along [001], with the detector 121° from the beam. They reported agreement with the classical root without fitting. The exact root shifts their (002) line by −1.6 eV at 60 keV and −0.7 eV at 300 keV, against an EDS resolution of about 80 eV. Their result is therefore consistent with recoil and cannot test it.
- Shi et al. (2023, Light Sci. Appl. 12, 148) and Balanov et al. (2024) give no measured low-energy line positions.
- Shi et al. (2024, Nat. Commun. 15, 7803) remains the B-option benchmark (30 nm Si, 30/300 keV). It does not bear on this gate.

## 2. Why the vacuum root survives in a crystal

**Conservation.** The emitter is the electron plus a static lattice.
- The lattice absorbs $\hbar\mathbf g$ without energy: a zero-phonon (Debye–Waller) transfer to an effectively infinite mass.
- Phonon-assisted transfers spread into a diffuse background, not a shifted line.
- The electron's initial and final states are Bloch waves. The final branch is the free-like one whose dominant plane wave is $\mathbf k_f=\mathbf k_i-\mathbf g-\mathbf q$. Away from electron Bragg conditions its energy is the free relativistic mass shell plus the constant mean inner potential plus second-order Bloch corrections.
- Energy conservation between these branches is exactly the vacuum root's mass-shell statement. The recoil term is the curvature of the electron's own dispersion, not a lattice recoil, so a crystal cannot remove it.

**Finite slab.** A slab breaks momentum conservation only along its normal. The finite-duration integral (`line-recoil-formation`) gives a $\mathrm{sinc}^2$ profile centred on the exact root, with width ~$2\pi\beta c/L$ over the free path between scatterings. Huang's intrinsic width is about 20 eV. The recoil is a centre shift and survives; the slab broadens the line around it.

**Corrections at the Huang geometries** (graphite 11.7 keV, 130°; hBN 14.0 keV, 119°):

| Correction | Single-line shift | Change of $2E_2-E_4$ |
|---|---|---|
| Mean inner potential, $U_0=-13.4$ eV (graphite) and $-13.1$ eV (hBN) from `cr-mean-inner-potential`, applied to both states | +0.31 to +0.66 eV | ≤ 0.003 eV (cancels: linear in $g$) |
| Photon refraction with exit-face tangential continuity ($\mathbf k_{\rm eff}=\omega(\hat n+\delta\nabla L)$, `xray-in-medium-resonance`), $\delta=1-\mathrm{Re}\,n$ from `refractive_index` (2–11 × 10⁻⁴); the backward photon exits the entrance face about 62° from its normal | +0.15 to +0.29 eV | +0.25 to +0.58 eV over all rows and both azimuth conventions |
| Electron refraction at the (001) face (≤ 130 µrad bend toward the normal) | ≤ 0.06 eV | < 0.001 eV |
| Graphite Bloch dispersion of the free-like branch; per-coupling two-beam shift, Mott–Bethe $U_h$ (largest 9.8 eV, (002)) | azimuth-dependent; +0.10/+0.33 eV at one arbitrary azimuth | median ≈0.4–0.5 eV over azimuth, several eV near a strong Bragg condition; centroid unshifted (see below) |

The non-Bloch corrections are at most 0.7 eV on a line and 0.6 eV on the defect. The recoil shifts they would compete with are 9–45 eV.

**Bloch branches.** Huang does not report the crystal's in-plane rotation about [001], and the free-like branch shift depends on it.
- The check evaluates one arbitrary azimuth. There the initial state lies 0.46 eV from the $(\bar3\bar1\bar3)$ condition, where $U_h=0.35$ eV, so each coupling uses the exact two-beam shift $-d/2+\operatorname{sign}(d)\sqrt{d^2/4+U_h^2}$.
- The fresh-context verification scanned all azimuths. The defect shift of the free-like branch has a median of about 0.4–0.5 eV and reaches 6–7 eV when a strong low-order reflection sits at Bragg.
- What justifies neglecting it is a sum rule. The incident plane wave projects onto the two-beam branches with weights $\lvert C_{0\alpha}\rvert^2$, and the excitation-weighted branch shift $\sum_\alpha\lvert C_{0\alpha}\rvert^2\lambda_\alpha$ is zero at two-beam order. At EDS resolution, branch splitting therefore broadens the line rather than moving its centroid. A many-beam calculation is outside this note.

## 3. Consequences for PyRITE

- The production root `montecarlo/spectrum/cxr/_kernels.py::_in_medium_kinematics` is classical. It solves `omega (1 - v.n - delta v.grad L) = v.g`. Its numerator lacks $-Q^2/2E$ and its denominator lacks $\mathbf Q\cdot\hat n/E$.
- The `recoil_shift_exceeds_limit` count (`line-data-coverage.v2`) is therefore not only a screen. On the evidence above, a counted line is a real percent-level error in line position.
- An example is HOPG (004) at 30 keV: about 47 eV at 1.43 keV. That exceeds EDS resolution at 10 keV and is resolvable by wavelength-dispersive or grating spectrometers.

## 4. Recommendation for option A

**Implement.** Open a production issue for recoil-exact line kinematics that is independent of B and C. Scope:

1. Replace the classical numerator and denominator in the first-order in-medium root with the exact vacuum root's. Keep the existing Snell decrement term, which the corrections above show is additive at this order. Evaluate it in float64 per row with the cancellation-free Doppler factor (`line-kinematics-precision`).
2. Make the width and Jacobian consistent: the mismatch slope becomes $(1-\beta_f\hat u_f\cdot\hat n)/\hbar c$ (`line-recoil-formation`). The incoherent $\mathrm{sinc}^2$ route and the phased-field routes must use the same root.
3. Update the CBS phase in the same pass, or explicitly scope it out, so that PXR and CBS stay on one resonance.
4. Bump `CXR_ESCAPE_MODEL`, which already versions line kinematics (v5, #439), and re-pin every identity digest derived from it.
5. Acceptance:
   - the Huang defect (this anchor) reproduced through `mc_cxr_spectrum` on straight segments;
   - bitwise-classical limit at $\hbar g\to0$;
   - MeV lines unchanged within the current tolerance;
   - the `recoil_shift_exceeds_limit` count retired or turned into a regime warning for roots beyond the first-order Snell validity.

B (transverse-recoil-resolved spectra) and C (shaped superpositions) remain open in #527. This gate only removes the doubt about whether the vacuum kinematics apply in-crystal.

## 5. Limits of this note

- There is one experimental group and two vdW materials. The fitted tilts make absolute energies circular, and only the defect is independent.
- The defect test rests on the stated ±0.5 eV calibration offset. A hidden 18 eV offset would mimic it, which the C–Si Kα calibration excludes.
- No measured line positions above 15 keV with resolution better than the shift exist in the library. The 30–300 keV check rests on the conservation argument and the Shentcis consistency.
- The Bloch argument rests on the two-beam centroid sum rule. It is not a many-beam dynamical-diffraction calculation, and Gaussian peak fits of a split, scattered line can still be biased (Huang's own Monte Carlo moves the defect by about 3 eV).
