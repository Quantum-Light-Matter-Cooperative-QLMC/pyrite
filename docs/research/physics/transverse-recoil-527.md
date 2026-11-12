# Transverse recoil for unshaped electrons: #527 option B

**Status:** Research note for [#527](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/527), 11 November 2026. It follows the [in-crystal recoil gate](quantum-recoil-in-crystal-527.md). It changes no production model. Line positions come from `materials/recoil.py::recoil_resonance_eV` and reuse `line-recoil-resonance`. The intensity prototype and observability table (§5) come from `checks/transverse_recoil_527.py` and are ledgered as `quantum-recoil-observability`.

**Finding.** For a plane-wave (unshaped) electron, the transverse-recoil splitting of Shi et al. (2024) is the exact recoil root evaluated separately for each reciprocal vector. The $Q^2=\hbar^2c^2(g_\perp^2+g_z^2)$ term in its numerator carries the $g_\perp$ dependence. The classical root depends only on $\hat u\cdot\mathbf g$, so it merges all reflections with the same $g_z$ into one line; the exact root separates them.

Pathways to different final transverse momenta $\mathbf q_i+\mathbf g_\perp$ are distinguishable and add incoherently. That is the same incoherent per-reflection sum that the classical unit-cell average of the impact parameter produces. For unshaped electrons, option B therefore reduces to option A ([#1099](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/1099)) applied per reflection, with two conditions:

- every symmetry member that shares $g_z$ is scored as its own row;
- the recoil-consistent Jacobian is applied.

Interference between pathways needs a shaped initial state ($\mathbf q_i+\mathbf g=\mathbf q_j+\mathbf h$ with $\mathbf q_i\ne\mathbf q_j$); that is option C.

## 1. Source

Shi, Wong, Huang, Wong & Kaminer, *Transverse recoil imprinted on free-electron radiation*, Nat. Commun. 15, 7803 (2024), DOI `10.1038/s41467-024-52050-w` (Zotero `Q4I3JISU`; main text and Supplementary Information, attachment `6YMV48GX`, were read).

The setup is Si along [001] at normal incidence, a 30 nm film, and a plane-wave electron. The photon lies in the x–z plane at θ = 100° (Fig. 2b, d, f). Methods Eq. (6) states exact energy–momentum conservation with the lattice supplying $\mathbf g$. Eq. (7) is its small-recoil expansion, and Eq. (9) splits the result into longitudinal and transverse terms. The text states that $g_\perp$ and $g_\parallel$ contribute "quantum corrections on the same order". Eq. (3) adds pathway amplitudes coherently only through the initial-state superposition $\varphi(\mathbf q)$; for a single plane wave, every final electron state is reached by one $\mathbf g$.

## 2. Line positions against Fig. 2

All roots use the PyRITE sign, $\mathbf g=-(n_x,n_y,n_z)\,2\pi/a$ with $a=5.431$ Å. Only diamond-cubic-allowed reflections are included.

| $T$ | Classical $n_z=-2$ / $-3$ (eV) | Fig. 2 CL peaks (read off the figure) | Strongest exact lines, $n_z=-2$ (eV) |
|---|---|---|---|
| 30 keV | 1418.4 / 2127.6 | ≈1.42 / ≈2.13 keV | (2,0) 1394.9; (0,±2) 1383.9; (−2,0) 1373.1; then down to ≈1.0 keV for larger $\lvert g_\perp\rvert$ |
| 300 keV | 3124.2 / 4686.3 | ≈3.12 / ≈4.68 keV | (2,0) 3119.5; (0,±2) 3104.2; (−2,0), (4,±2) 3089.1 |
| 5 MeV | 3876.0 / 5814.0 | ≈3.87 / ≈5.81 keV | (2,0) 3875.9; (0,±2) 3873.2 (converging on CL) |

- The classical roots coincide with the CL peaks.
- The tallest QM peak in Fig. 2d and 2f sits at the $(0,\pm2,2)$ root (pixel reads 1383.5 / 3105.4 eV against 1383.9 / 3104.2 eV).
- The QM comb extends down to ≈1.0 keV at 30 keV, as the larger-$\lvert g_\perp\rvert$ roots do.
- The splits collapse with energy, as Fig. 2a and 2b show.

These are visual readings of a log/linear figure. The data are not digitized, so the comparison supports the identification but is not an anchor.

## 3. Observability

### 3.1 Fig. 2 geometry (30 nm Si, θ = 100°)

These are closed-form estimates, with derivatives taken from the exact root:

| Quantity | 30 keV | 300 keV |
|---|---|---|
| Adjacent $n_z=-2$ split | ≈11 eV | ≈15 eV |
| Finite-thickness FWHM, $0.886\cdot2\pi\,\hbar c\beta/[L(1-\beta\cos\theta)]$ | 11.4 eV | 25.1 eV |
| Electron tilt sensitivity, $\lvert\partial E/\partial\theta_e\rvert$ | 0–1.8 eV/mrad | 1–5 eV/mrad |
| Detector-angle sensitivity, $\lvert\partial E/\partial\theta_{\rm obs}\rvert$ | 0.4 eV/mrad | 2.1 eV/mrad |
| Elastic mean free path in Si (screened Rutherford) | ≈35 nm (SI §S8: 40 nm) | ≈250 nm |

At large angles, adjacent split lines are about one thickness-limited width apart at 30 keV and below one width at 300 keV. The signature there is the shift plus an asymmetric comb toward low energy. At 30 keV about half the electrons scatter elastically within the film, and their lines move by tens of eV. SI §S10 agrees that large angles need high spectral and angular resolution, and names this as the reason earlier experiments saw no splits.

### 3.2 Forward geometry proposed in SI §S10 (100 keV, 100 nm Si, θ = 0–5°)

Near the forward direction, $1-\beta\cos\theta$ amplifies the $\lvert g_\perp\rvert^2$ term. Exact roots for $n_z=-1$ are:
- **θ = 0:** 2719 eV for $(\bar1,\pm1)$, then 2645, 2570, 2495 eV, against a classical value of 2770 eV;
- **θ = 2.5°:** 2717/2715, then 2644–2639 eV.

| Quantity | 100 keV, 100 nm, θ = 0–5° |
|---|---|
| Spacing between $\lvert g_\perp\rvert^2$ shells | ≈70–75 eV ($n_z=-1$), ≈150 eV ($n_z=-2$) |
| Spread within a shell across θ = 0–5° | ≈2–12 eV |
| Finite-thickness FWHM | ≈13 eV |
| Shift of the first shell from classical | ≈50 eV ($n_z=-1$) to ≈500 eV ($n_z=-4$) |
| Electron tilt sensitivity | ≈$\omega(g_\perp/g_z)\,\delta$: ≈4 eV/mrad for $(\bar1,\pm1,-1)$ |

On unscattered segments the shells are resolvable with a ~70 eV EDS, in agreement with SI §S10 (after transport the leading shell becomes a shoulder, §5.2) and Supplementary Fig. 8. The binding constraints are electron direction and access:
- the beam emittance must be small (SI cites < 10 nm·mrad at 200 keV);
- the detector must sit downstream of a deflected beam;
- elastic scattering (≈1 mean free path in 100 nm at 100 keV, characteristic angles of tens of mrad) smears scattered electrons' lines by ~100 eV, so only unscattered segments keep the shells sharp;
- forward CBS, bremsstrahlung and characteristic background must be part of the comparison.

### 3.3 Transverse coherence

SI §S6 (Supplementary Fig. 4) finds the plane-wave result unchanged for a transverse coherence of $\sigma_t$ = 1 nm, larger than the cell, and smoothed into fewer peaks at 0.1 nm. An ordinary beam therefore needs a coherence width of at least a unit cell. A sub-cell focused probe washes the splitting out. Real-space beam shaping belongs to option C.

## 4. Consequences and recommendations

- **B for unshaped electrons: implement through #1099, not separately.** #1099 must:
  - score every symmetry member of each pinned family as its own row. Catalogue `hkl_list` pins one representative and its Friedel mate, whereas `dominant_reflections` expands all members.
  - apply the recoil-consistent Jacobian.
  - add a Shi-geometry regression: 30 nm Si [001], 30/300 keV, θ = 100°. It would check line positions per $g$ against the exact root, the CL limit, and the collapse at 5 MeV.
- **Intensities and polarization: prototype done, sub-MeV weights open (§5.1).** PyRITE's classical per-reflection coupling matches Shi at 5 MeV but not at 30–300 keV. Per-line linear polarization follows from separating the lines; it needs no new physics.
- **Observability: table in §5.2.** Backgrounds, divergence and direction-dependent shifts on scattered flights remain for a production run after #1099.
- **C (shaped superpositions)** is unchanged by this note and remains a separate decision.

## 5. Prototype and observability table

`checks/transverse_recoil_527.py` computes PyRITE's classical PXR one reflection at a time (`mc_cxr_spectrum` under the ħ → 0 recoil hook, since production lines sit on the exact root after #1099; Si B = 0.4632 Å², HOPG B = 0.8 Å²). It then shifts each reflection's spectrum rigidly from its classical to its exact root.
- The "QM" curve is a prototype. It keeps PyRITE's classical per-reflection weight and line shape; it does not use Shi's quantum matrix element or the recoil Jacobian.
- Rows cover $\lvert h\rvert,\lvert k\rvert\le 8$ for each $n_z$ (7 for the forward case).

### 5.1 Against Shi Fig. 2 (straight 30 nm Si, θ = 100°)

Shi heights below are the original visual reads of Fig. 2b/d/f (about ±10 %). The vector-digitized values (`tests/data/recoil/shi-2024-fig2.json`, #1103) agree within 3 %; the check now reads them.

| $T$ | $n_z$ | PyRITE CL peak (photons/eV/sr/e) | Shi CL | ratio | prototype QM/CL height |
|---|---|---|---|---|---|
| 30 keV | −2 / −3 | 2.13e-9 / 1.22e-9 | 1.15e-9 / 1.2e-9 | 1.85 / 1.0 | 0.36 / 0.18 |
| 300 keV | −2 / −3 | 1.48e-10 / 6.7e-11 | 5.8e-11 / 5.5e-11 | 2.55 / 1.21 | 0.41 / 0.29 |
| 5 MeV | −2 / −3 | 4.21e-11 / 3.62e-11 | 4.4e-11 / 3.8e-11 | 0.96 / 0.95 | 0.92 / 0.94 |

- The classical peaks sit within 1 eV of the classical roots at every energy.
- **5 MeV:** the classical heights agree, and the split collapses as in Fig. 2b.
- **30–300 keV:** PyRITE's classical $n_z=-2$ peak is about 2× Shi's, and the weights are spread differently.
  - PyRITE's largest single row is $(0,\pm2,2)$, holding 18 % of the CL peak per member. Its exact root is 1383.9 eV at 30 keV and 3104.3 eV at 300 keV.
  - In Shi Fig. 2d/2f the tallest QM peak sits at the same $(0,\pm2,2)$ root (pixel reads 1383.5 / 3105.4 eV), so peak positions agree, but with about 0.9× the CL height.
  - PyRITE's tallest prototype peak is 0.36–0.41× CL, but its absolute height is close to Shi's tallest QM peak: 7.6e-10 against ≈1.05e-9 at 30 keV, and 6.1e-11 against ≈5.3e-11 at 300 keV.
  - Shi's QM area in Fig. 2f visibly exceeds its CL area, whereas the prototype conserves area by construction.

**Resolved in #1103** ([attribution note](shi-2024-intensity-1103.md)). Shi's CL curve is the classical PXR-only rate, while Shi's QM curve and PyRITE's classical line both carry CBS and the PXR×CBS interference. That accounts for the 1.9–2.6× gap and for Shi's QM area exceeding its CL area. The PXR-only remainder is the crystal potential: Si K-edge f′ and Debye–Waller in PyRITE, Yukawa without either in Shi. Line positions are not affected.

### 5.2 Transported observability table

The table uses 2000 electrons per case (`_simulate_trajectories`, seed 11) and the rigid nominal-direction shift; scattered flights keep their own classical line positions. The detector is a Gaussian with the stated FWHM. Shifts are prototype QM minus CL.

| Case | Detector FWHM | Peak shift (eV) | Centroid shift, all rows (eV) | FWHM CL / QM (eV) | QM/CL height |
|---|---|---|---|---|---|
| Si 30 keV, 30 nm, 100° | 0 / 5 / 70 eV | −35 / −35 / −38.5 | −101 | 13 / 18 → 81.5 / 125.5 | 0.40 / 0.41 / 0.59 |
| Si 300 keV, 30 nm, 100° | 0 / 5 / 70 eV | −19.5 / −19.5 / −27.5 | −65 | 25 / 40 → 77 / 106.5 | 0.42 / 0.43 / 0.64 |
| Si 100 keV, 100 nm, 2.5° ($n_z=-1$) | 0 / 5 / 70 eV | −125.5 / −125.5 / −121 | −239 | 15 / 90 → 83 / 282.5 | 0.27 / 0.27 / 0.34 |
| HOPG (004), 11.7 keV, 85 nm, 130° | 0 / 5 / 70 eV | −35.5 / −35 / −35 | −34 | 13.5 / 13.5 → 83 / 83.5 | 1.00 |

- **Multiple scattering is minor at these thicknesses.** There are 1.2–6.5 flights per electron. Transport changes peak shifts by ≤ 2.5 eV and FWHMs by ≤ 30 eV against the straight segment (largest: forward, 70 eV detector), and the scattered flights' own line shifts are already in the classical spectra. Re-evaluating the recoil shift at each flight's own direction changes it by ≤ 5.5 eV rms (mean ≤ 0.5 eV). Thicker targets will differ.
- **Forward geometry (SI §S10) is the clean target.** With a 70 eV detector the envelope peak moves −121 eV and the FWHM grows 3.4× (83 → 283 eV; sixfold without detector broadening). After transport the leading $(\bar1,\pm1)$ shell is a shoulder, not a separate peak. The −125 eV peak is the $\lvert g_\perp\rvert^2=10$ shell, so its position depends on PyRITE's unvalidated shell weights ($\lvert g_\perp\rvert^2=2$ at 0.74 of it). The binding constraints are beam direction, forward access past the beam, and forward bremsstrahlung/CBS background, none of which are modelled here.
- **Large angle.** The peak moves 20–40 eV and the line broadens 1.4–1.5× at 70 eV resolution. With a 5 eV detector the comb's leading lines are only marginally separated at 30 keV (spacing ≈ 11 eV against an 11 eV width) and merge at 300 keV. Shi's tallest line is the same $(0,\pm2,2)$ root, so its peak shift (≈ −35 / −18 eV) matches the prototype; only the relative heights differ (§5.1).
- **vdW (HOPG):** $g_\perp=0$, so the line moves rigidly by the longitudinal recoil (option A) with no transverse comb. This is Huang's measured regime.
- **Not included:**
  - bremsstrahlung, characteristic (Si Kα 1.74 keV and the Si K edge at 1.84 keV lie inside the 30 keV comb) and CBS backgrounds;
  - beam divergence and energy spread;
  - direction-dependent recoil shifts on scattered flights;
  - the recoil Jacobian;
  - counting statistics.
