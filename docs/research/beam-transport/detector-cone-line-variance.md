# Detector-cone electrons and MeV line-yield variance

**Status:** Research record, September–October 2026 (#201, #203). It describes a measured property of the current incoherent PXR/CBS estimator and the variance-reduction study it motivated. It is not a validated model; the production policy is the line-yield statistics warning in `montecarlo/runner/line_grid.py`.

## Finding

At MeV beam energies the incoherent PXR/CBS line yield toward a fixed detector is a rare-event estimator. A few electrons that elastic scattering turns into the detector's $1/\gamma$ radiation cone carry almost all of it. Every other electron contributes lines $10^2$–$10^3$ times lighter. The sample mean is unbiased, but its relative standard error (RSE) falls only as $1/\sqrt{np}$, where $p$ is the rate of the heavy class, so it stays large until many heavy electrons have been drawn.

## Line attribution (#201)

Case: h-BN, 5 MeV, 1 mm slab, tilt/azimuth 10°/100°, `resonance-local` line grid, seed 0, $N_e = 8000$ electrons, CUDA float32 and float64 on one identical transport (21.8 M segments, about 43 M lines). The attribution diagnostic (`python -m pyrite.energy_grid.bandwidth_check attribute`, launched by `convergence_job start-bandwidth --attribute`) records the heaviest lines by mass $w\pi/a_w$ with the factors that built them, and the per-electron line mass.

- **Precision is not the cause.** Line mass per electron is $2.589\times10^{-6}$ in float32 and in float64; per-electron and per-quarter sums agree to about $10^{-6}$.
- **Two electrons carry 99.6 % of the line mass**: electrons 5025 (65.1 %) and 4868 (33.4 %). The per-quarter electron sums are $9.8\times10^{-5}$, $5.1\times10^{-5}$, $2.05\times10^{-2}$ and $4.1\times10^{-5}$. All 200 heaviest lines come from those two electrons, from long straight flights of 90–110 consecutive segments on the $|g| = 1.89$ Å$^{-1}$ (0002)-type reflections.
- **The heavy lines are CBS-dominated.** The median $|A_\mathrm{CBS}|^2/|A_\mathrm{PXR}|^2$ is 11 and 68 for the two electrons. Lines lie at 5–42 keV with widths of 94–200 eV. The Bragg detuning is ordinary (3–4.6 Å$^{-2}$), so detuning is not the mechanism.
- **Mechanism.** Their $1 - \boldsymbol\beta\cdot\hat{\mathbf n} = 6.5\times10^{-3}$–$7.6\times10^{-3}$, against $1/(2\gamma^2) = 6.3\times10^{-3}$: each electron travels 15–120 mrad from the detector direction, inside the $1/\gamma = 112$ mrad radiation cone. Its $\mathbf v\cdot\mathbf g$ is 0.02–0.14 Å$^{-1}$, against 1.85 Å$^{-1}$ for the ordinary 3.6 keV line population. The CBS factor $f_\mathrm{cbs}\propto 1/(\gamma\,\mathbf v\cdot\mathbf g)$ (plus a $1/(\mathbf v\cdot\mathbf g)^2$ term) and the Doppler-boosted photon energy make each line $10^2$–$10^3$ times heavier than an ordinary one. With the detector at 90° to the incident beam, both electrons were scattered by about 90°.
- **The amplitude is within its stated validity.** $|U_g|g^2/(\gamma m c^2(\mathbf v\cdot\mathbf g)^2) \le 0.018$. The angle to the planes is 10–72 mrad, above the h-BN Lindhard angle (about 3 mrad at 5 MeV), so this is not planar channeling.
- **Statistics.** From the 20 heaviest electrons the RSE of the mean at $N_e = 8000$ is about 0.73. The heavy-class rate is about $2/8000 = 2.5\times10^{-4}$. The 70-fold jump in line yield between $N_e = 4000$ ($3.74\times10^{-8}$) and $N_e = 8000$ ($2.59\times10^{-6}$) is sampling, not a defect. The 80°/180° geometry is stable at $2.5$–$2.8\times10^{-8}$ because it has no such population, or has not sampled one.

## Measured stop and the statistics warning

The same electrons also set the resonance-population measured stop. At $N_e = 8000$ the stop is 672 keV although no line resonates above 77.5 keV: the cone electrons' lines are both heavy and wide (small $1 - \boldsymbol\beta\cdot\hat{\mathbf n}$ gives first-zero widths of 0.5–9.5 keV), and the $w/(\pi^2 D)$ tail bound decays only as $1/D$. Their tail shares above the stop are 0.63 (5025) and 0.37 (4868). At $N_e = 4000$ one electron (159) already holds 51 % of the mass. The stop is correct for the sampled population; it is a symptom of the rare-event estimator, not a selector defect.

| $N_e$ | stop (eV) | max resonance (eV) | top-1 / top-10 electron mass share |
|---:|---:|---:|---|
| 4,000 | 35,400 | 20,705 | 0.51 / 0.76 |
| 8,000 | 672,400 | 77,474 | 0.65 / 0.996 |

Policy adopted under #201: the resonance-population audit sums line mass per electron and records `line_yield_statistics` (electron count, RSE of the mean, largest electron share). Above `LINE_YIELD_RELATIVE_SE_LIMIT = 0.1` the case still runs, is flagged `statistics_limited`, and raises `LineYieldStatisticsWarning`. Adaptive stopping (#361) cannot see an unsampled heavy class; on this case it ran to its 16,000-electron ceiling at line RSE 0.39 (largest electron share 0.25) and ended `statistics_limited`. In the `high_energy` h-BN scan, 15 of the 37 cases that record the flag were statistics-limited.

## Variance reduction

Variance reduction for this population is studied under #203: a brute-force baseline, a choice between detector-cone splitting with Russian roulette and importance sampling of large-angle elastic scattering, and a per-electron CPU prototype. The analysis follows in {ref}`research-detector-cone-vr`.

(research-detector-cone-vr)=
### Study (#203)

In progress; results are added here as they are measured.
