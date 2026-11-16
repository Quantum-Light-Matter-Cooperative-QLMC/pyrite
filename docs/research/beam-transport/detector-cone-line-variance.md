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

In progress; results are added here as they are measured. The estimator
derivation and its validation status live in
[`detector-cone-variance-reduction`](../../validation/beam-transport/detector-cone-variance-reduction.md).

#### Pilot evidence (November 2026)

Jobs 3701/3702 (NVIDIA, five seeds × 512 primaries per arm) ran the CPU
prototype with a $1/\gamma$ cone and target copy weight $10^{-3}$. On 5 MeV
h-BN the largest single primary still held 98 % (analog) and 84 % (split) of
the line mass. The splitting arm made 6–53 copies per seed. At 15 and
29.999 MeV (HOPG, 45°/180°) almost no copies were made and line statistics
were unchanged.

Replay job 3705 (seed 0, code `f88349b3b`) traced the dominant primary, 159,
which holds 99.75 % of that seed's analog line mass. It never enters the
$1/\gamma$ cone: its closest approach is $\theta = 0.144$ rad, and its heaviest
lines sit at $\gamma\theta = 1.3$–$2.3$, measured at the local electron
energy. Re-analysis of the replay's copy records, where $u=\gamma\theta$ is the
launch angle of each copy measured from the detector direction:

| Quantity | Value |
|---|---|
| Copies / parents | 45 / 8 (38 from primary 159) |
| Unweighted copy line mass, median (range) | $2.4\times10^{-3}$ ($9\times10^{-5}$–$9.6\times10^{-2}$) |
| Median analog primary line mass | $6.0\times10^{-9}$ |
| Primary 159 line mass | $1.8\times10^{-3}$ |
| Copy line mass vs launch $u\in[0.19, 0.92]$ | no trend (log–log correlation $-0.08$) |
| Squared coefficient of variation of in-cone mass | 3.1 |
| Estimated in-cone entry rate, $\sum_c w_c / N$ | $5\times10^{-5}$ per primary |

Three conclusions follow. They are order-of-magnitude statements from one
seed, and the copy records are conditioned on primary 159.

1. **The $1/\gamma$ cone is the wrong importance region.** Copy line mass is
   flat over $u\lesssim1$, and an analog history at $u\approx1.5$ carries a
   comparable mass. Since the solid angle grows as $u^2$, histories outside
   the cone contribute at least as much as those inside. Splitting only
   inside $u\le1$ adds low-variance copies but leaves the dominant
   out-of-cone primaries analog, so the pilot RSE did not improve.
2. **The rare population does exist, and importance sampling reaches it.**
   An in-cone history is about $4\times10^5$ times heavier than a median one.
   With the in-cone class alone, analog sampling needs about $7.6\times10^6$
   primaries for RSE 0.1: rate $5\times10^{-5}$, $\mathrm{CV}^2=3.1$, so
   $N\approx(1+\mathrm{CV}^2)/(p\cdot0.01)$. The weighted copies reach the same
   RSE for that class with about $2.5\times10^4$ primaries, at roughly 9 %
   extra transport. At the pilot's ~0.1 s per primary, this is about nine
   GPU-days of brute force against under an hour. This projection is
   provisional, not the Plan's measured baseline.
3. **Intra-class spread sets a floor.** In-cone copies vary over three
   decades at fixed launch angle; their launch depth and energy matter more
   than $u$. Angular importance alone cannot remove $\mathrm{CV}^2\approx3$.

#### Method comparison

| Criterion | Splitting + roulette at elastic events (prototype) | Biased large-angle elastic sampling |
|---|---|---|
| Weight bookkeeping | Analog primary keeps weight 1. Each copy carries $P_B h/\pi$ (or $/s$ after roulette), bounded by $\max(1,w_t)$; copies do not split again. | One scalar weight per history, multiplied by $p/q$ at every biased event. |
| Weight-range growth at MeV | None: weights do not compound. | A primary has ~$10^4$ elastic segments (8104 traced for primary 159). Any per-event bias compounds. A defensive mixture $q=(1-\varepsilon)p+\varepsilon g$ caps out-of-region ratios at $1/(1-\varepsilon)$ but drifts by $(1-\varepsilon)^{-n}$, which forces $\varepsilon\lesssim1/n\sim10^{-4}$. |
| Other tallies (brem, characteristic) | The primary stays analog; copies add small-weight terms. No variance penalty. | Every tally of every history carries the fluctuating weight, which inflates bremsstrahlung and characteristic variance. |
| RNG for copies | Event key = splitmix(split key, flight). Copy transport keys re-hash it under a separate salt. No draw comes from the primary stream, so biasing off is bit-for-bit analog. | Mixture and branch draws need their own keyed stream to keep biasing-off bit-for-bit. No copies. |
| Block invariance (#361) and CUDA seeding | Copies are a function of (primary key, flight) only, so they are independent of block layout. They launch through the existing secondary-generation pass, as secondaries do. | Trivial: one history per primary. |
| Cost | Copy buffer and launch generation, already present for secondaries. Copy count is tuned by $w_t$. | No buffers. |
| Derivation / ledger | `detector-cone-variance-reduction`; the derivation holds for any half-angle $<\pi/2$ (rederivation in draft #1237). | New claim and new verification. |

**Choice: keep splitting with Russian roulette, and widen the importance
region.** The rederived estimator is unbiased for any cone half-angle
$\alpha<\pi/2$, so widening needs no new physics. Biased elastic sampling
offers no better coverage at an equal draw rate. It compounds weights over
~$10^4$ events and costs variance on the other tallies.

The next measurement is the region's size. Proposed run: replay 5 MeV h-BN
with half-angle $k/\gamma$ for $k\in\{1,2,3,4\}$, five seeds × 512 primaries,
warmed and counterbalanced arms, recording copy mass against launch $u$,
depth and energy. Choose $k$ and $w_t$ to minimize projected per-primary
variance × time. Keep `max_copies_per_electron` above the expected copy
count: primary 159 alone made 38 copies at $k=1$.

**Pixel scorers (#212).** Bias toward one cone about the scorer's central
direction, with half-angle equal to the scorer's angular radius plus
$k/\gamma$. Leave biasing off when that exceeds $\pi/2$, the prototype's
restriction. This keeps one transport per case, consistent with the
one-detector-per-transport decision.
