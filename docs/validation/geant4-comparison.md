# PyRITE versus Geant4: accuracy and performance

Geant4 is an **independent simulation reference**, not experimental ground truth.
The checks below match incident particles, materials, geometry and selected
scoring conditions where documented, but the codes do not always use identical
physics models or Monte Carlo estimators. These are **bounded, dated measurements**,
not universal validation or a claim that one engine is faster.

## Agreement on matched physical cases

| Comparison | Scope | Recorded result | Important boundary |
| --- | --- | --- | --- |
| [Primary electron transport](beam-transport/mev-full-track-transport.md) | Si and W, 3–100 MeV; 10 cases, 100,000 primaries each | **78/80** predefined observable gates pass; both exceptions are W exit-angle rms at 50 and 100 MeV | PyRITE ELSEPA/SBETHE/BremsLib versus Geant4 11.4.2 TestEm5 `empenelope`; primary only, without energy-loss straggling or transported secondaries |
| [Thick-target bremsstrahlung and photon transport](radiation-physics/photon-full-track-thick-brem.md) | Si and W, 0.3–10 MeV; eight cases, front and rear apertures | **100/100** revised gates pass with matched atomic vacancy-cascade settings (2026-10-27) | Tests selected redistributed-photon yields, fractions and W fluorescence lines; direct and total source spectra are report-only |

For the primary-electron study, a gate requires agreement within the larger of
**5% of the Geant4 value** or **three combined standard errors**. Selected
**PyRITE / Geant4** ratios are shown below (not percentage uncertainties):

| Case | Continuous energy deposit | Photon energy ≥10 keV | Exit-angle rms |
| --- | ---: | ---: | ---: |
| W, 3 MeV | 1.035 | 1.023 | 0.993 |
| W, 50 MeV | 1.001 | 1.029 | **0.928 (fails)** |
| W, 100 MeV | 0.998 | 1.026 | **0.926 (fails)** |
| Si, 30 MeV | 1.001 | 1.008 | 0.982 |

The two W failures reflect different nuclear-size scattering assumptions:
Geant4's reference uses a point-nucleus Mott correction, whereas the production
ELSEPA tables use a finite-size nucleus. A targeted point-nucleus PyRITE
control brings the two rms ratios to **0.981** and **0.988**, passing both
gates. See the [transport comparison and its full observable matrix](beam-transport/mev-full-track-transport.md)
for detailed limitations.

For the thick-target study, the historical Geant4 reference suppressed parts
of the atomic vacancy cascade, giving **98/100** passing revised gates. The
corrected complete-cascade reference passes **100/100** without changing PyRITE
outputs or comparison tolerances. These are disclosed, post-inspection revised
gates, **not** proof of agreement for all emitted photon spectra. See the
[full study and independent adjudication](radiation-physics/photon-full-track-thick-brem.md).
Human validation sign-off remains separate.

## Performance: measured timings, not a qualified speed ratio

The [equal-precision pilot](beam-transport/equal-precision-pilot.md) measured
100,000 primary electrons in two matched slabs on **2026-10-14–15**.
Hardware: AMD Ryzen 9 9950X3D and NVIDIA RTX 5080. Geant4 11.4.2 used
**one CPU physics worker**; PyRITE used CUDA. The recorded hard-photon
source was an *event tally*, not the production weighted per-segment
spectrum estimator.

| Case (100,000 primaries) | PyRITE CUDA warmed transport + emission (s), one run | Geant4 `empenelope` native process (s), median of 3 | Geant4 `emstandardSS` native process (s), median of 3 |
| --- | ---: | ---: | ---: |
| Si, 300 keV, 100 µm | 59.627 | 5.495 | 92.048 |
| W, 300 keV, 10 µm | 208.586 | 5.880 | 116.235 |

**Do not divide these columns to claim a speedup.** PyRITE's interval excludes
JIT compilation, warmup, setup and reductions, while Geant4's native
fresh-process interval includes startup and diagnostic logging. The default
Geant4 arm uses *condensed-history multiple scattering* while PyRITE resolves
individual elastic flights. The Geant4 single-scattering arm is closer in
stepping policy but also changes its ionization and bremsstrahlung models.
Numerical convergence, matched timing boundaries, equivalent observable
distributions and history counts chosen for equal precision have **not** all
been established.

The PyRITE CUDA times above predate a measured radiative-kernel optimization.
Its separate **PyRITE-to-PyRITE** 100,000-history measurement reduced warmed
CUDA transport time from **60.451 to 34.514 s** (Si) and **209.200 to
122.267 s** (W). Those are **not** updated Geant4 comparisons; see the
[optimization evidence](beam-transport/equal-precision-pilot.md#exact-cuda-radiative-evaluation-optimization).

![Measured source-yield relative precision for the 100,000-history pilot](check-records/equal-precision/scaling-precision.png)

## Interpretation and reproducibility

These studies establish **specific agreements and discrepancies** for the
declared configurations; neither validates PXR/CBS, full charged-particle
cascades, all detector observables, or arbitrary materials and energies.
The pilot's similar integrated source-yield relative errors (Si:
**3.87% CUDA / 3.89% Geant4**; W: **1.64% / 1.66%**) do not by themselves
establish equivalent physics or time-to-precision.

Reproduction inputs, physics switches, reductions, confidence rules and
provenance are retained in the [MeV comparison harness](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/tree/main/checks/mev_full_track),
[photon comparison harness](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/tree/main/checks/photon_full_track)
and [equal-precision benchmark](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/tree/main/checks/equal_precision).
For a qualified cross-engine performance claim, consult the
[equal-precision protocol](beam-transport/equal-precision-protocol.md) and
[validation methodology](methodology.md).
