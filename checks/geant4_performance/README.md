# Matched-bounds Geant4 single-scattering benchmark (#821)

Performance comparison of PyRITE CUDA against Geant4 11.4.2 single scattering
on the same problem, with physics bounds matched so that neither engine does
work the other skips.

## What is compared

Three arms share one transport problem; they differ only in how bremsstrahlung
photons are produced and scored.

| Arm | Transport | Photons ≥ 1 keV | Scored observable |
|---|---|---|---|
| **Geant4 discrete** | DPWA single scattering, eIoni, eBremSB | sampled discrete eBremSB secondaries | counted per primary at creation, then killed |
| **PyRITE discrete** | ELSEPA single scattering, SBETHE, coupled BremsLib soft/hard | sampled discrete hard events (energy and direction) | counted per primary at creation, then dropped |
| **PyRITE continuous** | identical to PyRITE discrete (the hard events still debit energy) | not counted | production weighted per-segment estimator: expected photons from the BremsLib DDCS by track length along every segment (`mc_coupled_brem_spectrum`) |

In every arm, radiative loss below 1 keV is a continuous energy loss.

Two observables:

1. **4π yields** (photons ≥ 10 keV per primary in bands, emitted energy):
   discrete arms only. Transport plus discrete brem in both programs.
2. **Detector-cone yield**: ≥ 10 keV photons emitted at creation into a cone
   centred 90° from the beam (x-z plane), 10° half-angle, unit transmission
   (`scenes.json` `cone`). All three arms. Comparing Geant4 discrete with
   PyRITE continuous is the "transport + discrete brem in Geant4 versus
   transport + continuous per-segment brem in PyRITE" comparison. Its fair
   metric is the figure of merit, because the estimators have different
   variance for the same mean.

The continuous arm integrates the cone with one of three paths of existing
code, chosen by measured speed (`--cone-mode`):

- `adaptive`: angular/energy Gauss rule refined until per-primary sums and
  sums of squares change by `rtol` (1e-3);
- `fixed`: one evaluation at the order the adaptive rule converged at;
- `point`: the cone axis only, times the cone solid angle (the production
  point-detector convention). Its bias against `adaptive` on identical
  trajectories is measured and reported.

No PXR/CBS, characteristic, coherent, line or detector-response computation
runs in any arm, and no photon or secondary electron is transported. Each
PyRITE process runs one extra untimed request under a call profiler and fails
if any `pyrite.montecarlo.spectrum.cxr` or `.lines` computation executes. The
only exceptions are three segment-geometry helpers the brem estimator imports
(`_clip_segments_to_cutoff`, `_observation_direction`,
`_validate_groove_escape_direction`).

## Matched bounds

| Bound | PyRITE | Geant4 |
|---|---|---|
| Electron tracking cutoff | `E_cut_keV=10` | `lowestElectronEnergy 10 keV` |
| Elastic scattering | ELSEPA single scattering, all angles | `G4eDPWACoulombScatteringModel`, single scattering, μ ∈ [0, 1] |
| Atomic-electron angular deflection | Kawrakow | DPWA scattering-power correction (elastic σ × scp(1+1/Z)) |
| Collision loss | SBETHE continuous, no straggling | `eIoni`, fluctuations off |
| Delta rays | none | none: electron cut 1 m (≈1 GeV threshold) |
| Photon threshold | 1 keV hard cutoff | γ cut exactly 1 keV (cut-table lower edge 1 keV) |
| Other processes | none | removed: ePairProd, PIXE, Auger, fluorescence, γ/e⁺ physics |
| Step control | `max_dE_frac=0.05` | default step function (0.2, 1 mm) |
| Geometry | 10 × 10 mm slab in vacuum, normal incidence | same |
| Termination | first slab exit or cutoff | same, primary killed on exit |

`geant4_bench.py` refuses a run whose dumped electron process list, γ/e⁻
thresholds, lowest energy or fluctuation flag differ. Process-resolved step
counts (Geant4) and row kinds (PyRITE) are recorded to show which limits bind.

Residual model differences remain and are reported, not hidden: SBETHE versus
Geant4 Bethe–Bloch stopping, BremsLib versus Seltzer–Berger DCS and angular
generator, the two atomic-electron deflection treatments, and different
step-integration schemes. `analyze.py` reports the agreement of each observable.

## Timing contract

- **Warm request**: PyRITE, a complete measured request after a same-size
  warmup request, CUDA-synchronized, including per-block score reduction.
  Geant4, the measured `BeamOn` after initialization and a warmup run.
- **Cold**: PyRITE imports/tables plus the first request; Geant4
  initialization, warmup run and measured run.
- **Figure of merit**: 1 / (RSE² · T_warm) for the ≥ 10 keV 4π yield (discrete
  arms) and the ≥ 10 keV cone yield (all arms). An arm's FOM ratio is its
  equal-precision speed ratio, independent of history count.
- PyRITE warm/cold times include block score reduction and, in the continuous
  arm, the estimator. `work.*.scoring_s` separates scoring from transport.

PyRITE runs the production `auto` (bounded continuation) schedule, which never
replays a history. The full-history fallback is also replay-free: initial
segment capacity 16384 (above the longest observed track), no probe batch,
per-scene `max_steps` (block × max_steps ≤ 5×10⁸ host rows; well above the longest track), 4 GiB scratch. The arm fails if launched and retained rows
differ.

## Running

From a synced lab checkout (`PYRITE_REMOTE_DIR=~/pyrite-821/repo uv run pyrite remote sync`):

```bash
sbatch checks/geant4_performance/pilot.sbatch
hold=$(sbatch --parsable checks/geant4_performance/hold.sbatch)
sbatch --dependency=after:$hold checks/geant4_performance/run.sbatch $hold 100000 1 2 3
uv run python checks/geant4_performance/analyze.py evidence/run/JOB --output summary.json
```

`hold.sbatch` takes the cpu partition's 16 CPUs on the GPU host while
`run.sbatch` takes the gpu partition's 8; partition caps leave the remaining 8
unallocatable, so the host runs nothing else during measurement. Geant4 runs at
1, 16 and 32 threads (16 cores, SMT 2).
