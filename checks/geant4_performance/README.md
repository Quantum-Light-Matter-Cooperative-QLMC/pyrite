# Matched-bounds Geant4 single-scattering benchmark (#821)

Performance comparison of PyRITE CUDA against Geant4 11.4.2 single scattering
on the same problem, with physics bounds matched so that neither engine does
work the other skips.

## What is compared

**Electron transport plus discrete bremsstrahlung photon emission, in both
engines.** Specifically:

| | PyRITE CUDA | Geant4 (`g4bench`) |
|---|---|---|
| Radiative loss below 1 keV | continuous BremsLib soft loss | continuous eBremSB loss below the 1 keV photon cut |
| Photons ≥ 1 keV | discrete hard events sampled in transport (`bremslib-soft-hard`), photon energy and direction completed per event | discrete eBremSB secondaries, energy and direction sampled per event |
| Photon scoring | each hard photon counted per primary at creation, then dropped | each secondary photon counted per primary at creation, then killed |

This is **not**:

- transport alone (radiative loss and photon sampling are on in both); or
- PyRITE's weighted per-segment continuous emission estimator (the production
  detector-spectrum scorer). That estimator integrates emission along every
  segment rather than sampling photons, has a different variance, and is not
  run here. Comparing it with Geant4 discrete photons needs an
  equal-precision study of the detector observable itself.

No PXR/CBS, characteristic, coherent or detector-response computation runs in
either engine. No photon or secondary electron is transported.

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
- **Figure of merit**: 1 / (RSE² · T_warm) for the ≥ 10 keV photon yield per
  primary. The PyRITE/Geant4 FOM ratio is the equal-precision speed ratio.

PyRITE runs the production `auto` (bounded continuation) schedule, which never
replays a history. The full-history fallback is also replay-free: initial
segment capacity 16384 (above the longest observed track), no probe batch,
`max_steps=200000`, 4 GiB scratch. The arm fails if launched and retained rows
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
