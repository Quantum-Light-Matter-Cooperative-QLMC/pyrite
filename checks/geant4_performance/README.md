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
- **Time to precision (headline)**: #510 `T(ε)`, the warm request time of a
  held-out run at the history count the arm needs for every headline observable
  of a set to reach RSE ε. See "Time to precision" below.
- **Figure of merit (diagnostic)**: 1 / (RSE² · T_warm) for the ≥ 10 keV 4π
  yield (discrete arms) and the ≥ 10 keV cone yield (all arms), from
  equal-history samples. It predicts the `T(ε)` ratio only in the
  1/√N regime with fixed costs negligible, so it is not reported as a speed ratio.
- PyRITE warm/cold times include block score reduction and, in the continuous
  arm, the estimator. `work.*.scoring_s` separates scoring from transport.

PyRITE runs the production `auto` (bounded continuation) schedule, which never
replays a history. The full-history fallback is also replay-free: initial
segment capacity 16384 (above the longest observed track), no probe batch,
per-scene `max_steps` (block × max_steps ≤ 5×10⁸ host rows; well above the longest track), 4 GiB scratch. The arm fails if launched and retained rows
differ.

## Time to precision

`scenes.json` `precision` freezes the protocol: targets ε = 5% and 2% (1%
optional), held-out seeds 101–103, and three observable sets:

- `transport`: transmitted, backscattered and stopped fractions, and
  transmitted energy per primary;
- `source_4pi`: ≥ 10 keV photons per primary and the 10–50, 50–100 and
  ≥ 100 keV bands;
- `cone`: ≥ 10 keV photons per primary into the 90°/10° cone.

Every Geant4 configuration and PyRITE discrete run all three sets. PyRITE
continuous runs only `cone`: its transport is the discrete arm's, and it counts
no 4π photons. Fractions below 2%, transmitted energy when transmission is
below 2%, 4π bands below 10⁻³ photons per primary, and bands at or above the
beam energy are diagnostics. The cone yield is always headline: it is rare by
design, and its cost is the point of the comparison.

1. **Variance pilot** (any fixed-history evidence: `run.sbatch`,
   `pilot3.sbatch`, `pilot4.sbatch`). `precision_plan.py` estimates each
   observable's per-primary CV per engine arm (Geant4 thread counts pooled) and
   freezes `N(ε) = ⌈(CV/ε)²⌉` (at least 2000) per (scene, configuration, set,
   ε), the largest over the set's headline observables binding. Predicted warm
   time scales the pilot's warm seconds per primary. Items above `--max-run-s`
   are kept as `over-budget` and not run.
2. **Held-out runs** (`precision.sbatch`, whole node held). Each planned
   (scene, configuration, N) runs once per held-out seed in a rotated order.
   PyRITE warms up on one block (`--warmup-histories`) instead of a
   full-size request; Geant4 warms up on N/100 (at least 100).
3. **Verification** (`analyze.py --plan`). A run counts only if every headline
   observable of its set achieved RSE ≤ ε. Otherwise the item is
   `precision-unmet`; it is never extrapolated. `T(ε)` speed ratios against
   each Geant4 thread count (median and range over all seed pairs) are given
   only between met items. PyRITE rows also carry the transport/scoring
   split.

```bash
uv run python checks/geant4_performance/precision_plan.py evidence/run/2835 \
    evidence/pilot3/2864 evidence/pilot4/JOB --output precision-plan.json
hold=$(sbatch --parsable checks/geant4_performance/hold.sbatch)
sbatch --dependency=after:$hold checks/geant4_performance/precision.sbatch $hold precision-plan.json
```

## Spectral-grid sensitivity (#278)

The continuous arm's cone scoring does not use a photon grid. For each
≥10 keV band it splits the band at the slab's EPDL absorption edges, cuts
panels wider than 100 keV into equal pieces, and places `2 × --cone-order`
Gauss–Legendre nodes per panel. For W at 1 MeV this gives 15 panels and 240
nodes at order 8. Each band is one `mc_coupled_brem_spectrum` call with
per-electron band weights.

Two repeatable flags also score every transported block with the production
continuum, `mc_coupled_brem_spectrum`, toward the cone axis with unit
transmission and the runner's brem chunk policy:

- `--spectrum-step-ev STEP`: the uniform grid `STEP, 2·STEP, … < E0`. The
  production `E_grid_brem` step is 25 eV.
- `--spectrum-geometric-nodes N`: `refined_continuum_grid`, `N` log-spaced
  nodes from the medium's continuum floor to `E0`. The grid is refined at
  absorption edges and at the endpoint (`continuum-node-refinement`).

Each grid is timed separately and excluded from the request timings.
`work.<request>.grid_spectra.<label>` records the seconds, node count and
chunk. It also records the cone-band yields and the ≥10 keV centroid in the
point convention, piecewise-linear in energy, which check each grid against
the arm's quadrature and the 25 eV grid. The densities go to
`<output>.grid.npz`. `gridsens.sbatch` runs W 1 MeV.

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
