# Frozen randomized fiber roll experiment (follow-up to #1527)

## Scope and audit

Reviewed branch `issue-1527-bounded-fiber-grid`, head
`70306da1da4527e89e4714ac23bc0cec2ecb5c97` (pilot tail study). This
is a check-only Phase I experiment, not a validated production feature.

- `FiberReflectionSet.__post_init__` enforces 60-degree closure and consistent
  thermal rows. `_average` integrates intensities over one sector, scores basal
  once, and shares the host bin-line sink across reflection batches.
- `fiber_lines` strips the fiber keys before dispatching each batch through
  `runner._cxr_for_segments`. It weights tail audits by quadrature weights.
  This check uses the same lowered domain, projected thermal values, dispatch,
  explicit bin-mean energy axis and shared sink. It does not hook tail audits.
- `electron_blocks` already defines additive disjoint row subsets with the
  original `Ne` and original electron/flight IDs. The check uses `_SEG_ARRAYS`
  and `_take` for that contract. Retained transport only; no block-local ID
  offset/replay contract is asserted. The CXR setup normalizes by `electron_limit`
  and groups physical flights by ID rather than row adjacency.
- #1527 pilot/certification remains untouched. A production stochastic mode
  would need the same realized assignments in stopping, pilot, tail, final
  scoring and checkpoint identity; those changes are not part of this patch.
- The old #1480 `frozen` probe has S arms on the full circle and a per-family
  scorer. The new check uses the production full domain and sector instead,
  explicitly isolates basal and non-basal, and saves independent seed replicates.

## Estimator

For incident primary IDs `0..Ne-1` and sector length `L=pi/3`, deterministic
QN scores all rows at every midpoint and applies `1/N` to the non-basal
intensity only. QN_shift shifts by half a node spacing modulo the sector.
Both use the same basal spectrum, computed separately once. The production
parity arm compares their separated sum at the reference Q against the real
`fiber_lines` path, with identical case and frozen rows. Host Fourier flush
boundaries may differ; inspect mass/shape differences rather than require
bitwise equality.

Independent I chooses one roll per primary from a check-owned, seed-keyed
SplitMix64 transform of the primary ID. This is a diagnostic RNG, not a
replacement for the production counter RNG. I uses one scalar-roll call per
populated primary and reflection batch; it is intentionally capped by default
because it is not the proposed GPU-native row-dependent kernel. No segment,
flight or secondary is randomized separately.

Grouped SK uses independent randomization groups of `K * histories_per_roll`
consecutive global primary IDs. Each group draws one jitter per sector stratum
and a random permutation of stratum labels; IDs cycle through that permutation.
Each primary is marginally uniform, including an incomplete last group. The
same seed and group size give the same assignment on a longer prefix. The
sum uses the **global** incident denominator: no extra `1/K` or occupancy
weight is applied to the globally normalized subsets. Misses contribute zero.
A shared slot creates dependent scores; only independent orientation replicate
seeds are used to estimate conditional angular sampling error here. No IID
per-electron RSE, adaptive stopping or production grouped mode is exposed.

## Run sequence

All commands assume the repository root and a prepared PyRITE environment.
The default scene is the public campaign research scene: 300 keV, 500 nm HOPG,
50-degree target tilt, 180-degree tilt azimuth, point beam, scalar 45-degree
lab detector, 5x5 footprint, cutoff 8 inverse angstrom, 256 reflection batch,
5 mosaic nodes and target mosaic disabled. This does **not** resolve the saved
`cleo_fiber` catalog profile. Check detector/beam/footprint and other resolved
settings against that profile before interpreting absolute yields. In particular,
`mosaic_nodes=5` alone does not enable target mosaicity. If the target has
catalog mosaic enabled, specify `--mosaic --mosaic-route mc`; auto on the
reviewed branch is not a supported full-fiber mosaic quadrature route.

Pure NumPy tests (no physical databases or PyRITE import required):

```bash
python -m pytest -q checks/test_fiber_randomized_rolls.py
```

Cheap production-seam smoke first (intentionally not an angular reference):

```bash
pixi run -e nvidia python checks/fiber_randomized_rolls.py \
  --histories 32 --cutoff 4 --step 4 --energy-max 13000 \
  --quadratures 4 --reference 8 --shifted 4 --strata 4 8 \
  --orientation-seeds 101 102 --independent --production-parity \
  --output scratch/fiber-roll/smoke --plot
```

First physics ladder, with the same frozen transport for every arm:

```bash
pixi run -e nvidia python checks/fiber_randomized_rolls.py \
  --histories 512 --cutoff 8 --tilt 50 --step 1 --energy-max 13000 \
  --quadratures 32 64 128 256 --reference 512 --shifted 32 128 \
  --strata 128 1024 --histories-per-roll 32 \
  --orientation-seeds 101 102 103 104 105 \
  --band 900 1100 --band 1900 2100 --band 3000 6000 \
  --output scratch/fiber-roll/tilt50-n512 --plot
```

If scalar per-primary I is too slow, test it separately at small N, retaining
Q32 and at least one candidate reference for that population. Extend the
reference to Q1024 if Q256/Q512 have not converged. The highest Q is explicitly
marked uncertified in the JSON. The example energy band is just a common
intrinsic comparison band; extend it if meaningful line/tail mass falls
outside it. Use the actual detector geometry and physically relevant windows.

Scale at fixed cutoff, photon axis and physical model (e.g. 4096, 32768,
262144 histories as resources permit). For larger N, use `--strata 128 1024
4096`, vary `--histories-per-roll` explicitly, and select a tractable reference.
Do not extrapolate from the smoke timing. Repeat at tilts 15, 30, 60, 75, 85;
repeat with different `--transport-seed` values to distinguish transport
uncertainty from conditional angular uncertainty. With a matching scene and
seed, the transport fingerprint should agree for changes to orientation seeds.

## Outputs and interpretation

- JSON: git SHA and dirty status, lowered case, geometry, backend, segment
  dtype, transport fingerprint, incident and represented counts, per-arm
  yield, strongest peak/centroid/FWHM, mass-normalized spectral L1 and peak
  L-infinity errors, optional fractional-overlap band errors and replicate
  uncertainty; synchronized wall times and row/primary-roll work, process
  high-water RSS and device-pool reservation at arm completion.
- NPZ: common centers/edges, separately scored basal, each non-basal and
  total spectrum, assignment labels/roll angles/occupancies, ensemble means
  and binwise standard errors. `--plot` makes a native-resolution PNG with
  separate basal and non-basal panels. No smoothing or convolution.
- Timing: warm-up is a separate one-roll call, not included in arm timing.
  `pre_final_flush_s` includes subset gathering, table setup, kernel work,
  device reductions and any intermediate host flushes; `final_host_flush_s`
  measures only the final shared-sink flush. `radiation_plus_setup_s` adds
  roll assignment setup and the common separately measured basal time to
  each non-basal arm. These are retained-path cold-table timings, not
  end-to-end campaign timings. The process RSS is cumulative and device
  pool reservation is **not** a measured peak of live device allocations.
  Exact table time, line counts, isolated GPU reduction timing, measured
  grid cost, transport+continuum cost and true per-arm peak memory require
  further instrumentation. Conditional yield FOM is based on seed-to-seed
  sample variance at fixed frozen transport, not total campaign variance.
- Check phase dependence of Q32 and convergence of the Q ladder. Compare
  randomized ensemble means against a sufficiently converged reference,
  and the variation/error of individual randomized arms against Q32 at
  comparable wall cost. Turning-point features can be real; no acceptance
  gate is based solely on smoothness. Binwise SE is not a simultaneous
  confidence band. For basal windows use the non-basal band errors as well
  as total peak metrics; fixed band choices are not automatic line finding.

The pure sampling tests establish range, seed/prefix invariance, additive
normalization including missed primaries, and a synthetic narrow-line example.
They do not establish PyRITE kernel accuracy or actual performance. The
physical experiment is not run or validated by this patch. No production
route, identity, validation ledger or scientific default is changed. Proceed
with a GPU-native independent-roll implementation only after the measured
accuracy/cost gate is met. A grouped production alternative must first qualify
its independent group statistics and fixed-N replay/grid contract.
