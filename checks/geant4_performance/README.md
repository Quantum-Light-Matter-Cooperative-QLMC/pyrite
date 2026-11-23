# Matched-bounds Geant4 single-scattering benchmark (#821)

Performance comparison of PyRITE CUDA against Geant4 11.4.2 single scattering
on the same problem, with physics bounds matched so that neither engine does
work the other skips.

Numerical convergence remains unverified (#1041). Statistical equivalence,
SHIFT/tail and precision gates alone do not qualify a speed headline.
`analyze.py --plan` retains their verdicts but emits `headline=false` with
an explicit `numerical_convergence` record for every ratio.

## Accuracy-only CPU references (#1041)

`accuracy-reference-plan-v1.json` derives the two missing Si cone populations
at epsilon = 2% from the frozen v3 plan, with disjoint seeds starting at
200001 and at most one million primaries per shard. The 8-thread CPU shards
may run on Hoffman2; their times cannot be compared with qlmc timings.
`accuracy-reference.sbatch` limits each shard to 4 GiB and 30 minutes; submit
at most four concurrently after validating the runtime on a compute node.
The runtime and Python environment must already be staged and verified.

```bash
uv run python checks/geant4_performance/accuracy_reference.py plan \
    checks/geant4_performance/precision-plan-v3.json --output reference-plan.json
# LAST is len(plan["tasks"]) - 1. RUNTIME is an absolute shared-filesystem path.
sbatch --array=0-LAST%4 checks/geant4_performance/accuracy-reference.sbatch \
    reference-plan.json "$RUNTIME" evidence/accuracy-reference/shards
uv run python checks/geant4_performance/accuracy_reference.py summarize \
    reference-plan.json evidence/accuracy-reference/shards --output reference-summary.json
```

The reducer requires every planned shard, disjoint planned seeds, consistent
binary identity, retained raw-score SHA-256 checks and exact incident-primary
counts. Missing or mismatched shards fail rather than produce a partial
reference. These records carry `purpose="accuracy-only"`; the timing analyzer
rejects them as inputs to `T(epsilon)`. Achieved RSE is reported separately
from numerical convergence, which these reference runs do not establish.

`evidence-inventory-v1.json` inventories the unpublished local backup of the
pilot and v2/v3 evidence plus the original v3 source snapshot. The archives
were SHA-256 verified after transfer; the inventory does not satisfy the
issue's requirement for raw evidence in the repository or release assets.
No release publication is authorized.

## Geant4 numerical controls (#1041)

The adapter accepts invocation-local `--electron-cutoff-kev` (default 10),
`--bins-per-decade` (7), `--step-dr` (0.2), and `--step-final-um` (1000).
The two step controls vary independently. For `--elastic gs|wvi`,
`--msc-range-factor` overrides the nominal 0.08/0.04 respectively; it is
rejected for single scattering. Invalid or nonfinite controls fail before
output creation. Photon production/scoring thresholds stay fixed.

Rebuild `g4bench` before using this adapter. After transport, its
`G4BENCH_NUMERICS` line reads back the effective cutoff, table density and
MSC parameters from Geant4. The adapter checks these and the effective step
function against the request and stores them in `bounds.numerics`.
A missing audit (including an older executable) or a silently ignored
setting fails the run. Existing retained records are unchanged.

Refined settings are automatically marked `purpose="accuracy-only"` and
excluded from `T(epsilon)` analysis. Use `--accuracy-only` also for nominal
settings run as convergence references. These controls are prerequisites for
the convergence ladder, not evidence that any ladder or cross-engine accuracy gate passes.
The design manifest below retains source-table limitations; measured numerical
gates and an executable submission manifest remain separate work under #1041.

## Frozen numerical design (#1041)

`convergence-design-v1.json` freezes **7,074** expected comparisons across
60 scene/arm configurations: six scenes, PyRITE SS/mixed scattering on CPU
FP64 and CUDA FP64/mixed precision, Geant4 SS/WentzelVI/GS, and PenRed.
The six-scene W density is **19.3 g/cm3**, distinct from the 44-case matrix.
PenRed retains its straggling-on identity and needs a separately frozen
matched family before any cross-engine equivalence comparison.

This is a numerical **design**, with `submission_ready=false`,
`numerical_gate="unmeasured"` and `headline=false`. `available` means an
adapter override exists, not that its numerical effect is qualified. Runtime
identities, effective-parameter audits, staged generated tables, same-track
replay, per-shard seed maps, resource forecasts and confirmation populations
remain explicit blockers. No scheduler submission or transport is performed.

Common positive scales come from the original v3 planning pool (seeds
101–103): PyRITE discrete transport means and continuous cone means, shared
by all arms. Headline membership is the union of the v3 plan's members across
arms; rare cones remain required and existing diagnostic exclusions remain.
The compiler verifies all four original v2 archives against the retained
inventory, then hashes every v3 input JSON and raw-score NPZ. These hashes
identify the unpublished backup; they do not satisfy durable raw publication.
No accuracy-reference or confirmation means enter scale selection.

At epsilon=2%, each absolute numerical budget is `S*0.02/3`. Four families
receive one quarter each, split equally among their active primitive controls.
Unsupported native meshes remain in the allocation and Bonferroni count;
inapplicable detector controls on transport consume no allowance and none is
reassigned. The three contrasts per primitive and the all-fine corner are
explicit. The point-cone approximation has one direct comparison against
the integrated L3 scorer, since it has no intermediate refinement levels.
All intervals use the frozen `0.05/7074` row alpha, subject to the issue's
count, VOV, tail, SHIFT and pairing eligibility rules. No interval evaluator
or measured verdict is supplied by this compiler.

```bash
# Read-only reproducibility check; no archives, tables, GPU or histories needed.
uv run python checks/geant4_performance/convergence_manifest.py --check
# Reconstruct from the original backup into a new file; compare before adoption.
uv run python checks/geant4_performance/convergence_manifest.py \
    --archives /path/to/verified/archives --output scratch/reconstructed-design.json
```

An existing different output is never overwritten. `--check` compares the
entire design with its compiler and pinned inputs, including the expected-row
list; it does not re-read the original archives. Changes require an explicit
version/amendment before confirmation, preserving failed and missing cells.
The compiler's `confirmation_count` helper implements the 1.25 variance margin,
half-allowance CI-width target, three equal repeats and 20k-per-repeat minimum.
It rejects zero/nonfinite variance; counts still require eligible pilot
evidence and forecasts under the frozen CPU/GPU ceilings.

## PyRITE numerical and table controls (#1041)

`pyrite_bench.py` accepts invocation-local `--electron-cutoff-kev`,
`--radiative-cutoff-ev`, `--elsepa-level 0|1|2|3`, repeatable
`--elsepa-table FILE.npz`, `--sbethe-table FILE.npz`, and
`--radiative-moments table|exact`. Defaults retain the frozen scene settings.
Photon scoring bands remain at their existing thresholds; the radiative
separator changes transport's soft/hard split. Numerical changes and explicit
table selections are accuracy-only and excluded from the nominal timing plan.
Use `--accuracy-only` also for nominal convergence references.

ELSEPA L1/L2/L3 insert one/two/three rounds of log-energy midpoints,
preserving all original nodes and endpoints. Each level requires fresh
generator evaluations, using the same source, target and physics options.
`pyrite.perf._benchmark_tables.nested_energy_grid` defines the exact requested
nodes. Generate those nodes with the existing `xsgen.elsepa.generate`
functions in an isolated table store before running the benchmark; transport
never launches a generator. Both free-atom and muffin-tin tables are required
where the nominal material uses both. The existing joining rule is preserved.
The native angular mesh stays fixed and remains unresolved.

The default lookup requires the exact refined keys; a missing refinement
fails rather than substituting nominal data. Explicit NPZ paths require
adjacent JSON manifests and override lookup only for this invocation. Every
explicit key must be used. The loader verifies request/manifest/payload hashes,
source/model/target identity and grid agreement. Preserve generator inputs and
raw outputs separately: these identity checks cannot prove that independently
evaluated generator data were used.

The current SBETHE generator has no independent grid-density control.
`--sbethe-table` admits a pinned regeneration of the same scientific request
on the same native grids; it rejects changed grids. Interpolating native
SBETHE or BremsLib knots onto denser arrays is not convergence evidence.
The BremsLib source mesh and ELSEPA angular mesh remain unresolved in the audit.
`--moment-level 0|1|2|3` selects the derived radiative-moment tolerance ladder:
rtol = 2e-4 divided by 1/2/4/8, with each nominal absolute tolerance
(1e-10, 1e-14, 1e-10) divided by the same factor. The node cap (32768) and
held-out probe count (8191) stay fixed. Explicit L0 is also accuracy-only;
omitting the flag preserves the nominal timing path. The private transport
override reaches the existing preparation routine and is part of its cache key.
A cap/refinement failure raises; it never substitutes nominal moments.

Diagnostic transport results record a separate model label, effective policy,
actual per-layer energy nodes and packed-buffer SHA-256. Benchmark cold/warm
work records retain distinct table identities across their blocks. The source
BremsLib data and exact photon sampler remain unchanged. `--tables-only`
records the requested policy but does not build moment tables. `exact` selects
the existing quadrature oracle and cannot be combined with `--moment-level`.
These controls enable a convergence study; they do not certify an error bound,
backend parity or a numerical gate. The explicit manifest and measurements
remain required.

`--cone-energy-order 4|8|16|32` independently sets the energy quadrature
(`2*q_E` Gauss nodes per existing edge-split panel) for continuous fixed/point
scoring. Hold it at 32 while varying `--cone-order` for angular tests; hold
`--cone-order 32` while varying energy order for energy tests. Point versus
integrated comparisons must use the same energy order and the same saved
histories. Omitting the new option preserves the coupled angular/energy rule.
Every explicit value, including 8, is accuracy-only; the record retains it in
`cone_scoring.energy_order`. Discrete, adaptive and table-only uses are rejected
before table loading or CUDA access. The v1 design predates this override;
its energy-control row remains unresolved until same-track replay evidence
exists. This option alone does not implement saved-track capture/replay or
qualify the point approximation.

## Confirmation resource forecasts (#1041)

`prepare_convergence_resources.py` converts pilot summaries into conservative
confirmation forecasts. This is a planning reference, not a submission or
numerical gate. It never runs histories, allocates seeds, submits jobs, or
changes the frozen design. Every expected row and the original multiplicity
remain present, including unsupported, unresolved and missing rows.

```bash
uv run python checks/geant4_performance/prepare_convergence_resources.py \
    --pilots path/to/pilot-summary.json --output scratch/confirmation-resources.json
```

The pilot-summary JSON uses schema
`pyrite.geant4-performance.convergence-pilot-summary.v1` and fields:

| Field | Meaning |
| --- | --- |
| `design_sha256` | Exact SHA-256 of the frozen design JSON |
| `phase`, `seeds` | `"pilot"`, nonempty unique subset of `[301,302,303]` |
| `evidence_files_sha256` | Relative file paths beside the summary mapped to SHA-256; each is verified |
| `rows` | Summaries keyed by an expected comparison's `id`; omitted rows remain missing |

Each row supplies `id`, `sampling`, `variance`, `evidence_files` and `arms`.
`variance` is per incident primary: `Var(D)` for `sampling="paired"`, or the
**sum** `Var(Y_left)+Var(Y_right)` for `sampling="independent"`. Paired scoring
uses one costed replay population; independent transport uses two costed arms,
each with the full calculated confirmation count. Pairing requires the issue's
saved-history/substream audit; the summary label alone does not establish it.
`evidence_files` is a nonempty subset of the top-level hashed paths. Hashes
verify retention integrity; the planner does not parse raw scores to certify
summary statistics, seed provenance, diagnostics or scientific identity.

Each arm supplies positive `histories`, `elapsed_seconds`, `cpus`,
`peak_memory_bytes`, and `memory_margin_factor` (at least one), plus nonnegative
`overhead_seconds`, `gpus` (zero or one) and `peak_vram_bytes`. GPU arms require
positive peak VRAM; CPU arms require zero VRAM. Explicit `hardware_identity`
and `runtime_identity` strings identify the pilot environment. Memory requests
round the observed peaks times the declared margin up to bytes.

Counts use the frozen row alpha and allowance: `ceil(1.25*4*z²*variance/b²)`,
rounded up to three equal repeats with at least 20,000 histories each.
Zero variance remains unresolved. Each repeat is split into balanced shards
under the 30-minute ceiling using pilot `elapsed_seconds/histories`; measured
overhead is **additionally** charged to every shard. This deliberately avoids
subtracting unverified setup costs from the pilot rate. CPU core-hours include
CPU allocation during GPU work. The planner charges every contrast separately;
shared-run scheduling and cost deduplication need a separate audited plan.

The aggregate is compared with the frozen 256 CPU core-hour and 8 GPU-hour
confirmation ceilings. An incomplete forecast cannot report `within-ceilings`;
known cost above a ceiling reports `over-budget` even if other rows are missing.
Pilot spend is separate. Input/source hashes and per-row evidence references
are retained; an existing output is never replaced. No measured forecast is
committed by this implementation. Actual audited pilots, collision-checked
seed/substream maps, runtime/table identities and memory qualification remain
required; `submission_ready=false`, `numerical_gate="unmeasured"` and
`headline=false` are unconditional.

## Whole-primary scorer capture/replay (#1041)

`pyrite_bench.py --estimator continuous --capture-tracks NEW_DIRECTORY`
retains the warm request's source-cone track fields in whole-primary shards.
The option always makes the sample accuracy-only. Cold/warmup and audit
requests are not captured. Capture requires the existing PXR/CBS audit; it
preserves explicit global incident-primary IDs, including primaries without
rows, local row IDs, original numeric dtypes, representative energies,
geometry and coupled-radiative metadata. Source Python, harness and table
identities are retained. Channeling-weighted/callback-based tracks are refused.

The total NPZ payload ceiling defaults to 512 MiB (`--capture-max-bytes`).
Reduce `--score-rows` and the declared population for bounded capture work;
exceeding the total ceiling fails without truncating histories. Existing
directories are refused. A partial/failed request leaves no `capture.json`
completion marker and cannot be replayed. Capture I/O is diagnostic overhead;
its wall time must never enter nominal `T(epsilon)` analysis.

```bash
# Retention audit only: no scorer/table imports, GPU or histories.
uv run python checks/geant4_performance/replay_cone.py path/to/capture \
    --audit-only --output scratch/capture-audit.json
# Explicit scorer computation on existing tracks; schedule remotely when heavy.
PYRITE_MC_BACKEND=numpy uv run python checks/geant4_performance/replay_cone.py \
    path/to/capture --output scratch/cone-replay.json
```

Replay validates the complete shard inventory, payload hashes, incident IDs,
array shapes and byte ceilings before scoring. Actual replay also requires
matching package/harness sources and matching BremsLib tables; the production
scorer checks the coupled partition. Output uses a distinct accuracy-only
schema with the capture digest, replay backend/precision, raw-score hash,
per-primary scores and signed differences for all three angular and energy
contrasts, both point candidates (energy orders 8/32), and the detector corner.
All angular tests hold energy order at 32; energy tests hold angular order
at 32. The detector corner is not the full transport all-L3 interaction check.
Zero-score incident primaries remain in every population. All-zero differences
remain statistically unresolved; no numerical verdict/interval gate is
computed. Same-byte deterministic scorer pairing does not establish transport
common-random-number pairing or CPU/GPU parity. No capture or real replay
measurement is supplied by the implementation tests.

## ELSEPA regeneration preparation (#1041)

`elsepa-refinement-plan-v1.json` retains the exact nine L1/L2/L3 requests for
the six scenes' three native tables (free-atom and muffin-tin where used).
It pins the nominal payloads, request/source/target/physics identities, angular
node counts and freshly requested nested energy grids, with joined coverage
checked down to the ladder's 1.25 keV cutoff. It preserves W=19.3 g/cm3 in the
scene family. All requests remain `requires-independent-generation`; output
payload and raw-generator-evidence hashes are null. No generator was run.

```bash
uv run python checks/geant4_performance/prepare_convergence_tables.py \
    --levels 1 2 3 --output scratch/new-elsepa-request-plan.json
```

The planner only reads installed native tables. It deduplicates requests,
preserves source/target/model options, and emits the same exact keys required
by benchmark table admission. It neither builds a generator nor writes to
catalog/table stores, and refuses existing output files. Generation must use
an isolated store and retain source/build/deck/raw-panel evidence separately.
Interpolating native knots into these requested grids cannot satisfy that
requirement. Native angular/inverse-CDF convergence remains unsupported.

For a CPU-only, read-only preflight before requesting a GPU:

```bash
uv run python checks/geant4_performance/pyrite_bench.py si-300keV-100um \
    --histories 2 --seed 301 --tables-only --electron-cutoff-kev 1.25 \
    --output scratch/table-audit.json
```

The preflight checks actual joined elastic, stopping and staged radiative
coverage over the requested electron range before importing CUDA or launching
histories. Its distinct audit schema cannot enter timing analysis. Transport
records retain effective bounds, complete table/source hashes and grid nodes.
Neither preflight success nor a successful transport run passes a numerical gate.

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
pixi run python checks/geant4_performance/precision_plan.py evidence/run/2835 \
    evidence/pilot3/2864 evidence/pilot4/JOB --output precision-plan.json
hold=$(sbatch --parsable checks/geant4_performance/hold.sbatch)
sbatch --dependency=after:$hold checks/geant4_performance/precision.sbatch $hold precision-plan.json
```

Since #1041 the Geant4 arms can run on the CPU host instead (`qlmc-queen`, the
same Ryzen 9 9950X3D as the GPU host). The `cpu` partition caps a node at 16
CPUs and jobs are not CPU-pinned, so one 16-CPU job owns that node and its
Geant4 runs see all 32 hardware threads without a hold. The PyRITE arms keep
the GPU host and its hold. `analyze.py --plan` then reads both evidence
directories:

```bash
ENGINES=geant4 sbatch -p cpu -w qlmc-queen --gres=none --mem=20G -c 16 \
    checks/geant4_performance/precision.sbatch none precision-plan.json 101
hold=$(sbatch --parsable checks/geant4_performance/hold.sbatch)
ENGINES=pyrite sbatch --dependency=after:$hold checks/geant4_performance/precision.sbatch \
    $hold precision-plan.json 101 102 103
```

#1041 v3 re-plans from held-out evidence with a variance margin, records the
trims in the plan, and adds 200 000-history continuous cone runs for the tail
slope. `analyze.py` then applies `scenes.json` `amendment_v3` (floor, skew,
tail and FOM rules; see the equal-precision protocol):

```bash
uv run python checks/geant4_performance/precision_plan.py evidence/precision/{14460,14462,14463,14464} \
    --variance-margin 1.10 --max-run-s 1800 --drop-config geant4-16t --only-at geant4-1t=0.05 \
    --tail-histories 200000 --protocol-version 3 --output precision-plan-v3.json
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

From a synced lab checkout (`PYRITE_REMOTE_DIR=~/pyrite-821/repo pixi run pyrite remote sync`):

```bash
sbatch checks/geant4_performance/pilot.sbatch
hold=$(sbatch --parsable checks/geant4_performance/hold.sbatch)
sbatch --dependency=after:$hold checks/geant4_performance/run.sbatch $hold 100000 1 2 3
pixi run python checks/geant4_performance/analyze.py evidence/run/JOB --output summary.json
```

`hold.sbatch` takes the cpu partition's 16 CPUs on the GPU host while
`run.sbatch` takes the gpu partition's 8; partition caps leave the remaining 8
unallocatable, so the host runs nothing else during measurement. Geant4 runs at
1, 16 and 32 threads (16 cores, SMT 2).

### Retained free-atom ELSEPA generation (#1041)

`generate_convergence_table.py PLAN KEY --output NEW_DIRECTORY` evaluates one
exact free-atom request from `elsepa-refinement-plan-v1.json`. Run remotely in a
bounded CPU allocation with the matching installed native tables and Fortran
toolchain. The runner validates the nested grid, native manifest/payload pins,
source digest and every deck model field before building a fresh binary. It
never selects a cached refined payload or modifies a catalog.

The new directory retains generator sources and atomic data, compiler command
and streams, executable, one fresh-process deck and raw output per energy, and
the NPZ/JSON table pair under `tables/`. `generation.json` records their hashes,
requested and actually printed panel energies, and completion only after all
panels parse and preserve the native angular mesh. Native Fortran energy
rounding and the existing parser tolerance remain explicit; no equation, deck
format or interpolation policy changes. A failure leaves incomplete evidence
without that completion record. Existing directories are never reused. The
default retained-file ceiling is 256 MiB, each generator process times out after
120 s, and the scheduler must bound aggregate runtime/memory. Compilation and
intermediate parser allocations are not hard-limited by the retained-file cap.

Muffin-tin requests are explicitly refused by this runner. Fresh free-atom
evaluations do not resolve the native angular/inverse-CDF limitation or qualify
numerical convergence. Admit the resulting explicit table pairs through
`pyrite_bench.py --elsepa-level N --elsepa-table FILE.npz` before transport.
Copy and checksum-verify the complete evidence directory into durable storage;
a generated remote directory alone does not meet raw-publication acceptance.
