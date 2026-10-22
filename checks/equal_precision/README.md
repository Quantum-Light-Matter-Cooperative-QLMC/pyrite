# Equal-precision comparator adapter pilot (#510)

Opt-in bounded lab workload. `manifest.json` freezes six adapter-pilot scenes,
incident-primary normalization, independent pilot/measurement seeds, population
sizes and numerical controls. This is the first stage of #510, not its completed
18-scene core or an accuracy-qualified speed comparison.

No PXR/CBS scorer is called: the entry point is `simulate_trajectories`, with
coupled BremsLib soft/hard loss and creation-level hard-photon scoring. Shared
package imports still load CXR definitions. Smoke runs trace calls into CXR/line
modules during transport and fail on computation; module/class definition and
Python annotation scaffolding are excluded from that audit. Profiling is off
for measured pilot runs. Characteristic emission, secondary and photon
transport are absent. Both engines stop the primary at its first slab exit or
10 keV tracking threshold. Geant4 secondaries are killed at creation; photons
are scored from the primary step before stacking.

## Lab execution

Use a dedicated linked issue worktree. Stage with the maintained remote sync:

```bash
PYRITE_REMOTE_DIR='~/pyrite-510-benchmarks' uv run pyrite remote sync
```

The current sync file list omits `PYPI.md`, needed by the editable package
build. Copy that file to the isolated checkout as a separate setup step. On the
lab box, install the locked environment with `uv sync --extra nvidia --locked`.
Then, from the isolated checkout, submit `sbatch checks/equal_precision/run.sbatch
pilot`. This site-specific script uses the existing Geant4 11.4.2 installation
under `~/.local/opt/pyrite-issue-182/`, copied into an isolated build. To reproduce
elsewhere, follow `../full_track_bremslib/GEANT4_BUILD.md`, apply its emission
patch, then run `prepare_geant4.py` on an isolated TestEm5 source copy. The
preparation tool checks input scoring-file hashes, adjusts Si/W composition to
the manifest, kills secondaries and records exact primary terminal states.

Each allocation has a ten-minute time limit and 12 GB RAM. Pilot execution
stops after nine minutes, leaving a scheduler yield point. Re-submit the same
stage to resume; successful and failed records are retained, never overwritten.
Adapter snapshots get separate output directories. A failed scene remains an
explicit failure until investigated; changing code creates a new snapshot.
The initial measurement budget is three bounded pilot allocations; preliminary
adapter/build debugging is retained separately. The live GPU partition caps
CPU allocations at eight and rejects whole-node `--exclusive`; this pilot uses
one requested CPU and one generic GPU GRES. It does not establish exclusive
host ownership or enforced physical-core affinity.

## Warmup and timing

Every PyRITE process performs **one untimed, identical-workload transport
warmup**, with seed offset +1,000,000. It then discards those trajectories and
reruns the requested measurement seed. Scene, history count, tables, transport
core and numerical controls stay identical. CUDA synchronizes after warmup and
at both ends of the measured interval. This warms in-process JIT, table-packing
and allocator caches; no computed-result cache or checkpoint is read. CPU uses
the same warmup policy.

`transport_and_emission` measures the warmed transport/emission call only.
`warmup` and `imports_and_tables` are separate. `.wall.json` measures the entire
fresh adapter process, including setup, warmup, scoring and serialization;
those values are not warmed compute-only times. Geant4 currently reports a
fresh-process interval including its diagnostic logging. This pilot therefore
supports no equal-timing-boundary speed ratio. No controlled cold-cache claim
is made. Numerical refinement, logging-overhead calibration, exclusive-resource
measurement and matched Geant4 warmup remain gates for the comparison stage.

## Evidence and analysis

Every successful record has raw per-primary `.npz` scores and their SHA-256;
Geant4 additionally retains the macro and full log, executable hash and converted
production cuts. SBETHE/ELSEPA/BremsLib table identities and digests are recorded.
Each job saves hardware, allocation and remote code-sync evidence.

Use the project environment to summarize a single snapshot after pulling it:

```bash
uv run python checks/equal_precision/analyze.py PATH/TO/evidence/pilot/SNAPSHOT \
  --output /tmp/equal-precision-summary
```

Raw scores include zero-score histories; photons from one primary are summed
before variance estimation. Replicas within a manifest/scene/arm/population group
must contain identical observable fields, including `photons_ge10kev`; incomplete
evidence is rejected before summaries are written. Text artifacts use UTF-8.
Integrated source-yield moments include the
within-primary covariance of energy bands. Independent replicas provide an
additional SE diagnostic. Zero observed variance never qualifies a target.
The same seeds at different history counts produce correlated curve points;
never pool those population sizes together. Seed integers across engines do
not pair histories. The output curve is diagnostic source-yield precision,
not qualified `T(ε)`. All numerical/equivalence gates remain unmeasured, and no
speed ratio is emitted. See the maintained [pilot report](../../docs/validation/beam-transport/equal-precision-pilot.md).

## Representative population extension

`scaling.json` freezes the additional user-directed 100000-history stage.
Use a separate isolated remote checkout and submit
`sbatch --mem=24G checks/equal_precision/run.sbatch scaling`.
Its budget is two ten-minute allocations. It uses CUDA/Geant4 on the two
300-keV anchors, a 5000-primary PyRITE block limit, 4096 MiB scratch and no
initial short probe. Global counter indices preserve independent primary
streams across blocks; scoring retains only per-primary values. Warmup runs
the complete request with identical batching. `request_with_reductions`
includes all block calls/reductions; `transport_and_emission` sums synchronized
transport intervals. Preserve both. Analyze `evidence/scaling/SNAPSHOT` just
as for the pilot. History counts remain equal-population diagnostics until
accuracy and precision gates establish the required count for each engine.
The event source tally does not stand in for the pending production weighted
per-segment estimator arm.

## Diagnostic profiling and deadline failures

`profile.sbatch` is a separate three-minute allocation for Nsight Systems and
cProfile on 1000 W histories; profiler timings never enter the comparisons.
Raw trace and host profile are retained, while the derived SQLite cache is
omitted from the evidence archive. Sample deadlines stop the launcher's
process group, including `uv` children. Failed-attempt markers override any
late result file; a retry gets a unique attempt filename. Budget deferrals
remain in job logs and incomplete replicate counts remain explicit.

## Geant4 single-scattering follow-up

`sbatch checks/equal_precision/single-scattering.sbatch` runs `emstandardSS`
with the same instrumented executable, scene manifest, scoring and cut settings.
The CPU-only allocation uses one worker and a ten-minute scheduler limit;
`run_single_scattering.py` enforces a nine-minute driver budget and process-tree
sample deadlines. It first retains two 1000-history preflights, then requests
three 100000-history replicas per anchor with the scaling seeds. Resume with
the same command; completed evidence is immutable. Preflight populations stay
separate from the large runs. Geant4 times include a fresh native process,
table setup and per-primary logging, as in the multiple-scattering baseline.
No new CUDA runs are required; retained CUDA comparisons have complete matching
warmups and synchronized transport intervals. The analyzer labels this physics
arm `geant4-emstandardSS` so it cannot pool with `empenelope` samples.

`sbatch checks/equal_precision/work-audit.sbatch` is a separate three-minute
GPU diagnostic on 1000 histories per anchor. Its wrapper preserves all adapter
physics and records the warmup/measured request segment counts, terminal counts,
path lengths, and returned-array byte sums. Counts exclude capacity replays;
array byte sums may include aliases and are not peak RSS/VRAM. These diagnostic
timings never enter the 100000-history benchmark table.

## Kernel-cost diagnostics (2026-10-15)

`kernel-diagnostic.sbatch` and `kernel-scaling-diagnostic.sbatch` each cap a
separate lab allocation at five minutes, one CPU, one GPU and 24 GB RAM.
`diagnose_kernel.py VARIANT AUDIT_JSON` accepts the remaining arguments of
`pyrite_adapter.py`. It changes only that diagnostic process: `nodes` replaces
successive-node scans with binary search; `both` additionally replaces the
SDCS row scan; `capacity` starts at 12500 segment slots; combined variants
apply both, optionally with 32 threads per CUDA block. This capacity is an
initial guess, never a hard bound. Capacity replay remains enabled.
`threads32` changes only CUDA launch geometry. `radiation-off` disables coupled
radiative transport, and explicitly changes the physical workload; its zero
photon columns are placeholders, not source-yield predictions.

Each process runs the same full-request warmup as the frozen adapter. Added
synchronized launch timers identify complete-history replays. All returned
trajectory arrays are hashed outside the transport timer. The analysis checks
every numerical/control variant's warmup and measured hashes against the same
scene/population/seed baseline; radiation-off is excluded from that equality
test. Instrumentation overhead is uncalibrated, so these records stay separate
from comparative benchmark summaries. The resource diagnostic retains generated
CUDA source, register/local-memory attributes and runtime device properties.

Job 1312 ran three seeds at 1000 histories per scene/variant; job 1313 ran two
seeds at 5000. Both finished all requested samples. The latter used the same
script staged as `diagnose_kernel_resources.py` to preserve job 1312's source.
Exact source snapshots, all 50 samples/logs/tallies, generated CUDA, and scheduler
records are retained in [the kernel diagnostic archive](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/releases/download/evidence-510-v1/kernel-diagnostic-evidence-2026-10-15.tar.gz).
To regenerate the summary after extracting the archive:

```bash
uv run python checks/equal_precision/analyze_kernel_diagnostic.py \
  evidence/kernel-diagnostic/1312 evidence/kernel-diagnostic/1313 \
  --output /tmp/kernel-diagnostic-summary.json
```

See the pilot report's kernel-cost investigation for measured effects and limits.

## Exact radiative integration optimization

`radiative-optimization.sbatch` compares the production candidate with a
retained copy of the pre-change `_jit_radiative_device.py`, staged beside the
adapters as `radiative_baseline.py`. For the measured source pin, retrieve it
without changing the current checkout:

```bash
git show ba8bb3f2:src/pyrite/montecarlo/transport/_jit_radiative_device.py \
  > checks/equal_precision/radiative_baseline.py
```

Expected baseline SHA-256:
`5c1c6f0c45ad47a92d49c8f1d8d6bf933dfc06ffb09f6c004a4d505e3cdf69e1`.
This generated baseline is an execution input, not another maintained source
implementation. The immutable evidence includes both exact source versions.
`benchmark_radiative.py` loads the selected baseline before importing any CUDA
transport kernel; the candidate uses the installed production module. Each
fresh process runs the frozen adapter through the existing diagnostic wrapper,
retaining identical warmups, full trajectory-array hashes and synchronized
launch timers. Both use baseline launch/capacity settings: no block-size,
scratch, physics, numerical-step or initial-capacity changes enter this comparison.

The five-minute pilot allocation runs the GPU radiative regression module,
then three seed pairs (56001–56003) at 5000 primaries per Si/W anchor. The
separately frozen `radiative-scaling.json` bounds two ten-minute allocations
at 100000 primaries and seed 57001. Submit `radiative-scaling.sbatch baseline-w`,
then `radiative-scaling.sbatch remaining`, sequentially. Large populations have
one fresh-process pair per scene; precision of a speed ratio cannot be inferred
from that single pair. Warmup covers the complete requested population.
The scaling driver reuses the process-group deadline helper from `run_pilot.py`;
deadline failures and budget deferrals are retained explicitly. Never summarize
a late file with a matching failure marker. Regenerate summaries with
`analyze_kernel_diagnostic.py` on the selected evidence directories, keeping
5000 and 100000 populations separate.

## Retained evidence storage

Raw evidence archives are hosted as assets of the
[versioned evidence release](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/releases/tag/evidence-510-v1).
The [manifest](results/manifest.json) records each download URL, byte size, and
SHA-256; [SHA256SUMS](results/SHA256SUMS) verifies downloads from their containing
directory with `sha256sum --check SHA256SUMS`. Summaries and plots remain in Git.

Download and unpack archives outside this checkout, for example under
`~/pyrite-evidence/510/`. Remote code sync includes `checks/`, so keep working
evidence outside that tree as well. Never replace an existing release asset
with different bytes; publish a new evidence version and update the manifest.


## Radiative precision experiment

This opt-in diagnostic leaves production source/defaults unchanged. See the
[measured report](../../docs/repo-design/compute/transport-precision-experiment.md)
and frozen `precision-experiment.json` for precision boundaries and limitations.
`precision_source.py` generates isolated FP64, FP32/52, FP32/24, mixed-cell/24
and stable-cell FP32/24 helpers; no production precision switch is added.

Stage with maintained `pyrite remote sync`, then use bounded GPU allocations
from the staged lab checkout (never execute GPU pilots locally):

```bash
preflight=$(sbatch --parsable --time=00:05:00 checks/equal_precision/precision.sbatch preflight)
si=$(sbatch --parsable --dependency=afterok:$preflight checks/equal_precision/precision.sbatch si evidence/precision/$preflight)
sbatch --dependency=afterany:$si checks/equal_precision/precision.sbatch w evidence/precision/$preflight
```

Pilot stages require six-scene trajectory preflight plus finite/nonnegative/
support and actual 1 keV-cutoff component checks. Endpoint stress failures remain
explicit unqualified diagnostics. Each process has a deadline; failures and
budget deferrals remain in the evidence. Keep raw evidence outside the source
checkout when linting. After pulling completed job directories, CPU analysis is:

```bash
uv run python checks/equal_precision/analyze_precision.py \
  /path/to/1343 /path/to/1344 /path/to/1345 \
  --output /tmp/precision-summary.json
```

Do not pool initial failed job 1339 with corrected job 1343. The analyzer rejects
source/control mismatches and duplicate replica seeds. These diagnostic samples
use `pyrite.equal-precision.precision-sample.v1`; the standard benchmark analyzer
does not treat them as production performance evidence. Ratios remain unqualified
until independent ensemble/endpoint accuracy gates pass.


### Endpoint coordinate follow-up

`precision-endpoint.json` freezes FP64, stable FP32 and `rad-boundary64-24`
comparisons. The new arm retains FP64 coordinate geometry while computing
radiative SDCS/cell work in FP32. Production precision stays unchanged.
`precision-endpoint.sbatch` uses the same allocation limits as the original
experiment, captures the submitted batch script and writes to
`scratch/endpoint/<job-id>`. Submit preflight with a five-minute limit; pass its
directory to Si/W jobs. Use `afterok` dependencies between all three stages.
Preflight fails unless all required variants pass; the boundary arm must also
pass the complete endpoint stress screen before timing. The stable FP32 arm is
an explicitly unqualified diagnostic control.

Analyze completed evidence separately from the earlier protocol:

```bash
uv run python checks/equal_precision/analyze_precision.py \
  /path/to/preflight /path/to/si /path/to/w \
  --protocol /path/to/preflight/source/precision-endpoint.json \
  --output /tmp/endpoint-summary.json
```

The analyzer verifies the selected protocol hash before pooling. Use the captured
protocol file for historical results; its byte hash includes line endings. For CPU-only
arithmetic diagnosis using the retained job-1343 reference probes:

```bash
uv run python checks/equal_precision/probe_precision_cpu.py rad-boundary64-24 \
  /path/to/1343 /tmp/endpoint-cpu.json
```

This NumPy execution removes device decorators/imports, verifies input table and
reference-source pins, and retains per-case arrays. It does not validate CUDA
compilation/math, full trajectories, ensemble equivalence or GPU performance.


### Selected quotients and paired equivalence

`precision-quotient.json` and `precision-quotient.sbatch` freeze the next study.
See the [method and reproduction](../../docs/validation/beam-transport/transport-precision-equivalence.md).
The same block/process-deadline runner supports `heldout-si` and `heldout-w`,
with explicit `--replica-start` / `--replica-stop` slices. Analyze fixed held-out
populations separately from pilot diagnostics. `analyze_equivalence.py`
requires all nine held-out directories, the captured protocol and preflight;
its paired-primary intervals preserve uncertainty even at zero differences.
Source/input/provenance mismatches raise errors; missing or failed fixed replicas
prevent qualification. No production precision option is introduced.

The completed interval family can also be audited against issue #510's stricter
one-third numerical budget without rerunning transport. `audit_precision_budget.py`
uses the independent frozen planning scales at 5%, 2% and 1% target RSE, keeps
observed RSE separate, and leaves zero planning means without absolute floors
unresolved. See the method page for the command, outcomes and planning limits.

### Small numerical/backend validation

`cheap-validation.json` freezes the small production-FP64 step/cutoff and CPU/GPU
pilot. `cheap-validation.sbatch` separates CPU/GPU allocations with explicit
5/10-minute submission limits. `benchmark_validation.py` wraps the unchanged
adapter with trajectory/domain/control audits; `analyze_cheap_validation.py`
requires the complete fixed grid before issuing paired intervals. A separate
`probe_step_activity.py` / `step-activity.sbatch` measures actual energy-limited
substeps with a two-minute CPU cap. See the
[method, results and commands](../../docs/validation/beam-transport/cheap-numerical-validation.md).
Small samples and inactive controls must not be reported as convergence.

`activity-ladder.json` freezes a separate 24-request CPU activity design:
all six scenes, fresh seed 69001, 64 primaries, and caps 0.05, 0.001,
0.00025 and 0.0000625. Submit `sbatch checks/equal_precision/activity-ladder.sbatch`
from an isolated staged checkout. It requests one CPU, 8 GiB RAM and five
minutes; the runner stops after 260 seconds and caps each process tree at
30 seconds. Every sample includes full-workload warmup, trajectory checks and
the PXR/CBS execution audit. Source snapshots, raw scores, failures/deferrals
and an explicit incomplete-grid outcome are retained. Existing output
directories are rejected. This planning population is separate from all
convergence/equivalence families; nonzero energy-limited counts show an active
control, not numerical convergence.
`activity-capacity.json` and `activity-midpoint.json` freeze separate six-scene
follow-ups with an explicit 80000-step limit; pass either path as the batch
script's first argument. Earlier step-limit failures remain in the original
grid. The adapter's default history limit stays 20000. See the
[active-control results](../../docs/validation/beam-transport/cheap-numerical-validation.md#active-refinement-levels-and-the-history-step-limit).

## Full matrix freeze and coverage audit

`matrix-spec.json` defines the 18-scene core, the six pilot anchors and the
sparse extensions; `build_matrix.py` expands it to the committed `matrix.json`
(stable case ids, observable applicability, coverage table). Range-scaled
thicknesses come from the committed ESTAR CSDA tables in
`checks/reference/estar/`, pinned in the spec; `--estar --range MATERIAL=PATH`
overrides them. No engine's own range is used.
`audit_coverage.py --output PATH` resolves the installed released tables for
every case without transport and keeps every gap explicit. The comparison
schema for all adapters is `pyrite.perf.comparison`. See the
[protocol page](../../docs/validation/beam-transport/equal-precision-protocol.md)
and the [comparator notes](../../docs/research/beam-transport/equal-precision-comparators.md).
