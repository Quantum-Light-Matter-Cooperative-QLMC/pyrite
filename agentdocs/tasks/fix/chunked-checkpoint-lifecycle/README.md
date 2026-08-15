# Chunked checkpoint lifecycle and truthful progress

Branch: `fix/chunked-checkpoint-lifecycle`

## Problem and evidence

Chunked remote sweeps can spend more wall time decoding and rewriting checkpoints
than computing cases, while the dashboard presents the interval as a running or
apparently hung simulation.

Observed on `qlmc` (RTX 5080), profile `hopg_hbn`, 2026-08-14:

- HOPG computed 5,508 cases in 276 s, then serialized `brem.pkl` (540 MB) for
  about 95 s and `line.pkl` (530 MB) for about 92 s: roughly 187 s of finalization.
- h-BN computed 2,824 cases in 137 s before its slice deadline, then serialized
  two roughly 250 MB components for about 47 s each: roughly 94 s of finalization.
- A replacement job with fully cached HOPG spent about 155 s decoding the
  1.07 GB checkpoint, reported 5,508 cached cases, then began rewriting both
  components despite having no new shards. It remained in the HOPG subprocess
  after 4m46s; no h-BN simulation had started. Allocation 1634 was cancelled.
- During these phases one host core was saturated (about 3-4% node CPU), GPU use
  was 0%, and the redirected log remained buffered. The dashboard showed
  `running`, initially regressed to `0/5508`, and labelled the last completed
  case as `NOW TESTING`.

The immediate causes are:

1. `runs.run.run_sweep` consolidates every component-directory run after the
   case loop, including budget stops and fully cached runs with no `parts/`.
2. Consolidation rewrites the full generic HDF5 mapping twice, sequentially,
   through `checkpoints._checkpoint_store.save`.
3. Resume eagerly decodes all component arrays before it can prove a material
   is complete.
4. Shard-aware recovery exists in `runs.run`, but checkpoint discovery,
   `checkpoints._checkpoint_store.load`, remote partial pull/slim, and archive
   paths still treat component monoliths as authoritative existence.
5. The scan deadline is converted back into a fresh relative deadline after
   setup/resume work; setup and finalization are therefore outside the intended
   soft slice budget.
6. Progress JSON is initialized with zero counts before resume filtering;
   `current` is written from the post-case callback; consolidation exposes no
   activity; redirected Python output is buffered.

Relevant design history:

- [`docs/repo-design/compute/compute-performance-optimization.md`](../../../../docs/repo-design/compute/compute-performance-optimization.md)
  introduced incremental shards but deliberately consolidated on both clean
  completion and budget stop to preserve older consumers. The measured estimate
  there (1-2 s consolidation) no longer describes the current generic HDF5
  representation or this 5,508-case workload.
- [`docs/repo-design/compute/gpu-transport-rawkernel.md`](../../../../docs/repo-design/compute/gpu-transport-rawkernel.md)
  defines the current single-process CUDA execution topology; zero GPU use in
  these intervals is checkpoint work, not a transport scheduling stall.

## Scope

Own the checkpoint state transition used by local and remote material sweeps:

```text
loading/resuming -> computing -> paused with shards
                              -> complete -> one consolidation
```

Also own the progress/status representation of those transitions. Preserve:

- atomic authoritative component replacement;
- crash recovery with shards winning over stale component records;
- exact dataset-identity checks and case-level cache semantics;
- partial pull, slim, archive, cleanup, analysis, and recompute compatibility;
- exit `75` for resumable budget exhaustion and current chunk handoff safety;
- existing progress JSON fields for backward-compatible readers.

Non-goals:

- physics, RNG, spectrum, or transport changes;
- a wholesale result-format migration or renaming `.pkl` to `.h5`;
- tuning HDF5 compression without a representative A/B measurement;
- changing scheduler priority or the default 10-minute yield policy.

## Implementation path

### 1. Regression baseline and lifecycle contract

- Add tests proving that a fully cached component checkpoint performs neither a
  component rewrite nor a case launch.
- Add budget-stop tests proving shards remain durable and resume correctly over
  a stale monolith or with no monolith.
- Cover shards-only and component-plus-shards behavior through storage discovery,
  slim/partial pull, archive, cleanup, manifest signatures, and final consolidation.
- Record phase timings separately for resume load, case compute, shard writes,
  consolidation, and handoff. Retain the 2026-08-14 measurements as the baseline.

### 2. Make shards first-class incomplete checkpoint state

- Move component-plus-shard union semantics into the checkpoint storage owner,
  rather than keeping a special recovery path only in `runs.run`.
- Teach `checkpoint_exists`, discovery, signatures, load, slim/pull, archive, and
  remote checkpoint listing to recognize atomic shard files.
- On budget pause, leave shards in place and do not rewrite components.
- On true material completion, consolidate exactly once, publish components
  atomically in the existing brem-then-line order, refresh metadata, then clear
  shards.
- Skip consolidation entirely when no shards exist. A stale monolith plus newer
  shards must still resolve with shards winning per `(name, E0_keV)`.

### 3. Fast proof for fully cached material

- Add a compact, versioned completed-case-set digest/index to checkpoint metadata.
  It must bind the exact requested case identities, not infer completeness from
  record count alone.
- When dataset identity and completed-case proof match the requested sweep, emit
  cached progress and return complete without decoding spectrum arrays.
- Fall back to the current full load for legacy/missing/stale metadata; never turn
  uncertain metadata into a false cache hit.
- Measure the fast path on the 5,508-case HOPG checkpoint. Target setup is seconds,
  not the observed ~155 s full decode, with zero component writes.

### 4. Budget the whole slice

- Thread one absolute deadline from scan orchestration through resume, compute,
  finalization, and shell handoff; do not restart a relative deadline after setup.
- Check the deadline before beginning expensive optional work. Completion
  consolidation may cross the soft deadline only under an explicit, tested rule;
  incomplete slices must remain safely resumable without it.
- Preserve the SLURM hard backstop and exit-75 self-resubmission contract.

### 5. Truthful progress and logs

- Extend progress JSON with an optional backward-compatible activity value such
  as `loading`, `computing`, `saving`, or `handoff`; keep terminal state separate.
- Preserve the previous valid counts while a new slice loads its checkpoint,
  then replace them with authoritative resume-filtered counts. Never regress a
  known partial checkpoint to zero merely because setup was interrupted.
- Populate `current` from case-start activity. Store/display the post-case value
  as `last_completed` if useful; do not label it `NOW TESTING`.
- Clear the active case and show checkpoint finalization/handoff explicitly.
- Make remote scan output unbuffered so diagnostic phase lines reach the log
  during execution. Keep machine JSON clean and progress on the correct stream.

### 6. Documentation and measured closure

- Update the compute-performance design record with the measured HDF5 baseline,
  corrected shard lifecycle, commands, backend, cache state, and before/after
  results.
- Regenerate the CLI reference only if public help/output contracts change.
- Run a small real remote chunked smoke after fast tests pass; do not run a heavy
  sweep locally.

## Decisions

- One task owns storage and status integration: separating them would allow the
  UI to claim lifecycle states that storage cannot yet guarantee.
- Incomplete state is components plus immutable shards, or shards only. It is
  not required to publish fresh component monoliths at each scheduler yield.
- Full-cache fast paths require an exact case-set proof plus dataset identity.
  `n_records == len(cases)` alone is insufficient.
- Existing progress fields remain valid. New activity/current semantics are
  optional fields so older status readers degrade to counts and state.
- HDF5 schema redesign is deferred. First remove redundant reads/writes; profile
  the one remaining completion consolidation before considering format work.

## Bounded manifest-rescan slice (2026-08-14)

The production trace identified an independent O(N^2) cost inside the existing
shard lifecycle: every completed config called
`_manifest_save(checkpoint_path, _material_subset(), ...)`, rebuilding sweep
values and record counts from all accumulated results. Shard publication now
keeps a run-local manifest accumulator. It scans resumed results once, then adds
only each newly completed config; partial manifests intentionally omit the exact
completed-case proof, which the final authoritative consolidation publishes.
The metadata-only completion path already refuses proofs while `parts/` exists.

A three-repeat synthetic reproduction of 1,836 configs with three energies each
measured 6.399--6.479 s for repeated full scans and 0.0300--0.0302 s for the
incremental accumulator (about 213x for manifest construction). The focused
regression asserts that a complete sharded sweep performs one full manifest scan,
at final consolidation, instead of one per config plus consolidation.

Real `qlmc` confirmation used the same full, uncached 5,508-case hopg workload,
parameter digest `4135cde714d5`, RTX 5080 backend, and performance sampler. Job
`hopg_hbn-12` / SLURM `1644` is the pre-change baseline; `hopg_hbn-15` / SLURM
`1647` is the candidate:

| metric | baseline | candidate | delta |
| --- | ---: | ---: | ---: |
| checkpoint cumulative | 45.581 s | 28.981 s | -36.4% |
| final consolidation | 2.665 s | 2.727 s | +0.062 s |
| case-loop/session elapsed | 161.330 s | 131.513 s | -18.5% |

The session delta also includes the independently measured shared line-table
optimization; the checkpoint counters isolate 16.60 s of this slice's gain.
The complete `line.pkl` and `brem.pkl` SHA-256 hashes match the baseline. Residual
checkpoint time is about 26 seconds across 1,836 immutable shard writes; changing
that durability granularity is a separate lifecycle decision, not part of this
bounded fix.

## Open questions for implementation owner

- Choose the completed-case proof encoding (sorted identity digest versus a
  compact explicit index) after checking legacy and exotic case-key behavior.
- Decide whether live partial pull reads a stable shard snapshot directly or
  first creates a job-local transfer snapshot. It must not race an in-progress
  shard publish or mutate the active checkpoint.
- Define whether completion consolidation may exceed the soft deadline, or
  whether the next slice owns it as a distinct `saving` phase. Either choice
  must keep reservations and exit codes correct.
- Benchmark shard-count resume overhead. If thousands of small HDF5 shards are
  themselves slow to enumerate/decode, add an index or bounded compaction tier;
  do not restore per-slice full component rewrites.

## Delegation slices

1. **Storage lifecycle and tests** — owner: `lead-task` + `performance` +
   `remote-gpu-jobs`; includes all checkpoint consumers and measured baseline.
   Not `one-shot`: it must resolve the transfer snapshot and completion-boundary
   questions above before landing.
2. **Progress schema/rendering** — owner: `implement-task` + `cli-ui-ux` +
   `regression-testing`; may proceed after the lifecycle/activity vocabulary is
   fixed. Self-contained enough for Serena `one-shot` at that point.
3. **Remote integration and measured closure** — owner: `lead-task` +
   `run-cxr-mc` + `remote-gpu-jobs` + `performance` +
   `documentation-maintenance`; integrate both slices and run the real smoke.

No slice may push, edit `TODO.md`, retire task records, or delegate further
unless the dispatcher explicitly grants that authority.

## Acceptance checks

- Fully cached HOPG-equivalent run:
  - reports all cases cached without loading spectrum arrays;
  - launches zero cases;
  - writes zero component or shard artifacts;
  - does not change checkpoint mtimes.
- Budget-paused run:
  - exits 75 with durable atomic shards;
  - performs no full component rewrite;
  - resumes without recomputing completed cases;
  - preserves exact records and identity after eventual consolidation.
- Shards-only and stale-components-plus-shards checkpoints work through load,
  manifest, discovery, slim/partial pull, archive, cleanup, and analysis-facing
  APIs; shards win on overlap.
- Exactly one full consolidation occurs on true completion and clears `parts/`
  only after both authoritative components and metadata are safely published.
- A nominal 10-minute slice accounts for setup and finalization under the chosen
  tested completion rule and preserves self-resubmission behavior.
- During resume/finalization, status says `loading checkpoint`/`saving checkpoint`,
  retains prior counts, shows no stale active case, and emits live unbuffered logs.
- During compute, `NOW TESTING` (or its replacement label) represents a case that
  has started; the completed frontier is not presented as active.
- Focused checks:

  ```bash
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/scan/test_run.py
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/scan/test_scan_budget.py
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/remote/test_remote.py
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/checkpoint/test_slim.py
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite cli
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev lint
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev typecheck
  UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev docs
  ```

- Real remote smoke records command, profile, case count, cache state, slice
  duration, resume/finalization/compute timings, CPU/GPU use, artifact mtimes,
  and before/after wall time. Heavy sweeps remain remote-only.
