# Cross-profile case reuse

Branch: `feature/cross-profile-case-reuse`
TODO scope: new P2 item, "cross-profile checkpoint case reuse."

## Problem

`cxr run <profile>` writes each named `catalog_profile`'s results to its own
checkpoint stem: `variant_stem` (`src/cxr_mc/profiles.py:315`) builds
`<material>@<profile-label>-<parameter_sha256[:12]>`, where the label is the
`catalog_profile` name itself and the hash covers the full resolved
`settings`/`sweep` payload (`dataset_identity`, `src/cxr_mc/profiles.py:188`).

So `cxr run sub_100keV` and `cxr run sub_200keV` land in two disjoint files
(`material@sub_100keV-<h1>`, `material@sub_200keV-<h2>`) even when the two
profiles share individual case configs (same energy/thickness/tilt/azimuth
for some subset of cases). `run_sweep`'s resume/skip logic
(`src/cxr_mc/run.py:320`, `resume=True` default) only skips configs already
present in *that same* pickle — it never looks at sibling stems. Running
`sub_200keV` after `sub_100keV` fully recomputes any overlapping case instead
of pulling it from cache. Confirmed via `tokensave_context` read of
`dataset_identity`/`variant_stem`/`run_sweep`, not yet reproduced end-to-end.

## Implementation path (not yet decided — investigation task)

Likely-touched files: `src/cxr_mc/run.py` (`run_sweep`, checkpoint load/save),
`src/cxr_mc/profiles.py` (`dataset_identity`, `variant_stem`), `src/cxr_mc/scan.py`
(`_resolved_run`, `_checkpoint_stem`).

## Open questions

- Which resolved-identity fields are physics-relevant per case (energy,
  thickness, tilt, azimuth, detector geometry, fidelity, emission mode) vs.
  profile-label-only (i.e. same case would produce bit-identical output under
  either profile)? Need this split before any dedup key can be defined.
- Dedup granularity: reuse only when two profiles hash identically (coarse,
  no new logic) vs. reuse individual matching cases across differing
  profiles (fine-grained, needs a case-content key independent of profile
  label/hash).
- Storage model: keep one pickle per stem and add a lookup step that scans
  sibling stems for matching case keys before recomputing, vs. a shared
  per-material case store keyed by case-content hash with per-stem files
  becoming thin manifests/views into it.
- Interaction with in-flight `feature/run-no-cache`
  (`tasks/feature/run-no-cache/`), which also modifies `run_sweep`'s
  load/resume path (the opposite direction: forcing *no* reuse). Sequence or
  coordinate so the two don't land conflicting refactors of the same
  function.
- Interaction with `feature/checkpoint-variant-naming` (stem scheme this
  reasoning is based on) — confirm that item's current landed state before
  building on `variant_stem` details here.

## Delegation

Design/investigation only for now — no implementation slice cut yet. Once the
dedup key and storage model are decided, this likely needs `lead-task` tier
(touches shared caching architecture across `run.py`/`profiles.py`/`scan.py`
and interacts with two other in-flight checkpoint items). Relevant skills:
`monte-carlo`, `run-cxr-mc`.

## Acceptance

- Design phase: a written proposal here for the dedup key + storage model,
  checked against `tests/test_run.py` (`run_sweep` resume tests) and
  `tests/test_profiles.py` (`dataset_identity`/`variant_stem` tests) for
  compatibility.
- Implementation phase (separate dispatch once designed): `cxr run
  sub_200keV` after `cxr run sub_100keV` reuses cached results for any case
  shared by both profiles instead of recomputing; `uv run cxr-dev verify`
  passes.
