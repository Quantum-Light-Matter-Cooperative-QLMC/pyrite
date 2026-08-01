# Cross-profile case reuse

Branch: `feature/cross-profile-case-reuse`
TODO scope: P2 "cross-profile checkpoint case reuse" (+ absorbs the `--no-cache`
flag scope, pending TODO reconciliation — see §7).

## 1. Problem

`cxr run <profile>` writes each named `catalog_profile`'s results under its own
checkpoint stem. `variant_stem` (`src/cxr_mc/profiles.py:315`) builds
`<material>@<profile-label>-<parameter_sha256[:12]>`, where the label is the
`catalog_profile` name and the digest covers the **full resolved sweep payload**
(`dataset_identity`, `src/cxr_mc/profiles.py:188`).

So `cxr run sub_100keV` and `cxr run sub_200keV` land in two disjoint files
(`material@sub_100keV-<h1>`, `material@sub_200keV-<h2>`) even when the two
profiles share individual cases (same energy/thickness/tilt/azimuth subset).
`run_sweep`'s resume/skip logic (`src/cxr_mc/run.py`, `resume=True` default)
only skips configs already present in the *same* pickle — it never looks at
sibling stems. Running `sub_200keV` after `sub_100keV` fully recomputes every
overlapping case instead of pulling it from cache.

Root cause: the checkpoint identity is a single hash over the *whole profile*,
and the profile **name** is in that hash. Two bit-identical cases requested by
two differently-named profiles get different keys and never share.

## 2. Design decision (approved)

**Per-case content-addressable store (CAS), sharded blobs, thin per-profile
manifests.**

- Dedup granularity: **per-case**, not per-profile.
- Storage: a **shared per-material CAS** of sharded blob files. Each unique case
  record is written once, addressed by a **content key** = hash of only the
  fields that determine its numeric output (profile name excluded).
- Each profile run keeps a thin **manifest** mapping its `(name, E0_keV)` cases
  to content keys — the pointer/address table. The manifest replaces today's
  monolithic per-stem pickle as the profile-facing artifact.
- `run_sweep` resume checks the shared store by content key instead of the
  per-stem pickle.

### Why this enables reuse

The **content key must exclude the profile label**. That single exclusion is the
whole feature: `sub_100keV`'s 100 keV case and `sub_200keV`'s 100 keV case are
bit-identical physics, so with the label out of the key they hash equal, share
one blob, and the second run reads instead of recomputes.

## 3. Content key — field sort

Key insight: **do not re-derive the key field-by-field from `Settings`/`Sweep`.**
`build_cases()` already resolves every physics-relevant field INTO each case
dict. The content key is a **canonical hash of the resolved case dict minus a
denylist** of label / perf / flux-scale fields.

| Class | Fields | In key? |
|---|---|---|
| Swept per-case coords | `thickness_ang`, `E0_keV`, `tilt_deg`, `tilt_azim_deg`, `crystal_width_mm`/`_height_mm`, `n_electrons`/`n_electrons_brem` (when swept), resolved `longitudinal_distribution` | **yes** |
| Global physics (same all cases, still determines numbers) | `material`, crystallography (`crystal`/`hkl_list`/`beam_uvw`/`surface_hkl`), transverse FWHM x/y, bunch fields, `substrate`/`stack`, `max_reflections`, `E_grid_line`/`E_grid_brem`/`E_grid_line_by_energy`, `emission`, `apply_detector_qe`, `convolve_with_det`, `brem_source`, mosaic fields, `theta_obs_deg`/`dtheta_obs_deg`, `divergence`/`energy_spread` (when set) | **yes** |
| Label (names, no numeric effect) | `catalog_profile`, `variant`, `fidelity` | **no** — THE exclusion that enables reuse |
| Perf / numerically-invariant | `spec_chunk`, `brem_chunk` | **no** — chunk-invariance goldens prove it |
| Flux scale / reporting metadata (post-multiply at read) | `beam_current_na`, `rep_rate_hz`, `bunch_charge_pc`, `domega_sr` | **no** — not in stored `spec`/`brem` arrays |

**Rule when unsure: KEEP in the key.** Over-inclusion only loses some reuse;
under-inclusion serves wrong numbers.

Three judgment calls to nail down against goldens before finalizing the denylist:

1. `mosaic_route="analytic"` re-broadens at read from one intrinsic record —
   likely *not* in the intrinsic key; `mosaic_route="mc"` broadens inside
   `mc_spectrum` and IS in the key.
2. Confirm `rep_rate_hz` / `bunch_charge_pc` / `domega_sr` are provably pure
   multipliers on `spec`/`brem` (comment in `sweep.py`/`store.py` says
   flux-metadata; verify against goldens before excluding).
3. Confirm `beam_current_na` genuinely never touches stored arrays (it's
   documented reporting metadata / current derivation).

## 4. Storage model — sharded blob CAS

Per-case blob files, sharded git-style by content-key prefix under a per-material
directory:

```
<checkpoint_dir>/<material>/<first2hex>/<content_key>.pkl
```

- 256 buckets from the first 2 hex chars → bounded dir sizes.
- "Have this case?" = compute content key, `stat` the path. No whole-file load.
- Concurrency-safe: parallel runs writing the same material touch distinct
  blob files, so no clobber of a monolithic pickle. (Still decide: atomic
  write via temp-file + rename; whether a per-blob lock is even needed given
  write-once-by-content semantics.)
- GC: a blob is dead when no manifest references it. Enables a future
  `cxr clear` that drops a profile's manifest and sweeps unreferenced blobs
  (unblocks TODO #6, checkpoint-command-rework).

### Manifest

Per profile-run artifact mapping `(name, E0_keV)` → `content_key` (+ retained
label metadata: `catalog_profile`, `variant`, `parameter_sha256` for provenance
/ dual-read). Format + path TBD by lead-task (JSON beside the CAS, or a small
manifest pickle keyed by the existing `variant_stem`).

## 5. Flags (`--no-cache` / `--recompute`) — absorbed here

Natural home: both flags gate exactly the store read/write this task builds.

| flag | read store | write store |
|---|---|---|
| (default) | yes | yes |
| `--recompute` | no | **yes** (force fresh, repopulate) |
| `--no-cache` | no | no (ephemeral) |

Mutually exclusive. Wire through `run_sweep`'s resume path (the read gate) and
the blob-write path (the write gate).

### `-p/--perf` implies `--no-cache` (absorbs `perf-flag-no-cache-defaults`)

`cxr run -p/--perf` (`src/cxr_mc/scan.py:304`) samples performance metrics. It
does NOT gate caching today, so a perf run silently resume-skips cached cases
and measures near-nothing. Fix: when `-p/--perf` is set, **force `--no-cache`
semantics** (no read, no write) so the profile measures real compute and does
not pollute the store with a measurement run. An explicit `--no-cache`/
`--recompute` on the same line stays honored (explicit wins); absent one, `-p`
defaults to no-cache.

## 6. Likely-touched files

- `src/cxr_mc/run.py` — `run_sweep` resume/skip → content-key lookup in CAS;
  blob load/save; `--no-cache`/`--recompute` gates.
- `src/cxr_mc/profiles.py` — `dataset_identity`, `variant_stem`: add the
  per-case content-key derivation (denylist hash) alongside the existing
  whole-profile digest (kept for the manifest label/provenance).
- `src/cxr_mc/scan.py` — `_resolved_run`, `_checkpoint_stem`.
- CLI entry (`src/cxr_mc/scan.py` `run` command): `--no-cache` / `--recompute`
  flags + `-p/--perf` no-cache default (flag already at `scan.py:304`), help
  text, regenerate `docs/cli-reference.md`.
- Migration: import existing `<material>@<label>-<digest>.pkl` checkpoints into
  the CAS (or dual-read old stems on miss).

## 7. Interactions / sequencing

- **`feature/checkpoint-variant-naming`** — LANDED. Introduced the
  `<material>@<label>-<digest>` @-stem (`variant_stem` docstring dated
  2026-07-29, present in this worktree). Confirmed: this design reasons about
  the real current stem format. The @-stem becomes the manifest key.
- **`feature/run-no-cache`** — the `--no-cache`/`--recompute` flags, now absorbed
  into §5.
- **`feature/perf-flag-no-cache-defaults`** — making `-p/--perf` default to
  no-cache, now absorbed into §5 (`### -p/--perf implies --no-cache`).
- Folding both here means they likely get retired/absorbed. **TODO
  reconciliation across three items — needs explicit user go before rewriting
  `main:TODO.md` or touching those branches.** Both branches are still at base
  (undiverged `run_sweep`), so no merge conflict if this task lands first.
- **`feature/checkpoint-command-rework`** (TODO #6) — BLOCKED on this design.
  The CAS + manifest + GC model above defines what `cxr clear` clears and how
  provenance is shown. This task's landing unblocks it.

## 8. Delegation

Design decided (this doc). Implementation is a `lead-task` slice — touches shared
caching architecture across `run.py`/`profiles.py`/`scan.py`, adds CLI flags, and
needs a migration path. Relevant skills: `monte-carlo`, `run-cxr-mc`,
`cli-ui-ux`, `regression-testing`.

Lead-task must still resolve, before/during implementation:
- exact resolved case-dict keys + the final denylist (verify against goldens;
  the 3 judgment calls in §3);
- manifest format + path (§4);
- atomic-write / locking decision (§4);
- migration of existing @-stem pickles (§6);
- confirm with user before absorbing/retiring the two no-cache branches (§7).

## 9. Acceptance

- Content key excludes `catalog_profile`/`variant`/`fidelity` and the perf/scale
  denylist; two profiles sharing a case produce the same content key.
- `cxr run sub_200keV` after `cxr run sub_100keV` reuses cached results for any
  case shared by both profiles instead of recomputing.
- `--no-cache` neither reads nor writes the store; `--recompute` skips read but
  repopulates; default reads+writes.
- `cxr run -p/--perf` with no explicit cache flag behaves as `--no-cache`
  (measures real compute, no store pollution); explicit flag on the line wins.
- Checked against `tests/test_run.py` (`run_sweep` resume) and
  `tests/test_profiles.py` (`dataset_identity`/`variant_stem`); existing goldens
  unchanged (bit-for-bit stored records).
- `uv run cxr-dev verify` passes; `docs/cli-reference.md` regenerated.

## 10. Implementation decisions (2026-08-01)

- `cases.json` is an additive thin reference manifest beside each profile's
  existing component checkpoint. Component artifacts remain materialized for
  compatibility with analysis, archive, prune, slim, and remote-pull consumers;
  converting those consumers to manifest-only storage belongs with the blocked
  checkpoint-command rework. Cross-profile compute reuse and future CAS GC do
  not depend on that conversion.
- CAS blobs use `_checkpoint_store._atomic_dump` (same-filesystem temporary file
  plus `os.replace`). No lock: a SHA-256 key identifies deterministic raw output,
  and independent processes use PID-qualified temporary paths.
- Existing compatible profile checkpoints seed missing CAS blobs during normal
  resume. Migration compares the requested and stored case content keys before
  writing, preventing legacy checkpoints without identity metadata from
  poisoning the shared store.
- Content keys include schema `cxr.case-content-key.v1`; SHA-256 paths are
  validated before filesystem access. Exclusions are labels (`name`, profile,
  variant, fidelity), execution chunks, pulse/current/solid-angle scale fields,
  and analytic `mosaic_fwhm_rad`. Exact-MC mosaic broadening remains keyed via
  `mosaic_mc_fwhm_rad`; RNG seed remains keyed.
- Default reads and writes CAS. `--recompute` skips reads and writes fresh blobs.
  `--no-cache` and implicit `-p/--perf` neither read nor write CAS. Explicit
  `--recompute` overrides the performance default, including Nsight re-exec.
- The two absorbed no-cache backlog branches remain untouched. Reconciliation
  still requires explicit lifecycle/TODO authority after this branch lands.

### Verification evidence

- Focused CAS/profile/CLI/reference checks: 195 passed.
- Neighboring scan/checkpoint/archive checks: 50 passed.
- `cxr-dev lint` and `cxr-dev typecheck`: passed.
- `cxr-dev verify`: 2224 passed, 40 skipped; sole sandbox failure was Python
  forkserver socket creation (`PermissionError: [Errno 1] Operation not
  permitted`). The isolated failed multiprocessing test passed outside sandbox.
- Real `cxr run --help` probe exposes both flags; incompatible flags return
  Click usage exit 2 with the documented diagnostic.
