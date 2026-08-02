# RFC: Content-addressed artifact model for grids, materials & checkpoints

- **Status:** Accepted — 2026-08-01
- **Author:** Alex Amador
- **Created:** 2026-08-01
- **Parent:** `docs/cli-redesign-rfc.md` (this is the split-off D3; the parent
  covers the surface layer — ordering, flags, job lifecycle, deprecation)
- **Depends on:** nothing in the parent blocks this; but it is sequenced **last**
  (parent §4 phase 5) so it lands behind a stabilized command surface.
- **Companion:** `docs/package-structure-rfc.md` (package/repo reorg; its
  freeze-test convention is reused for the hash-input guard in §5 Q2).

> Split from the parent CLI-redesign RFC because it is a **data-model** change,
> not a surface one. The parent restructures how commands are spelled; this
> restructures how derived data is identified and stored. The two can be
> designed, critiqued, and shipped on independent timelines.

---

## 1. Why

Today's data model couples things that should be layered, which is the root of
the "where do `profile` / `material` / `energy-grid` boundaries actually sit"
question:

- `energy-grid apply` **mutates the `standard` profile** (writes brem bounds into
  profile overrides and adds derived beam energies).
- Grids live in **two places at once**: a shared per-material derived store *and*
  per-profile `[energy_grids.NAME]` fallback buckets.
- Removal is asymmetric: adding energies via a profile is easy, but
  `energy-grid line delete` is "the only way to remove bounds from the shared
  store — editing a profile never deletes them."

The result: no clean answer to what is mutable vs immutable, no dedup, no cheap
"is my (local or remote) artifact stale?" check, and a bespoke `delete` verb
standing in for garbage collection.

## 2. Model

Adopt DVC's split ([DVC][dvc-push]): derived data is content-addressed and
immutable; the profile is the sole mutable, git-ref-like pointer.

- **Artifact = immutable, content-addressed.** `material` and `energy-grid`
  derived data are keyed by a content hash of their defining inputs
  `(material, geometry, energy, …)`. Identical inputs ⇒ identical hash ⇒ one
  stored copy (free dedup). Remote sync diffs by hash (free staleness check).
- **Profile = mutable ref.** A profile names a *set of artifact hashes* the way a
  git branch names commits. Editing a profile moves the ref; it never mutates or
  deletes an artifact.
- **Checkpoint = derived, keyed** by `artifact-hash + case params`. It carries no
  independent identity of its own.

### Consequences

- `energy-grid apply` **stops mutating `standard`.** It records/derives an
  artifact and points the target profile's ref at it.
- The dual store collapses: one content-addressed store; `[energy_grids.NAME]`
  buckets become profile refs into it.
- **No bespoke delete.** An artifact no profile references is unreachable and is
  reclaimed by `gc` (parent D4). `cxr … rm` deletes an explicit target; `cxr …
  gc` reclaims unreachable artifacts — mirroring git/DVC.

### Lifecycle verbs (parent D4, defined here)

Uniform across `material` and `energy-grid`:

- `add` — register/derive an artifact into the store.
- `verify` — recompute the hash and confirm the stored bytes match.
- `gc` — reclaim artifacts unreachable from any profile ref.

**`gc` grace window (steal git reflog).** `gc` never reclaims an artifact the
instant it becomes unreachable. Mirror `git gc --prune=<date>`: an artifact
orphaned less than a grace window ago (propose default 14 days) is retained, and
`gc --prune-all` (with the D6 destructive contract) forces immediate reclamation.
This is cheap insurance against a mis-repointed profile (open Q4) deleting data
that was reachable minutes earlier. Records the orphan timestamp in the store
metadata, not in the artifact bytes (must not perturb the hash — see Q2).

## 3. Campaign lockfile (proposed)

A `dvc.lock` / `Cargo.lock` analogue capturing `profile + resolved artifact
hashes` at run time. Makes a campaign reproducible and **citable** — high value
for a physics-validation repo where provenance is first-class. Records exactly
which immutable artifacts a published result depended on, independent of later
profile edits.

**Recommendation: adopt in the first cut** (resolves §5 Q1). For a
physics-validation repo, "which exact immutable artifacts did this published
result depend on" is the highest-value property either RFC delivers, and it is
far cheaper to emit the lockfile from day one than to reconstruct provenance for
results produced before it existed. The foundation is already in place —
`profiles.py` has deterministic serialization and SHA-256 `dataset_identity` — so
the lockfile is mostly a matter of writing the resolved `profile → artifact hash`
map at run time. A campaign without a lockfile is not reproducible; treat the
lockfile as part of "a run completed," not an optional add-on.

## 4. Migration

Deepest change in the redesign; land behind the stabilized parent surface
(parent §4 phase 5).

1. Introduce the content-addressed store alongside the current layout; derive
   hashes for existing grids/materials without changing behavior.
2. Repoint profiles at artifact hashes; keep `[energy_grids.NAME]` readable as a
   compatibility shim that resolves to refs.
3. Switch `energy-grid apply` to ref-repointing (stop mutating `standard`),
   behind a deprecation warning for the old behavior.
4. Introduce `gc`; make `energy-grid line delete` an alias that warns and
   redirects to `rm`/`gc`.
5. Drop the compatibility shims per the parent's deprecation policy (D7).

## 5. Open questions

1. **Lockfile** — ~~first cut, or follow-up?~~ **Resolved: first cut** (§3
   recommendation).
2. **Hash inputs** — exact tuple that defines a grid/material artifact hash;
   which fields are identity vs. annotation (e.g. provenance notes must *not*
   change the hash). *Recommendation:* steal DVC's normalize-then-hash — define
   one canonical byte serialization of the identity tuple, list the excluded
   annotation fields explicitly, and **freeze both with a test** (à la the
   existing `test_*_exports.py` guards). The hash-input tuple is a compatibility
   contract the moment it ships: a silent change re-hashes every artifact and
   invalidates every lockfile. Build on `profiles.dataset_identity`, which
   already excludes annotation from the digest.
3. **Checkpoint reachability** — are checkpoints roots for `gc`, or reclaimable
   when their artifact is? (parent already treats archived checkpoints + active
   manifests as CAS reachability roots — reconcile with that.) *Note:* the grace
   window (§2) softens the blast radius of getting this wrong either way.
4. **Migration reversibility** — is step 2 safely rollback-able if a repointed
   profile misbehaves? *Note:* the grace window guarantees the pre-repoint
   artifacts survive the rollback window; a repoint is reversible as long as `gc
   --prune-all` has not run.

## References

- [dvc-push] DVC push/pull/remote (content-addressed store): <https://dvc.org/doc/command-reference/push>
- DVC `dvc.lock`: <https://dvc.org/doc/user-guide/project-structure/dvcyaml-files>
