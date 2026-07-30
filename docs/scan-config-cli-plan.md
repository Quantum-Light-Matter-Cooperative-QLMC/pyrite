# Checkpoint variant naming and multi-config checkpoints

Status: **implemented** (naming migration landed 2026-07-29 on
`feature/checkpoint-variant-naming`). Decisions locked 2026-07-29 supersede the
2026-07-26 draft. Owner: Alex.

This document describes how differently-configured runs of the same material are
kept in separate checkpoints and how each is named, discovered, and pulled. It
replaces an earlier draft that proposed a root-level override file and a
`cxr config` subcommand; that surface was **retired** in favor of the
`catalog_profile` / `data/materials.toml` mechanism that shipped instead (see
"Retired design" below).

## Decisions (locked 2026-07-29)

1. **Checkpoint naming — @-stems.** A non-canonical run writes
   `checkpoints/<material>@<label>-<digest>` (component dir) /
   `<material>@<label>-<digest>.pkl` (legacy flat). The label names the run's
   non-standard `catalog_profile` (else its fidelity); the 12-hex
   `parameter_sha256` prefix guarantees two runs that share a label but differ
   in resolved parameters never collide on disk. The canonical `full`/`standard`
   run keeps the bare `<material>` stem and `--quick` keeps `<material>_quick`.
   This replaces the shipped-but-opaque `<material>--<fidelity>-<hash>` scheme,
   whose stem named no profile at all.
2. **No separate override surface.** The `cxr config` subcommand,
   `cxr-overrides.toml`, and a `cxr scan --preset` flag are **dropped**.
   Per-material and per-campaign reconfiguration is expressed as a
   `catalog_profile` in `data/materials.toml` (immutable `CATALOG`), which is
   also what the digest and the `@`-stem label already key on. One mechanism,
   not two.

## Motivating gap (still valid)

Scan grids are projections of the immutable `CATALOG` (`config.material_sweep`).
Before per-variant stems the checkpoint store keyed records only by a config
name (`"{label} {thickness} pol=.. az=.."`) that did **not** encode energy
grids, `n_families`, `beam_uvw`, electron counts, coherence, or the
`catalog_profile` — so two differently-resolved parameter-sets run into one
pickle would silently resume-skip or mix records. Per-variant stems, each keyed
by the full `dataset_identity` digest, close that hole.

## Naming scheme (@-stems)

`profiles.variant_stem(identity, canonical_full=...)` maps a resolved
`dataset_identity` to its on-disk stem:

| Run | Stem |
|---|---|
| canonical `full` + `standard`, no overrides | `<material>` |
| `--quick` | `<material>_quick` |
| non-standard `catalog_profile` (full/survey) | `<material>@<catalog_profile>-<digest>` |
| standard-profile `survey` | `<material>@survey-<digest>` |
| coherent / high-energy-floored / other override | `<material>@<fidelity>-<digest>` |

`<digest>` is the first 12 hex of the identity's `parameter_sha256`. `scan._resolved_run`
withholds `canonical_full` for any run that is not exactly the pristine
standard/full grid (a non-standard profile, `--quick`, a floored grid, coherent
emission, or a custom beam), so those all take an `@`-stem.

### Dual-read migration (non-destructive)

Checkpoints written under the older `<material>--<fidelity>-<digest>` scheme are
**not** renamed or migrated. `profiles._VARIANT_STEM_RE` and
`profiles.identity_from_stem` resolve both stem shapes, so existing on-disk
checkpoints keep loading, resuming, slimming, and pulling exactly as before;
only newly-written stems use `@`. No rename script, no disk-touching migration.

### Provenance sidecar (`meta.json`)

Every checkpoint (component dir or flat pickle) has a `meta.json` sidecar written
at run time by `run._manifest_save`, recording the resolved `dataset_identity`.
This is the **authoritative** identity source:
`profiles.identity_from_stem(stem, root)`, `archive._dataset_identity`, the
remote lifecycle, and `analyze.profile_menu` all read it directly rather than
recomputing a digest against the (possibly since-edited) live catalog. Recompute
survives only as a fallback for sidecar-less stems.

## Consumer awareness

- **`run.py` / `scan.py`** — write the `@`-stem and its `meta.json` sidecar; no
  pickle schema change.
- **`analyze.material_menu` / `analyze.profile_menu`** (`cxr app analysis`) — a
  material counts as browsable if it has a bare `<material>` checkpoint **or**
  any stem whose sidecar identifies it as that material. `profile_menu` lists
  the canonical run plus every named-profile / non-canonical variant, each
  labeled by its `catalog_profile` (or fidelity) and a short digest. This makes
  every named-profile checkpoint selectable in the analysis app — the primary
  fix this milestone delivers.
- **`slim`, `archive`, `restore`, `merge`, `prune`** — stem-based; `@`-stems
  round-trip. `slim._grid_from_stem` and `prune` resolve identity via the
  sidecar; `prune` regenerates current named-profile `@`-stems and leaves
  legacy `--` stems untouched (they are "unrecognized" under current profiles,
  as documented).
- **`cxr remote`** — `MATERIAL@PROFILE` remains a *pull selector* resolved via
  each candidate's `meta.json` (`resolve_profile_stem`). A full on-disk `@`-stem
  (with a `-<digest>` tail) is treated as a literal checkpoint, not a selector.
  `@` is permitted through the remote shell-token gate (safe as a bare word);
  stem prediction (`_remote/scripts._stems`), pull, and prune handle both
  legacy `--` and new `@` stems.

## `cxr checkpoints` listing — deferred (recommendation)

The 2026-07-26 draft proposed a `cxr checkpoints` command listing active stems
with material / set / record-count / grid summary. It is **not implemented in
this milestone**: browsability — the binding requirement — is delivered by
`cxr app analysis`, and `cxr checkpoint list` already means "archive shelf", so
an active-stem listing needs a fresh `cli-ui-ux` naming/output decision beyond
the two locked decisions. Recommendation for a follow-up task: add
`cxr checkpoint ls` (read-only), listing every active stem with its sidecar
material / profile / record count, flagging sidecar-unresolvable stems
`unknown` rather than hiding them.

## Retired design (superseded by `catalog_profile`)

The earlier draft's override layer is retired and intentionally **not** built:

- `cxr-overrides.toml` at repo root + `CXR_OVERRIDES` / `--overrides-file`
  resolution.
- `cxr config set/get/unset/clear/list/materials/diff` subcommand
  (`user_config.py`).
- `cxr scan --preset NAME` / `--set KEY=VALUE` / `--no-overrides` session flags
  and the `<material>@adhoc.pkl` stem rule.

These solved per-material reconfiguration a second way; `catalog_profile`
profiles in `data/materials.toml` already own that role, feed the identity
digest, and now name the `@`-stem. Adding a parallel override file would create
two sources of truth for the same grids. If pristine-catalog session overrides
are ever wanted again, reopen as a new task rather than reviving this file.

## Non-goals

- No nested-pickle schema change; legacy checkpoints load unchanged.
- No runtime editing of packaged `materials.toml`.
- No cross-variant merged analysis view (load stores manually or use `merge`).
- No rename of existing on-disk `--<fidelity>-<hash>` checkpoints (dual-read).
