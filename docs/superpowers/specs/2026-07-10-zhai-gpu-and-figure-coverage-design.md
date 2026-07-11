# GPU lab-box Zhai reproduction + supplementary figure coverage

**Date:** 2026-07-10
**Status:** Proposed
**Scope:** new `checks/anchor_figures.py::reproduce_all`, new root shim
`reproduce_zhai.py`, `src/cxr_mc/remote.py` (new `check` subcommand),
`src/cxr_mc/check.py` (new `--export` path), new `docs/validation/zhai-supplementary.md`,
tests under `tests/`.

## Problem

The Zhai validation figures (`checks/anchor_figures.py` + the "Zhai reproduction"
and "Zhai supplementary" sections of `notebooks/validation_app.py`, launched via
`cxr check`) run at publication sample counts — `ne=20_000`/`ne_brem=200` for the
Fig. 1c HOPG anchor, `ne=200` for each of the WSe₂/MoSe₂/h-BN supplementary
studies × thickness. This is CPU-minutes-to-tens-of-minutes per study on a
laptop; the user has a GPU lab box (`qlmc`, RTX 5080) already wired up for
material sweeps via `cxr remote scan/start/pull` but nothing plays that role for
the Zhai reproduction.

Separately, the supplementary figure set (`figure_supplementary_tmd`,
`figure_supplementary_hbn`) has thinner test coverage than the main Fig. 1c
anchor path, has no single command that renders the whole publication figure
set in one shot, has no combined/overview figure for the validation appendix,
and its literature inputs (thicknesses, tilts, energy windows, the pending
azimuth = 0 assumption) have no provenance write-up — `TODO.md` currently
carries this as a loose note.

## Why the cache is portable box → laptop

`cached_model_spectra` / `cached_coherent_spectra` (`checks/anchor_figures.py`)
key their output filename on a SHA-256 over `{schema, inputs, ne, ...}` plus the
**bytes** of every `src/cxr_mc/**/*.py` file (`_zhai_cache_key` /
`_supplementary_cache_key`). The repo's `.gitattributes` pins `* text=auto
eol=lf` and the on-disk source is confirmed LF-only, so the same source tree
hashes identically on the box and the laptop. A cache file computed on the box
at the same `(anchor/study, ne, ne_brem)` inputs therefore lands at the exact
filename the local app's `cached_*` loader already checks for — no format
translation, no manual matching, just a file copy. This is the mechanism the
whole design rides on, so it gets its own regression test (component 7).

## Design

### 1. `reproduce_all()` — `checks/anchor_figures.py`

A new function that force-populates `checkpoints/zhai_reproduction/` with every
cache the validation app can hit, at the app's own defaults so a pulled cache is
a guaranteed hit locally:

- `cached_model_spectra(ZhaiAnchor(), ne=20_000, ne_brem=200)` (Fig. 1c anchor)
- `cached_coherent_spectra(study, thickness_nm, ne=200)` for every
  `(study, thickness_nm)` pair across `ZHAI_SUPPLEMENTARY_STUDIES` (wse2 × 3,
  mose2 × 3, hbn × 1)

`ne`/`ne_brem`/`ne_supp` are parameters (defaulting to the app's values) so a
caller can override them — e.g. a smoke-test run overrides to a tiny `ne`. No
matplotlib import, no figure rendering: this function's only job is to leave
correct, hash-addressed `.pkl`s on disk. Returns a list of
`(label, path, cache_hit)` for each of the 8 computations so the caller can
print a summary table.

### 2. Root shim `reproduce_zhai.py`

Mirrors the existing `scan.py` shim exactly (sys.path insert of `src/`, thin
`__main__` guard, docstring pointing at the real logic in `checks/`): the box
invokes `python reproduce_zhai.py` with no project install. Flags: `--ne`,
`--ne-brem`, `--ne-supp`, `--refresh` (forces recompute even if a matching cache
already exists), `--cache-dir` (defaults to `checkpoints/zhai_reproduction`,
override for smoke tests). Calls `reproduce_all` and prints the returned table.

### 3. `cxr remote check` — `src/cxr_mc/remote.py`

New subcommand alongside `scan`/`start`/`pull`, reusing existing machinery
rather than inventing a parallel path:

- **Foreground (default):** `cxr remote check [--refresh] [--ne N] [--ne-brem N]
  [--ne-supp N] [--no-sync]` → `sync_code()` (unless `--no-sync`) → ssh run
  `reproduce_zhai.py` on the box with the given flags → pull the whole
  `checkpoints/zhai_reproduction/` directory back over the existing tar-over-ssh
  transport used by `sync_code`/`SYNC_PATHS` (new helper, since this pulls a
  *directory of generated artifacts* rather than one named file the way
  `pull()` does for material checkpoints).
- **Detached:** `cxr remote check --detached [--follow]` — reuses the existing
  job-dir queue (`_queue_script`/`_launch_queue_command`/`_live_jobs`/`jobs`/
  `status`/`logs`/`attach`/`stop`), which is already material-name-driven; the
  Zhai job is keyed by a synthetic pseudo-material token (`"zhai"`) purely for
  the busy-check/stop UX, since it produces a fixed directory rather than a
  `<material>.pkl` stem. `attach`/`status`/`logs`/`stop <jobid-or-"zhai">` work
  unmodified.
- **`cxr remote check --pull`:** fetch `checkpoints/zhai_reproduction/` only,
  for after a detached run finishes — same directory-pull helper as the
  foreground path's trailing pull.

This keeps `remote.py`'s one real behavioral rule intact: two runs must never
race the same output. `_refuse_if_busy` extends to recognize the `"zhai"`
pseudo-stem the same way it does material stems.

### 4. Batch figure export — `cxr check --export`

`src/cxr_mc/check.py` gains an `--export` flag: instead of launching marimo, it
loads every cache already on disk under `checkpoints/zhai_reproduction/` (via
the same `cached_*` functions, so a miss recomputes locally rather than
failing) and renders the full figure set — the existing Fig. 1c trio
(`figure_spectra`, `figure_flux_anchor`, `figure_enhancement`) plus every
supplementary panel (`figure_supplementary_tmd` × 2 materials × 3 thicknesses,
`figure_supplementary_hbn` × 1) plus the new overview figure (component 5) — to
`figures/`, mirroring `anchor_figures.main()`'s existing PNG+PDF write pattern.
One command turns a GPU pull into the complete publication figure set with no
per-study clicking in the app.

### 5. Combined overview figure

`figure_supplementary_overview(spectra_by_material: dict[str, dict[float,
np.ndarray]])` in `checks/anchor_figures.py`: one row of three panels
(WSe₂/MoSe₂/h-BN side by side), each showing that material's steepest
requested polar tilt (`-20°`, the extremum already in every study's
`polar_tilts_deg` and the tilt Zhai's SI associates with the largest coherent
yield) at that material's thinnest listed thickness — a single representative
slice per material rather than the full tilt × thickness grid, for the paper's
validation appendix. Follows the existing `figure_supplementary_*` signature
style (explicit crystal-membership guard, `fig.tight_layout()`, returns the
`Figure`).

### 6. Supplementary provenance write-up

`docs/validation/zhai-supplementary.md`: cross-checks
`ZHAI_SUPPLEMENTARY_STUDIES`'s thicknesses/tilts/energy windows/materials
against the Zhai et al. SI, states what's confirmed vs assumed, and gives the
open azimuth = 0 question (currently a loose `TODO.md` note) a proper home with
the specific SI figure/table it needs checking against. This is a
**documentation/provenance pass, not a ledger-status change** — per
`docs/validation/README.md`'s lifecycle rules, moving any of this to
`rederived`/`signed-off` requires the actual fresh-context re-derivation and
(for sign-off) the user's certification; this write-up records status and the
open question so that pass has a clear starting point. `TODO.md`'s loose
azimuth note gets superseded by a pointer to this doc.

### 7. Tests

- `figure_supplementary_tmd` / `figure_supplementary_hbn`: unit tests (currently
  thin) covering the guard clauses (wrong crystal, incomplete tilt set) and a
  smoke-render at trivial `ne`.
- `figure_supplementary_overview`: new tests for the same guard-clause /
  smoke-render pattern.
- `reproduce_all`: a test at tiny `ne`/`ne_supp` against a `tmp_path` cache dir,
  asserting all 8 expected cache files are written and `reproduce_zhai.py`'s
  CLI wiring calls through correctly.
- **Cache-key portability test**: pins the LF-only invariant this whole design
  depends on — asserts `_zhai_cache_key`/`_supplementary_cache_key` are stable
  across a simulated CRLF round-trip of the hashed source tree (or, more
  directly, asserts the repo's tracked physics-module sources contain no
  `\r\n`), so a future line-ending regression fails loudly in CI instead of
  silently breaking box↔laptop cache sharing.

## Out of scope

- The azimuth = 0 re-derivation itself (component 6 documents the question;
  answering it is a separate fresh-context physics-validator pass per
  `docs/validation/README.md`).
- Moving any ledger row's status (`docs/physics-validation-ledger.md` is
  untouched by this work).
- GPU/CUDA acceleration of the Monte Carlo kernels themselves — this design
  only relocates *where* the existing CPU-bound `simulate_trajectories`/
  `mc_spectrum` calls run (the lab box's CPU, not its GPU); true GPU
  utilization is a separate, larger effort already noted elsewhere in
  `TODO.md`/`CUDA backend` context and not part of this spec.
- Grid/config caching parity with `cxr remote scan --grid` filtering — the Zhai
  cache directory has no analogous "current grid" concept to filter against;
  the whole directory is always the desired set.

## Sequencing

1. `reproduce_all` + `reproduce_zhai.py` (components 1–2) — the GPU-runnable
   unit, testable locally with tiny `ne` before ever touching ssh.
2. `cxr remote check` (component 3) — wires 1–2 onto the existing remote
   transport/queue.
3. `cxr check --export` (component 4) — local batch rendering once caches
   exist (from step 2's pull, or purely local runs).
4. Overview figure (component 5) — needed by component 4's full export.
5. Provenance write-up (component 6) — independent of 1–5, can land anytime.
6. Tests (component 7) — threaded alongside each component as it lands, plus
   the standalone cache-portability test.
