# issue-161-xsgen-subsystem

`pyrite.xsgen`: generate, store, and resolve cross-section and stopping-power
tables produced by external Fortran codes (ELSEPA, SBETHE, BremsLib).

GitHub issue: #161. Branch: `issue-161-xsgen-subsystem`.
Worktree: `/home/alexa/dev/pyrite.worktrees/issue-161-xsgen-subsystem`.
Design: `agentdocs/specs/2026-09-21-external-fortran-code-integration.md`.

Prerequisite #162 (third-party notices, license metadata, import-linter
`forbidden_modules` entries, `extern_codes` marker) landed on `main` at
`5e12f8f2`; this branch starts from it.

## Session authority

Direct `/lead-task` invocation. Task-local checkpoint commits and
branch/worktree creation only. **No push, no GitHub issue edits, no
delegation.** Issue checkboxes stay unticked; this file is the local record.

## Milestones

#161 is not one slice. Decomposition, in the spec's sequencing order:

| M | Scope | State |
| --- | --- | --- |
| M1 | `sources.py`, `toolchain.py`, `_run.py`, `store.py` + fake-binary tests | this session |
| M2 | `pyrite tables` group + `cli-reference.md` regeneration | this session |
| M3 | manifest hash into run identity; stale checkpoints cannot be served | this session |
| M4 | vendor ELSEPA + `sbethe.f`, `.gitattributes` `-text`, `tables fetch` | not this session |
| M5 | `elsepa/`, `sbethe/`, `bremslib/` deck+parse + vendor-reference parser tests | not this session |
| M6 | shipped BremsLib-derived tables + maintainer refresh path | gated, see below |

M4-M6 are deliberately out of this session's scope. M6 is additionally gated on
a measurement the issue requires *before* committing tables: derived-table size
across the built-in catalogue.

## Local environment

All three upstream trees are present beside the checkout, and gfortran is
installed, so `extern_codes` tests are runnable here even though CI cannot run
them:

- `../elsepa-2020`, `../sbethe`, `../BremsLib_v2.0.8`
- gfortran 15.2.0 (`/usr/bin/gfortran`)

## Findings against the spec

### F1 — D2's material identity hash does not exist

D2 says the material identity hash "comes from the existing
`pyrite.materials._identity`". It does not. That module derives a *display*
identity -- formula, phase, full name, slab cut, bracket notation -- and
contains no hash of any kind. Nothing else in `materials/` hashes a material
record either (`atomic.py` blake2b digests are energy-grid cache keys).

Consequence for the store key: the `Material` arm of D2 needs a hash defined by
this issue, over the fields the Fortran decks actually consume -- composition,
density, mean excitation energy, and for muffin-tin ELSEPA the
nearest-neighbour distance and band gap -- not over the display label. Hashing
the catalogue key alone would be wrong: it would not change when the
catalogue's composition or density for that key changed.

### F2 — the identity marker must be conditional, not a constant

`campaign/profiles.py:_identity_v1` records model markers as *unconditional*
constants (`stopping_model`, `characteristic_model`, `bremsstrahlung_model`,
`line_kinematics`). Each such addition perturbs every digest exactly once and
orphans every existing checkpoint -- the file documents this as a deliberate
rev-and-re-run cost paid when the physics actually changed.

No consumer resolves an xsgen table until #89/#90. Adding an unconditional
xsgen marker now would orphan every checkpoint in exchange for zero change in
computed numbers. So M3 follows the *divergence-only key* pattern already in
the same function (`if emission != "incoherent"`, `transport_numerics`): the
marker appears only when a run actually resolves an xsgen table, and a run that
resolves none keeps its `parameter_sha256` bit-for-bit.

D5's `BREMSSTRAHLUNG_MODEL` prerequisite is untouched by this and stays with
#86/#87, per the spec: it must land in the same change that exposes the model
selector.

### F3 - ELSEPA compiles from `elscata.f` alone

`elscata.f` opens with `INCLUDE 'radial.f'` and `INCLUDE 'elsepa2020.f'`.
Naming all three on the compiler command line compiles the included bodies
twice and the link fails on dozens of duplicate symbols (`sfas0_`, `dbas_`,
`splset_`, ...). The build therefore passes `elscata.f` only, plus
`-I<tree root>` so the INCLUDE lines resolve from the cache directory the
build runs in.

All three files stay in `digest_sources`: they are part of the binary, so
patching `radial.f` is a different code even though the compiler is never
pointed at it directly.

Found by the `extern_codes` anchor, not by review. The spec's architecture
section does not mention it.

### F4 - the symlink copy fallback is implemented

The spec leaves "should the scratch directory fall back to copying when the
filesystem refuses symlinks" open. Implemented in the safe direction:
`_link_dir` tries `symlink_to` and falls back to `copytree`, recording which
was used in `RunResult.link_modes`. The codes only ever `OPEN` paths *beneath*
the directory, so a copy is behaviourally identical and differs only in cost.
Still untested against a real symlink-refusing filesystem.

### Vendor anchor result

`elscata.in` (Z=80, the shipped deck) reproduces
`test-run-output/dcs_1p000e03.dat`: 649/649 lines, 24 differing only in the
last of six significant digits, confined to the interference column
(~1e-9 against a 1e-12 to 1e-15 DCS scale, 1 part in 5.6e5). That is a libm
difference between our gfortran 15.2.0 and the vendor's compiler, so the
anchor compares numerically at `rtol=1e-5` -- one last-digit unit at six
significant figures -- rather than byte-for-byte. A byte comparison would be
an assertion about gfortran, not about ELSEPA.

### F5 - run identity has two surfaces, not one

D5 names `campaign/profiles.py` and `api.py`. The identity that gates
*checkpoint stems* is `_identity_v1` -> `parameter_sha256`, but there is a
second, independent cache: `case_content_key` keys the content-addressable
blob store (`checkpoints/_checkpoint_store.py`, `runs/run.py`,
`checkpoints/persistence.py`). Its own docstring already records why the
stopping/characteristic/bremsstrahlung constants are in it -- "without them a
blob from an earlier physics/data generation could be served silently".

A table marker in `dataset_identity` alone would move the stem while the blob
store still served arrays computed from the superseded table. Both take
`xsgen_tables`, and both take it conditionally.

`api.py`'s `bremsstrahlung_model` entries are provenance *record* fields
written into results, not identity inputs, so they need nothing here.

### F6 - the manifest had to cover the payload, not just the recipe

The first `identity_markers` test failed: regenerating a table with different
numbers produced the *same* manifest digest. Every "how it was made" field
agreed (same code, deck, shapes, dtypes) and `created_utc` has one-second
resolution, so two writes in the same second collided -- and that digest is
exactly what invalidates checkpoints.

The manifest now carries `arrays_sha256` over the stored arrays, normalized to
little-endian so a shipped manifest verifies on a big-endian host. Found by
the test, not by review.

## Checklist

- [x] M1 `sources.py` / `toolchain.py` / `_run.py` / `store.py`
- [x] M1 tests: scratch isolation, symlinks, cwd, fake binary; store round-trip
- [x] M1 `extern_codes` anchors against the real ELSEPA tree (opt in with
      `PYRITE_EXTERN_CODES_TESTS=1`; ~84 s)
- [x] M2 `pyrite tables` group (`path`, `list`, `show`, `sources list|set`),
      `cli-reference --check` clean. `generate`/`fetch` deferred to M5/M4.
- [x] M3 conditional identity marker on both surfaces + regression tests
- [x] import-linter contracts pass with `xsgen` populated (11 kept, 0 broken)
- [ ] `pyrite-dev verify`
