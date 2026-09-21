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

## Checklist

- [ ] M1 `sources.py` / `toolchain.py` / `_run.py` / `store.py`
- [ ] M1 tests: scratch isolation, symlinks, cwd, fake binary; store round-trip
- [ ] M2 `pyrite tables` group, `cli-reference --check` clean
- [ ] M3 conditional identity marker + stale-checkpoint regression test
- [ ] import-linter contracts pass with `xsgen` populated
- [ ] `pyrite-dev verify`
