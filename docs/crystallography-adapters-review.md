# External crystallography library adapters — review

Read-only review of two unmerged branches against `main @ 5f94424` (current `main` tip at
review time: `1416ef1`). Reviewed with `git log`/`git show`/`git diff`/`git merge-tree` only —
neither branch was checked out, merged, or executed. Scope is TODO P2 #1.

- `origin/feature/diffpy` @ `67f27e0` — `diffpy.structure` CIF/structure importer
- `origin/feature/dans-diffraction` @ `d4dbeff` — optional `Dans_Diffraction` validation-oracle
  checks (built on top of `feature/diffpy`; `d4dbeff`'s parent is `0fc98a5` "Research
  Dans_Diffraction validation oracle", whose parent is in turn `67f27e0`)

`TODO.md` P2 #1 names these `codex/diffpy-structure-importer` and
`codex/dans-diffraction-research`; those branches do not exist. The correct refs are the two
above. This is fixed on this branch's `TODO.md`.

Both branches were cut from `511a067`, which is an ancestor of `5f94424` — i.e. cut from a
point on `main`'s history before the `feature/grazing-grating` and `worktree-marimo` merges.
`main` has since moved `docs/physics-validation-ledger.md`, `docs/repo_map.md`, and
`src/cxr_mc/crystallography.py` (the `optical_constants` addition). `pyproject.toml` was
*not* touched on `main` in that range (`git diff 511a067 5f94424 -- pyproject.toml` is
empty), so it needs no reconciliation.
Drift/conflict detail is in each branch's section below.

---

## `feature/diffpy` — `diffpy.structure` CIF/structure importer

**What it adds.** Two pure-addition functions in `src/cxr_mc/crystallography.py`:

- `diffpy_structure_to_crystal_info(structure, mosaic_fwhm_deg=None)` — converts a
  `diffpy.structure.Structure`-like object (matched via a local `Protocol`, not an isinstance
  check) into the existing `CRYSTALS`-entry dict shape: `lattice` (`system="general"` + the six
  cell parameters), `basis` (list of `(element, xyz)`), `V_cell` (via the module's existing
  `_direct_lattice_vectors`), and `mosaic_fwhm_deg`.
- `load_crystal_from_cif(path, mosaic_fwhm_deg=None)` — thin wrapper: `diffpy.structure.Structure(filename=..., format="cif")` then the function above.

Plus three new `Protocol` classes (`_DiffpyLatticeLike`, `_DiffpyAtomLike`,
`_DiffpyStructureLike`) that structurally type the diffpy object without importing `diffpy` at
module scope — the `from diffpy.structure import Structure` import is local to
`load_crystal_from_cif` only.

**What it changes in existing code paths.** Nothing. `git diff main...origin/feature/diffpy --
src/cxr_mc/crystallography.py` is a pure insertion after `_reciprocal_basis`; no existing
function body, signature, or behavior changes. `reciprocal_g_vector` and everything after it in
the file is untouched (the earlier `diff main diffpy` I ran without `...` initially showed
`optical_constants` as "removed" — that is an artifact of diffing against current `main`
directly instead of the merge-base; `optical_constants` did not exist yet at the branch's cut
point (`511a067`) and was added to `main` later by `feature/grazing-grating`. The triple-dot
diff against the correct merge-base confirms pure addition.)

**Hard or optional dependency?** **Hard.** `diffpy-structure>=3.4.0` is added to
`[project].dependencies` in `pyproject.toml` (not an `optional-dependencies` extra), and
`uv.lock` resolves it + its `diffpy-utils` transitive dependency. The *import* is lazy
(function-local, so `import cxr_mc.crystallography` never touches `diffpy` unless
`load_crystal_from_cif` is called), but the *package* is not optional at the packaging level —
`uv sync` always installs it once this merges. That's a legitimate design choice (a CIF importer
arguably deserves to be a real dependency, not gated), but it means merging this branch requires
an actual `uv sync` of the shared venv to take effect — worth calling out given this task's own
environment-trap warning about uncoordinated `uv sync`/`uv run --active` in a shared venv.
I confirmed the current shared venv (`C:/dev/cxr-mc/.venv`) does **not** have `diffpy.structure`
installed today, so as of right now two of the branch's three new tests would fail to collect
(`ModuleNotFoundError`) if run as-is against gate 1 (`pytest`) — not a defect in the branch, but
a real pre-merge step that has to happen deliberately, not silently.

**Duplicate physics?** No. `structure_factor`, `chi_g`, `U_g`, `debye_waller`, and the atomic
form factors are untouched; the adapter only reshapes lattice/basis data that then flows through
the *existing* physics functions. Single source of truth for X-ray scattering physics is
preserved.

**Test coverage.** Three new tests in `tests/test_crystallography.py`:
`test_diffpy_structure_adapter_preserves_lattice_basis_and_volume` (uses a hand-written fake
`Structure`/`Lattice`/`Atom` — no `diffpy` import, always runs),
`test_diffpy_structure_adapter_accepts_installed_diffpy_structure` and
`test_load_crystal_from_cif_uses_diffpy_structure` (both `from diffpy.structure import ...`
directly, unconditionally — no `importorskip`/try-except guard). Given the hard-dependency
choice above, this is internally consistent (the package is *supposed* to always be present
post-merge), but it does mean these two tests have no fallback if the dependency is ever
missing from an environment — unlike the `dans-diffraction` branch's tests (see below), there is
no skip path. Given `pyproject.toml` correctly declares it as a hard dependency, I'd call this
acceptable rather than a defect, but it's the direct consequence of the hard-vs-optional
decision and worth the user confirming is what they intended (see the open question at the end).

**Physics-obligation compliance.** Both new functions carry a `Validation: diffpy-structure-adapter`
marker and a ledger row (status `anchored`, pointing at the two installed-package tests above).
Strictly, this isn't "new physics" in the equation-derivation sense the CLAUDE.md rule targets —
it's a data-shape conversion, no formula is introduced — but the branch treats it as if it were,
which is more conservative than required and exceeds the repo's own baseline (17/22 existing
ledger rows lack an in-code marker at all). One accuracy note on the ledger row itself: it
claims status `anchored` ("regression test green"), but that claim rests on the two
diffpy-requiring tests, which — per the point above — cannot be green in an environment that
hasn't yet run `uv sync` for this dependency. The status is not wrong once the dependency lands,
but it was written as if that step had already happened; it hasn't, in the shared venv.

**Drift/conflicts against `main @ 5f94424`.** `git merge-tree` against the actual merge-base
(`511a067`) shows exactly 3 conflict hunks, all trivial "both sides appended near the same
anchor" conflicts, not logical conflicts:
- `TODO.md` — expected; this branch's `TODO.md` follows the per-branch-summary convention and
  gets replaced/re-triaged into `main`'s `TODO.md` at merge time per the repo's own workflow.
- `docs/physics-validation-ledger.md` — conflict is only on the `Progress: **0 / N signed-off**
  · ...` summary line (both branches and `main` independently advanced the counts) and the
  insertion point for the new row; trivially resolved by recomputing the summary line and
  keeping both new rows.
- `tests/test_crystallography.py` — both sides append new test functions at end of file;
  trivially resolved by keeping both blocks.
`src/cxr_mc/crystallography.py`, `docs/repo_map.md`, and `pyproject.toml` all merge cleanly
(`git merge-tree` reports them as auto-mergeable, not conflicted). `main` moved in the first
two since the branch point; `pyproject.toml` it did not touch at all.

**Recommendation: MERGE-AFTER-FIXES.**
1. Rebase (or let a human resolve) the three trivial conflicts above onto current `main`.
2. Run `uv sync` on the shared venv as a deliberate, coordinated step (not inside a worktree,
   not concurrent with other agents) so `diffpy-structure` actually installs, then re-run gate 1
   and confirm the two diffpy-requiring tests pass for real before calling the ledger row
   `anchored`.
3. Confirm the hard-dependency choice is intentional (see the closing question) — if the intent
   was "optional CIF import," this needs to move to an `optional-dependencies` extra with a
   guarded import and `importorskip`-style tests instead.

No code-quality objection: I extracted `crystallography.py` and `test_crystallography.py` from
this branch and ran the repo's own `ruff check` config against them standalone — clean, no
findings. Style, docstring density, and typing idiom (Protocol-based structural typing, not
`isinstance`) match the surrounding module closely.

---

## `feature/dans-diffraction` — optional `Dans_Diffraction` validation oracle

Built on top of `feature/diffpy` (`d4dbeff` → `0fc98a5` → `67f27e0`), so this section only
describes what it adds *beyond* that branch; `crystallography.py` and
`tests/test_crystallography.py` have zero further changes on this branch
(`git diff origin/feature/diffpy...origin/feature/dans-diffraction -- src/cxr_mc/crystallography.py tests/test_crystallography.py`
is empty).

**What it adds.**
- `src/cxr_mc/validation_oracles.py` (new module, ~250 lines): `load_dans_crystal_from_cif`,
  `build_dans_crystal_from_cxr` (P1 crystal built from an internal `CRYSTALS` entry, full
  occupancy, `uiso=0` by default to neutralize the oracle's Debye-Waller term),
  `compare_lattice`, `compare_reflection_geometry` (`|g|` vs. `Dans_Diffraction`'s
  `Cell.Qmag`), `compare_structure_factor_magnitudes` (`|F_hkl|²` vs. the oracle's
  `Scatter.structure_factor`), three frozen result dataclasses, and
  `DansDiffractionUnavailableError`.
- `checks/dans_diffraction_oracle.py` — a runnable example/smoke script over silicon, sapphire,
  and `mote2_product`, following the exact `sys.path.insert(...  "src")` + `# noqa: E402`
  pattern every other `checks/*.py` script in this repo already uses.
- `tests/test_validation_oracles.py` — unit tests against hand-rolled `FakeCell`/`FakeScatter`/
  `FakeCrystal` doubles.
- `claudedocs/research_dans_diffraction_20260702_0956.md` — the research write-up behind the
  branch; recommends `Dans_Diffraction` as validation-only, explicitly *against* making it a
  production dependency or replacing `diffpy.structure` as the importer. This document
  references `claudedocs/research_crystals_alternatives_20260702.md` as "local prior context"
  for the original `crystals`-package evaluation — **that file does not exist anywhere in this
  repository's history** (checked `git log --all -- claudedocs/*`). Whatever prior research
  informed the "crystals" comparison in `TODO.md` P2 #1 was not committed, so my answer to the
  open question at the end relies on general knowledge of the `crystals` PyPI package, not an
  in-repo source — flagging that explicitly rather than presenting it as repo-verified.

**What it changes in existing code paths.** Nothing beyond the `diffpy` branch's own additions
(which this branch inherits, unmodified). `validation_oracles.py` only *reads* from
`cxr_mc.crystallography` (`CRYSTALS`, `reciprocal_g_vector`, `structure_factor`) — it imports
no `montecarlo`/`sweep`/`results` modules and nothing imports it back, so it's a pure leaf
hanging off `crystallography`, matching how the branch's own `docs/repo_map.md` edit describes
it (`crystallography ├── validation_oracles`).

**Hard or optional dependency? Is the optional import guarded — correctly.** `Dans_Diffraction`
is **not added to `pyproject.toml` or `uv.lock` at all** (confirmed:
`git diff origin/feature/diffpy...origin/feature/dans-diffraction -- pyproject.toml uv.lock` is
empty). The only import is inside `_dans_crystal_class()`, wrapped in
`try: importlib.import_module("Dans_Diffraction") except ModuleNotFoundError: raise
DansDiffractionUnavailableError(...)`. This is the correct pattern for a genuinely optional
dependency, and it's the cleanest of the two branches on this axis.

**Duplicate physics?** This is the one place a second source of truth legitimately appears —
and it is *deliberate and correctly scoped*, not a mistake. The module docstring is explicit
that `Dans_Diffraction` uses the opposite structure-factor phase-sign convention from cxr_mc, so
the comparisons are restricted to phase-insensitive quantities (`|g|`, `|F_hkl|²`, lattice/cell
parameters) and complex-phase comparison is explicitly called out as a separate, unimplemented
concern. `structure_factor`/`chi_g`/`U_g`/`debye_waller` in `crystallography.py` remain the only
code path anything in `montecarlo`/`results`/`plots` ever calls; `Dans_Diffraction`'s own
scattering computation only feeds the *comparison* dataclasses in `validation_oracles.py`, which
nothing in production imports. If this had instead been wired so that production code could
optionally route through `Dans_Diffraction`, that would be the two-sources-of-truth hazard the
task asked me to watch for — it isn't wired that way.

**Test coverage, and would it run in CI without the optional dep?** Five tests in
`tests/test_validation_oracles.py`, all against local `Fake*` doubles — none import
`Dans_Diffraction`, so all five run and pass in CI with zero optional dependencies installed
(verified: `Dans_Diffraction` is not in the shared venv either). The one test that exercises the
missing-package path (`test_missing_dans_diffraction_raises_clear_optional_dependency_error`)
monkeypatches `importlib.import_module` to raise `ModuleNotFoundError`, so it tests the guard
without needing the real package absent/present either way — a good pattern. `checks/dans_diffraction_oracle.py`
is not pytest-collected (`pyproject.toml`'s `testpaths = ["tests"]` only) and itself
catches `DansDiffractionUnavailableError` and exits 0 with a "skipping" message when the
optional package isn't installed, matching the module docstring's stated contract ("a missing or
optional oracle must not affect runtime imports or production calculations").

**Physics-obligation compliance.** The module docstring gives the cxr_mc-side structure-factor
equation, states the phase-sign-convention caveat, lists assumptions (full-occupancy,
isotropic-only displacement, non-magnetic X-ray scattering) and a limiting case (missing oracle
must not affect runtime), and carries `Validation: dans-diffraction-oracle`. The ledger row is
added at status `unverified` (not `anchored` — appropriately conservative, since the "fake
oracle" unit tests confirm the harness's own plumbing but nothing here has actually been checked
against an installed `Dans_Diffraction` build yet). This meets the letter of the CLAUDE.md rule
and is honest about what is and isn't verified — better than the 17/22-unmarked baseline, and
more conservative in its status claim than the `diffpy` branch's `anchored` row.

**Drift/conflicts against `main @ 5f94424`.** Same three trivial conflicts inherited from
`feature/diffpy` (`TODO.md`, the ledger's progress-summary line, `test_crystallography.py`
append point), plus the same additive, non-conflicting `docs/repo_map.md` insertion. No new
conflict surface beyond what `feature/diffpy` already has — `validation_oracles.py`,
`checks/dans_diffraction_oracle.py`, `tests/test_validation_oracles.py`, and the research doc
are all new files, so `git merge-tree` reports them as clean additions.

**Recommendation: MERGE-AFTER-FIXES**, but the fixes needed are lighter than `feature/diffpy`'s:
1. This branch is currently based on `feature/diffpy`, so it can only land after (or together
   with) that branch's conflicts are resolved — it doesn't introduce independent conflicts.
2. No dependency/environment step is required (that's the point of it being optional) — this one
   is mergeable as-is once its base is current.
3. Minor: the dangling reference to `claudedocs/research_crystals_alternatives_20260702.md`
   should either be dropped from the research doc or that file should be located/recovered and
   committed alongside — right now it cites a source that isn't in the repository.

Same standalone-lint check as above: I extracted `validation_oracles.py`,
`tests/test_validation_oracles.py`, and `checks/dans_diffraction_oracle.py` and ran the repo's
`ruff check` config against them directly — clean, no findings. The `checks/` script matches the
existing scripts' `sys.path.insert` + `# noqa: E402` convention exactly.

---

## Does the original `crystals` package still offer unique value?

Caveat up front: unlike the two branches above, there is no in-repo research document for the
`crystals` package (PyPI: `crystals`, by L. René de Cotret; part of the `scikit-ued` ecosystem)
— `TODO.md` P2 #1's reference to prior evaluation and the dans-diffraction research doc's own
citation of `claudedocs/research_crystals_alternatives_20260702.md` both point at something that
was never committed. What follows is drawn from general knowledge of that library, not a
repo-verified source, and should be weighted accordingly — recommend an actual install-and-probe
spike (mirroring the dans-diffraction branch's own methodology) before treating this as settled.

What each of the three libraries uniquely provides for cxr-mc's purposes:

- **`diffpy.structure`** (now the production importer, per `feature/diffpy`): a minimal
  structure *container* — lattice + fractional-coordinate basis parsing from CIF, with the
  Structure/Lattice/Atom object model this branch's adapter targets. Its unique value to
  cxr-mc is exactly this: the lightest-weight, already-integrated path from "a CIF file" to
  the `CRYSTALS` dict shape, feeding cxr_mc's own X-ray physics. It's `diffpy`'s own CIF parser
  (`diffpy.structure.parsers.p_cif`) that would need to be relied on for any symmetry expansion
  (turning a space-group + asymmetric-unit CIF into the full P1 atom list) — the branch's own
  tests only exercise an already-P1 CIF, so whether `load_crystal_from_cif` correctly expands a
  real space-group CIF (e.g. a corundum/sapphire-style structure, which the repo's own ledger
  notes had to be *manually* expanded from Wyckoff sites — see the `sapphire-corundum-structure`
  ledger row) is untested and, as far as I can tell from this review, unverified either way.
- **`Dans_Diffraction`** (validation-only, per `feature/dans-diffraction`): CIF read +
  symmetry expansion + its own independent X-ray structure-factor/reflection-intensity engine
  with its own scattering-factor tables. Its unique value is specifically as a *second
  implementation* to check `structure_factor`/`chi_g` against — not as an importer (the branch's
  own research explicitly rejects that role for it) and not for production use.
  Dans_Diffraction generally does perform space-group symmetry expansion internally, but nothing
  in cxr-mc currently exploits that (`build_dans_crystal_from_cxr` builds P1 crystals *from*
  cxr_mc's already-expanded basis, going the opposite direction).
- **`crystals`**: two capabilities distinct from both branches above. (1) `spglib`-backed
  space-group symmetry determination and full asymmetric-unit-to-P1 expansion, plus
  primitive/conventional cell reduction — a capability gap the repo's own ledger evidences is
  real today (the sapphire entry: "expanded from R-3c Wyckoff sites because the loader does not
  apply symmetry"). If new crystal structures keep arriving as space-group + Wyckoff-position
  CIFs rather than pre-expanded P1 CIFs, this is the one concrete, recurring pain point neither
  landed branch closes for certain — `diffpy.structure` *may* close it (its CIF parser is
  symmetry-aware), but that's unverified per the point above, and if it does turn out to close
  it, `crystals`' symmetry engine becomes redundant for cxr-mc's purposes. (2) `crystals` ships
  its own *electron* scattering form-factor tables (it's built for ultrafast electron
  diffraction / `scikit-ued`, not X-ray work) — that capability is irrelevant to cxr-mc, which
  is exclusively X-ray (PXR + coherent bremsstrahlung) and already sources X-ray form factors
  from `xraydb` (see `docs/atomic-data-sources.md`).

**Recommendation (not a merge decision — the user's call):** the strongest remaining argument
for `crystals` is narrowly the symmetry-expansion gap above, and even that is contingent on
`diffpy.structure`'s CIF parser *not* already handling it — a one-hour empirical check (feed a
real space-group CIF, e.g. re-derive `sapphire` from its literature space group, through
`load_crystal_from_cif`, and see whether the returned `basis` is already the full 30-atom
conventional-cell set) would settle this directly and should happen before any `crystals`
integration work is scoped. If `diffpy.structure` does expand symmetry correctly, `crystals`
adds nothing cxr-mc doesn't already get more cheaply from the two landed branches, and I'd call
it redundant. Its electron-diffraction form-factor tables are not relevant to this project
either way.

---

## Summary table

| branch | dependency | adds physics? | duplicate source of truth? | tests run w/o optional dep? | ledger row | recommendation |
|---|---|---|---|---|---|---|
| `feature/diffpy` | hard (`diffpy-structure`, in `dependencies`) | no — structural conversion only | no | no (2/3 new tests need the package; not yet installed in shared venv) | `anchored` (premature until `uv sync` + real test run) | MERGE-AFTER-FIXES |
| `feature/dans-diffraction` | optional, correctly guarded (not in `pyproject.toml`/`uv.lock`) | no in production; yes as an independent oracle, by design | intentionally, validation-only — not a hazard | yes (all 5 tests use fakes) | `unverified` (appropriately conservative) | MERGE-AFTER-FIXES (lighter: rebase + drop dangling doc reference only) |
