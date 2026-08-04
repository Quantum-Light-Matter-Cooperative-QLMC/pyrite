# fidelity namespace swap

## Scope

Resolve the CLI-redesign RFC §6 Q3 (`profile` vs `context`) by fixing **`profile`
as the campaign term** and moving the **fidelity** concept (`full`/`survey`) onto
its own `fidelity` namespace. Internal rename; no command surface, output, or
checkpoint-format change. One Python-API behavior change — see flagged item 1.

## The five "profile" senses

| Sense | What | Owns "profile"? | Action |
|---|---|---|---|
| A | **fidelity** — `full`/`survey` presets | no | rename → `fidelity` (this task) |
| B | **campaign / catalog_profile** — `standard`, `[profiles.*]`, `cxr profile`, `--profile` | **yes** | keep — canonical campaign term |
| C | **performance profile** — compute-perf logs (`performance_profile.py`, `cxr remote profile`) | no | keep own `performance_profile` name; unrelated domain |
| D | **named_profile / dataset run-identity** — `named_profile_stem/identity`, `high_energy_floor_*` | no | out of scope; suggested future term `identity`/`dataset` |
| E | **line-shape profile** — `line_grid/golden.py spec.profile` | no | keep; physics, unrelated |

## Checklist

- [x] `profiles.py`: `SweepProfile`→`FidelityPreset`, `get_profile`→`get_fidelity_preset`, `_PROFILES`→`_FIDELITY_PRESETS`.
- [x] `recompute_defaults.py`: drop local `PROFILE_NAMES` (import `FIDELITY_NAMES`), `_validate_profile`→`_validate_fidelity`, `settings`/`sweep` `profile`→`fidelity`.
- [x] `run.py`: `repair_brem_wide`/`repair_line_spec` `profile=`→`fidelity=` params + internal uses. KEEP persisted `c["brem_profile"]`/`c["line_profile"]` field literals.
- [x] `rebrem.py`/`reline.py`: option-dict key `"profile"`→`"fidelity"`.
- [x] `scan.py`: local `profile = identity["fidelity"]`→`fidelity` (output text `[profile=…]` unchanged).
- [x] `results/selection.py`: `_grid_names(profile=)`→`fidelity=`.
- [x] `config.py`: drop the fidelity branch of `material_sweep(profile=)` (now a `catalog_profile` alias only).
- [x] docs: `repo_map.md` public-API entries; `physics-validation-ledger.md` `coherent-emission` anchor.
- [x] tests: `test_profiles.py` imports/usages; `test_run.py` `profile=`→`fidelity=` (params + captured-kwargs asserts).
- [x] Green: targeted tests + lint + typecheck.

## Independent verification (2026-08-04)

Clean env (`UV_PROJECT_ENVIRONMENT=/tmp/cxr-mc-venv-fid`, `CXR_MC_BACKEND=cpu`):

- Branch as-is: lint ✅, typecheck ✅, suites packaging **183**, cli **890**,
  core **967 passed / 40 skipped**, apps **287**.
- Merge preview against `main` @ `4b777fe` (slices 1 + 2 landed): merges clean,
  no conflicts, no semantic drift — zero residual `SweepProfile` /
  `get_profile` / `PROFILE_NAMES` / `_validate_profile` in `src`, `tests`,
  `notebooks`. Re-ran on the merged tree: lint ✅, typecheck ✅, and the same
  183 / 890 / 967+40 / 287. Preview aborted; branch left unmerged.
- `scan.py`'s `[profile=…]` progress line and the persisted
  `brem_profile`/`line_profile` case fields are unchanged, as intended.

### Flagged, not blocking

1. **`config.material_sweep(profile=…)` is a semantic change, not a pure
   rename.** It used to mean fidelity for `"full"`/`"survey"` and catalog
   profile otherwise; it is now a catalog-profile alias only, so
   `material_sweep(profile="survey")` goes from selecting the survey fidelity
   to raising on an unknown catalog profile. No in-repo caller passes
   `profile=` (all use explicit `fidelity=`/`catalog_profile=`) and
   `cxr_mc/__init__.py` re-exports neither, so nothing breaks here — but it is
   a Python-API behavior change, outside the README's "pure internal rename".
2. `docs/cli-energy-grid-sweep-rework-plan.md` lines 15 and 134 still name
   `SweepProfile` and `profiles.py PROFILE_NAMES`. Historical plan doc, left as
   a point-in-time record (same treatment slice 2 gave `package-structure-rfc.md`).
3. `repo_map.md` had listed `PROFILE_NAMES` as public on `profiles.py`; it
   actually lived in `recompute_defaults.py`. Pre-existing staleness, corrected
   by this branch.

## Non-goals / deferred

- **Sense D rename** (`named_profile_*` → `identity`/`dataset`) — separate future slice.
- **Persisted `brem_profile`/`line_profile` case-field rename** — checkpoint-format change; needs dual-read migration. Deferred.
- No command/help/output/exit changes; `docs/cli-reference.md` unaffected.
