# fidelity namespace swap

## Scope

Resolve the CLI-redesign RFC §6 Q3 (`profile` vs `context`) by fixing **`profile`
as the campaign term** and moving the **fidelity** concept (`full`/`survey`) onto
its own `fidelity` namespace. Pure internal rename; no command surface, output,
or checkpoint-format change.

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

## Non-goals / deferred

- **Sense D rename** (`named_profile_*` → `identity`/`dataset`) — separate future slice.
- **Persisted `brem_profile`/`line_profile` case-field rename** — checkpoint-format change; needs dual-read migration. Deferred.
- No command/help/output/exit changes; `docs/cli-reference.md` unaffected.
