# Consolidate material command tree

Branch: `feature/material-command-tree`

TODO scope: P1 CLI Work sub-items 8–9.

## Goal

Collapse the thin catalog group into the material domain and place blazed
material sweeps there:

```text
cxr material validate [CATALOG]
cxr material show MATERIAL [--profile PROFILE]
cxr material set MATERIAL [--profile PROFILE] ...
cxr material blaze MATERIAL ...
```

## Evaluation result

Combine `catalog` and `material`: yes. Current `catalog` owns only validation;
separate top-level nouns force users to distinguish storage from the domain
object without adding useful structure.

Move `blaze` under the combined `material` group: yes. Blaze is a specialized
single-material operation with geometry-specific required options, not the
profile-oriented flat run surface. Nesting avoids overloading `cxr run` while
removing another top-level verb.

## Decisions

- `material` is the retained top-level noun.
- Canonical validation path: `cxr material validate`.
- Canonical blaze path: `cxr material blaze MATERIAL`.
- Preserve profile membership ownership under `cxr profile members`; do not
  move it into this group.
- Root/group help remains lazy and must not load catalog data, NumPy, CuPy, or
  simulation modules.
- Existing material `show`/`set`, JSON schema, mutation confirmation, and
  atomic write contracts remain unchanged.
- Existing blaze numerical behavior, checkpoint naming, progress records, and
  option names remain unchanged; this task moves command ownership only.

## Owning paths

- `src/cxr_mc/cli/__init__.py`
- `src/cxr_mc/cli/material.py`, `src/cxr_mc/cli/catalog.py`
- `src/cxr_mc/check_config.py`, `src/cxr_mc/blaze.py`
- `tests/test_cli_material.py`, catalog startup/config tests,
  `tests/test_blaze.py`, CLI completion/core tests
- `tests/data/cli_contract.json`, `docs/cli-reference.md`, command examples

## Implementation path

1. Convert `material` to a lazy group for `validate` and `blaze` while keeping
   `show`/`set` registration import-light.
2. Route `validate` and `blaze` to existing owning commands; avoid wrapper
   copies of option declarations or behavior.
3. Remove canonical root `catalog` and `blaze` registrations. Retain hidden
   aliases only if migration policy requires them.
4. Update completions, examples, generated contract/reference, and stale public
   path mentions.
5. Remove `cli/catalog.py` only if no compatibility/import owner remains.

## Verification

- Root/material help and completion tests, including optional-dependency-free
  startup.
- Existing material JSON/edit tests unchanged through grouped command.
- Catalog validation success/error/exit tests through
  `cxr material validate`.
- Blaze command/physics/checkpoint tests through `cxr material blaze`, plus a
  real `--help` and minimal smoke probe.
- Regenerate `tests/data/cli_contract.json` and `docs/cli-reference.md`; run
  focused material/catalog/blaze/CLI suites and lint.

## Non-goals

- Catalog schema/data changes, profile membership changes, blaze physics,
  checkpoint format, or merging flat and blazed execution implementations.

## Integration

Rebase onto `feature/cli-run-alignment` before implementation/landing. Root
dispatch, completion, contract snapshot, reference, and examples overlap.
