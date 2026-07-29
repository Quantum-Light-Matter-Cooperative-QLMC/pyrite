# Move prune under checkpoint

Branch: `feature/checkpoint-prune`

TODO scope: P1 CLI Work sub-item 7.

## Goal

Make `cxr checkpoint prune` the canonical local prune path.

## Decisions

- Add lazy `prune` registration to `cxr checkpoint`.
- Remove `prune` from public root help. Keep a hidden compatibility alias only
  if current migration policy requires it; canonical docs/examples use the
  grouped path.
- Preserve preview, confirmation, `--all`/`--profile` exclusivity, atomic
  rewrite, stdout/stderr, and exit contracts unchanged.
- Remote pruning remains `cxr remote prune`; no lifecycle changes.

## Owning paths

- `src/cxr_mc/cli/__init__.py`
- `src/cxr_mc/cli/checkpoint.py`
- `src/cxr_mc/prune.py`
- `tests/test_prune.py`, root/checkpoint CLI and completion tests
- `tests/data/cli_contract.json`, `docs/cli-reference.md`

## Implementation path

1. Register `prune` in checkpoint lazy command/help tables.
2. Adjust root registration according to the compatibility decision; update
   command examples and diagnostics to canonical path.
3. Update completion, CLI contract, reference, and repository docs containing
   live command examples.
4. Search for stale public `cxr prune` references.

## Verification

- Help/dispatch tests for `cxr checkpoint prune`, including lazy imports.
- Existing prune behavioral suite through new path: preview, confirmation,
  `--yes`, profile/all selection, failure and atomicity.
- Compatibility-path test if retained.
- Regenerate `tests/data/cli_contract.json` and `docs/cli-reference.md`; run
  focused prune/checkpoint/CLI tests and lint.

## Non-goals

- Pruning algorithm, checkpoint identity, remote prune, or storage layout
  changes.

## Integration

Can land independently. Rebase before regenerating CLI artifacts if another
CLI branch lands first.
