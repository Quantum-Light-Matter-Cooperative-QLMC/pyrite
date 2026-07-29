# Group interactive apps and exports

Branch: `feature/cli-app-suite`

TODO scope: P1 CLI Work sub-item 6.

## Goal

Replace scattered top-level app commands with one hierarchy:

```text
cxr app analysis
cxr app analysis export
cxr app viewer
cxr app viewer export
cxr app validation
cxr app validation export
```

Invoking an app leaf without `export` launches it. Each `export` writes a
non-interactive artifact and never starts a browser/server.

## Decisions

- Top-level group name: `app`; leaf names: `analysis`, `viewer`, `validation`.
- Canonical launch paths above replace top-level `analyze`, `viewer`, and
  `validate`.
- Export semantics must be explicit per leaf:
  - analysis: existing static HTML export.
  - viewer: static HTML export using the viewer marimo app.
  - validation: preserve existing cached validation-figure export; name output
    type in help and keep the operation non-interactive.
- Preserve result/progress stream separation and existing artifact overwrite
  safeguards.
- Keep lazy imports: `cxr --help` and `cxr app --help` must not import marimo,
  plotting stacks, GPU backends, or open network connections.

## Owning paths

- Root/group dispatch: `src/cxr_mc/cli/__init__.py`, new
  `src/cxr_mc/cli/app.py`
- Analysis launch/export: `src/cxr_mc/analyze.py`, `src/cxr_mc/export.py`
- Viewer launch/export: `src/cxr_mc/viewer.py`
- Validation launch/export: `src/cxr_mc/check.py`
- Apps: `notebooks/analysis_app.py`, `notebooks/trace_app.py`,
  `notebooks/validation_app.py`
- Tests/contracts: `tests/test_analyze.py`, `tests/test_export.py`,
  `tests/test_check.py`, `tests/test_viewer.py`, CLI completion/core tests,
  `tests/data/cli_contract.json`
- Generated docs: `docs/cli-reference.md`

## Implementation path

1. Inventory launch arguments, persisted defaults, edit/watch behavior,
   smoke modes, artifact types, overwrite policy, exit codes, and streams for
   all three apps.
2. Add lazy nested `app` groups. Use `invoke_without_command` only at app leaves
   so bare leaf invocation launches while `export` dispatches cleanly.
3. Move existing launch behavior without duplicating orchestration logic.
4. Reuse analysis export plumbing for viewer where compatible; keep validation
   figure export behind a leaf-specific adapter.
5. Remove canonical top-level registrations and update completion/examples.
6. Regenerate CLI contract/reference and remove stale public paths.

## Verification

- Root/group/leaf help remains import-light and stable without optional app
  dependencies installed.
- `CliRunner` coverage for launch-vs-export dispatch, argument forwarding,
  stdout/stderr, failures, overwrite policy, and no-browser export behavior.
- `uv run marimo check` for every touched app.
- Real smoke probes for all launch paths; static export probes to temporary
  output directories.
- Regenerate `tests/data/cli_contract.json` and `docs/cli-reference.md`; run
  focused app/CLI suites and lint.

## Non-goals

- App UI redesign, simulation/plot physics changes, or output-format
  convergence between HTML and validation figure bundles.

## Integration

Rebase onto `feature/cli-run-alignment` after it lands because both edit root
dispatch, completion, contract snapshot, and generated reference.
