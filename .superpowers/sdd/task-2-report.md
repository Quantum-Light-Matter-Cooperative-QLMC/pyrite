# Task 2: Validation launcher report

## Red/green evidence

- Exact requested command: `python -m pytest tests/test_check.py -k tunnel -v`
  could not run because `/usr/bin/python` has no `pytest` installed.
- Repository-equivalent red command: `uv run pytest tests/test_check.py -k tunnel -v`
  collected 2 tests; both failed as intended: `_command()` rejected the unknown
  `tunnel` keyword, and argparse rejected `--tunnel`.
- Green command: `uv run pytest tests/test_check.py -k tunnel -v`
  collected 2 tests; `2 passed, 17 deselected`.

## Verification

- `uv run pytest tests/test_check.py`: `19 passed`.
- `uv run ruff check src/cxr_mc/check.py tests/test_check.py`: `All checks passed!`.
- `git diff --check`: passed with no output.

## Files changed

- `src/cxr_mc/check.py`: add conditional tunnel port handling, forwarding, and
  local SSH forwarding instructions.
- `tests/test_check.py`: add tunnel command and CLI forwarding coverage.
- `.superpowers/sdd/task-2-report.md`: this report.

## Commit

`feat: add validation tunnel launcher` (the final commit hash is available in
the worktree history).

## Self-review

- `--tunnel` only inserts Marimo's `--port 2718`; normal launches do not receive
  a fixed port.
- The launcher only prints an SSH command; it neither starts SSH nor changes the
  bind address.
- `_cli` retains its export return before construction of launch arguments, so
  `--export --tunnel` exports without launching Marimo.
- Existing Task 1 changes in `tests/test_analyze.py` were not modified.

## Concerns

- The exact system-Python test command is unavailable in this environment
  because pytest is installed only in the project uv environment; the required
  red/green verification was executed through `uv run` instead.
