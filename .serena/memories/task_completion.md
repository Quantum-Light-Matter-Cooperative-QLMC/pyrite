# Task completion

- Run the smallest focused canonical check first, then neighboring suites and runtime probes proportional to risk.
- Use `UV_CACHE_DIR=/tmp/cxr-mc-uv-cache uv run cxr-dev lint`, `typecheck`, and broader `verify` when warranted.
- CLI changes: test affected help/dispatch/completion/output/exit contracts and regenerate/check `docs/cli-reference.md`.
- Marimo edits: run `uv run marimo check <app.py>`; strip legacy notebook output.
- Before commit: inspect scoped status/diff, stage explicit paths, and commit only independently valid work. Do not push without authority.
