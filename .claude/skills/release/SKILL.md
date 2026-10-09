---
name: release
description: Use when cutting a PyRITE release PR, covering version bump, due deprecation removals, generated docs, and release notes via pyrite-dev release.
---

# Release

Policy: `docs/repo-design/releasing.md`. Versions bump only in a release PR
titled `chore(release): bump version to X.Y.Z`; never mix with feature work.

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev release X.Y.Z --check
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev release X.Y.Z --notes-file <scratch>/notes.md
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
```

1. `--check` lists deprecation removals due at or before X.Y. Remove each
   (shim, call sites, tests, docs) in earlier commits; no release while due.
2. Run without `--check`: bumps `pyproject.toml`, `src/pyrite/__init__.py`,
   and `uv.lock`, regenerates CLI reference/deprecations, prints notes since
   last `v*` tag (else last release commit; `--base REV` overrides).
3. Review notes: breaking and physics-changing sections are heuristic.
4. Commit only those paths as the release commit; notes go in PR body.

Never tag or push without explicit human authorization. No PyPI publishing.
