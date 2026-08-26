# chore/shim-lifecycle-enforcement

Issue: [#68](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues/68)
Worktree: `/home/alex/dev/wt/pyrite/shim-lifecycle-enforcement`

## Problem

Audited at `0bb4a075`, `pyrite-xray` 0.3.0. Three deprecation registries, 178
rows, and **142 of them carry `remove_in == "0.3.0"` while the package is at
0.3.0**. Nothing failed, because the support-window tests assert only internal
arithmetic:

```python
# _entry() produced the value with _window(since) ...
def test_deprecation_support_window() -> None:
    for path, dep in DEPRECATIONS.items():
        assert dep.remove_in == _window(dep.deprecated_in)  # ... same function
```

No test in the repo imports `__version__`. The schedule cannot fail as releases
go by.

## Shim inventory

| Family | Location | Rows | Deprecated | Remove in | Warns |
| --- | --- | --- | --- | --- | --- |
| CLI commands | `cli/_deprecations.py` → `DEPRECATIONS` | 68 | 0.1.0 | 0.3.0 | stderr |
| CLI options | `cli/_deprecations.py` → `DEPRECATED_FLAGS` | 74 | 0.1.0 | 0.3.0 | stderr |
| Module re-exports | `_module_deprecations.py` | 36 / 168 LOC | 0.2.0 | 0.4.0 | never |
| D7 scene adapter | `campaign/legacy.py` | 135 LOC | — | unset | public door only |
| Checkpoint v1 reader | `checkpoints/_checkpoint_io.py` | ~30 LOC | — | never (by design) | n/a |

Shim code is cheap (36 modules = 168 LOC). The governance apparatus is 1,243 LOC
and does not enforce the one thing it exists to enforce.

## Implementation path

Land in this order — step 1 is what makes the rest verifiable.

1. **Enforcement test.** Compare every `remove_in` against `__version__` across
   all three registries. Expect it to fail immediately on the 142 overdue rows;
   that failure is the point. Land it together with step 3 so the tree stays
   green, or gate it behind an explicit xfail for one commit.
2. **Module-shim warnings.** Module-level `__getattr__` warning once per path,
   driven from `MODULE_DEPRECATIONS` (which currently has no runtime consumer at
   all). Restart the two-minor clock from the first warning release; drop the
   0.4.0 target.
3. **Remove the overdue CLI cohort.** 68 commands + 74 flags + their aliases,
   one breaking change, regenerating `docs/repo-design/cli/cli-deprecations.md`
   in the same commit. Existing bidirectional tests catch orphans in both
   directions.
4. **Split the D7 adapter.** Rename the internal `build_legacy_cases` path
   (`runs/scan.py:82`, `runs/blaze.py:74` — production, not legacy). Separately
   migrate `apps/scan_app.py:129` and `apps/trace_app.py:440,447` off
   `Sweep.from_legacy()`, then unpin `tests/notebooks/test_scan_app.py:25` and
   `tests/notebooks/test_trace_app.py:123`, which assert the exact source string.
5. **`_compat.py` → `_env.py`.** 14 importers. CXR logic was removed in
   `681ae0f2`; only `os.environ` wrappers remain under a misleading name.
6. **Stale pointers.** `cli/_deprecations.py` cites a nonexistent
   `tests/test_cli_deprecations.py`; generated table credits
   `scripts/generate_cli_deprecations.py` vs `AGENTS.md`'s
   `pyrite-dev cli-deprecations`.

## Decisions

- The module cohort is the one family to **extend, not enforce**. It has never
  emitted a warning, so removing it on schedule is an unannounced break.
- `campaign/legacy.py` is not deletable. Only the public bridge retires.
- Checkpoint v1 reader is correct as-is and out of scope. It is also the model:
  it states its lifecycle in the code it governs.

## Checks

```bash
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test-suite cli
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev test tests/test_module_deprecations.py
UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev verify
uv run marimo check src/pyrite/apps/scan_app.py src/pyrite/apps/trace_app.py
```
