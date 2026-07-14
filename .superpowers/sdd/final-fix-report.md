## Final review fixes: grouping and prism-face coverage

- Added finite footprint dimensions to the default streaming group key, while
  treating absent dimensions as the legacy infinite-slab key.
- Added a `run_sweep` regression proving otherwise-identical finite footprints
  stream as separate chunks; the existing omitted-footprint group test remains.
- Added parameterized direct exit tests for all six finite-prism faces, retaining
  the existing corner, parallel-ray, and infinite-slab fallback coverage.

Verification:

```text
uv run python scripts/dev.py lint
All checks passed!

uv run ruff format --check src/cxr_mc/run.py tests/test_run.py tests/test_finite_transverse_geometry.py
3 files already formatted

uv run python scripts/dev.py test tests/test_run.py tests/test_finite_transverse_geometry.py
29 passed in 1.59s
```
