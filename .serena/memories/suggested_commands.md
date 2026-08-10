# Suggested commands

- Always use `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite-dev <command>`; never bare pytest, venv Python, path-hack imports, or RTK wrappers.
- Tests: `test`; suites: `test-suite core|cli|apps|packaging`; focus: `test path/to/test.py -k test_name`.
- Quality: `lint`, `format`, `typecheck`, `verify`, `precommit`; notebooks: `nbstrip`.
- Real CLI: `UV_CACHE_DIR=/tmp/pyrite-uv-cache uv run pyrite ...`; marimo validation: `uv run marimo check <app.py>`.
- If the project environment is unwritable, add `UV_PROJECT_ENVIRONMENT=/tmp/pyrite-venv`; do not switch interpreters.
- Set `PYRITE_ONLINE_TESTS=1` only for online tests. Route heavy/GPU work through `pyrite remote`.
