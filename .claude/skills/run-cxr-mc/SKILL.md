---
name: run-cxr-mc
description: Use when running, launching, smoke-testing, or confirming real cxr-mc CLI, component checkpoints, plotting, or marimo behavior beyond unit tests.
---

# Run cxr-mc

User surfaces: `uv run cxr ...`; four marimo apps under `notebooks/`; library
under `src/cxr_mc/`. Active checkpoints:
`checkpoints/<stem>/{line,brem}.pkl`.

Use existing-checkpoint smoke path:

```bash
uv run python scripts/dev.py smoke --material hopg --output-dir /tmp/cxr-mc-smoke
```

Use `uv run cxr --help` for discovery and `uv run marimo run <app.py>` for
interactive work. Smoke requires existing checkpoint and remains outside CI.
Quick/survey scans still perform real Monte Carlo; route heavy work through
`remote-gpu-jobs`.

Full repository gate:

```bash
uv run python scripts/dev.py verify
```
