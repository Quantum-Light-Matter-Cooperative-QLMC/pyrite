---
name: run-pyrite
description: Use when running, launching, smoke-testing, or confirming real PyRITE CLI, component checkpoints, plotting, or marimo behavior beyond unit tests.
---

# Run PyRITE

User surfaces: `uv run pyrite ...`; four packaged marimo apps under
`src/pyrite/apps/`; library under `src/pyrite/`. Active checkpoints:
`checkpoints/<stem>/{line,brem}.pkl`.

Use existing-checkpoint smoke path:

```bash
uv run pyrite-dev smoke --material hopg --output-dir /tmp/pyrite-smoke
```

Use `uv run pyrite --help` for discovery and `uv run marimo run <app.py>` for
interactive work. Smoke requires existing checkpoint and remains outside CI.
Quick/survey scans still perform real Monte Carlo; route heavy work through
`remote-gpu-jobs`.

Own local runtime confirmation only. Unit/regression design belongs to
`regression-testing`; remote submission and monitoring belong to
`remote-gpu-jobs`.

Full repository gate:

```bash
uv run pyrite-dev verify
```
