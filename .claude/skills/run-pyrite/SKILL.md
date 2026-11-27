---
name: run-pyrite
description: Use when running, launching, smoke-testing, or confirming real PyRITE CLI, component checkpoints, plotting, or marimo behavior beyond unit tests.
---

# Run PyRITE

User surfaces: `pixi run pyrite ...`; packaged marimo apps under
`src/pyrite/apps/`; library under `src/pyrite/`. Active checkpoints:
`checkpoints/<stem>/{line,brem}.pkl`.

Use existing-checkpoint smoke path:

```bash
pixi run pyrite-dev smoke --material hopg --output-dir /tmp/pyrite-smoke
```

Use `pixi run pyrite --help` for discovery and `pixi run marimo run <app.py>` for
interactive work. Smoke requires existing checkpoint and remains outside CI.
Quick/survey scans still perform real Monte Carlo; route heavy work through
`remote-gpu-jobs`.

Own local runtime confirmation only. Unit/regression design belongs to
`regression-testing`; remote submission and monitoring belong to
`remote-gpu-jobs`.

Choose only the affected runtime probe and test modules using `AGENTS.md`
"Choose tests by impact". Do not start all tests after a local smoke check.
Lean `runtime-*` environments are for simulations, not developer checks or
plotting. Use a developer environment for the checkpoint-to-plot smoke above.

Static repository gates (no tests):

```bash
pixi run --locked verify-quick
pixi run --locked verify  # full gate including docs
```

Use `pixi run --locked doctor` for dependency-policy checks; see `AGENTS.md`
if a stale lock prevents activation.
