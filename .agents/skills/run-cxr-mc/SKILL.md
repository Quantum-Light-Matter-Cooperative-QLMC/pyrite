---
name: run-cxr-mc
description: Use when running, launching, smoke-testing, or confirming real cxr-mc CLI, checkpoint, plotting, or marimo-app behavior beyond unit tests.
---

# Run cxr-mc

cxr-mc has three user-facing surfaces:

- `uv run cxr ...`: headless scan, export, slim, and archive commands.
- `notebooks/scan_app.py`, `notebooks/analysis_app.py`, and
  `notebooks/validation_app.py`: marimo applications.
- `src/cxr_mc/`: the importable physics and plotting library.

## Fast path

Use the headless smoke harness against an existing checkpoint. It follows the
analysis app's checkpoint-to-figure path and writes an Altair spectrum plus a
matplotlib best-spectra grid:

```bash
uv run python scripts/dev.py smoke --material hopg --output-dir /tmp/cxr-mc-smoke
```

Use `uv run cxr --help` for CLI discovery and `uv run marimo run <app.py>` for
interactive app work. A quick scan still performs real Monte Carlo transport;
do not use it as a liveness stub.

## Verification

```bash
uv run python scripts/dev.py verify
```

No GPU is required; the backend falls back to NumPy. The smoke command requires
an existing `checkpoints/<material>.pkl` artifact and remains outside CI.
