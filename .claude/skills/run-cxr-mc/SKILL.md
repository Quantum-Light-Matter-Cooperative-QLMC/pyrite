---
name: run-cxr-mc
description: Run, drive, or screenshot the cxr-mc app — the `cxr` CLI (scan/export/slim/archive), the marimo analysis & scan notebook apps, and the physics+plotting library. Use when asked to run, launch, build, test, smoke-test, or screenshot cxr-mc, or to confirm a change works in the real app (figures render, checkpoints load) rather than only in unit tests.
---

# Run cxr-mc

`cxr-mc` is a coherent-X-ray Monte Carlo simulation. It has three surfaces:

- **`cxr` CLI** (`pyproject.toml [project.scripts]`) — headless: `scan` (run an
  MC sweep → `checkpoints/<material>.pkl`), plus `export`, `slim`, `archive`,
  `restore`, `archives`.
- **marimo notebook apps** — `notebooks/analysis_app.py` (load a checkpoint,
  draw every figure) and `notebooks/scan_app.py` (run sweeps interactively).
  These are web GUIs.
- **the importable `cxr_mc` package** — physics + plotting the two above call.

Most changes here touch the physics/plotting internals, not UI chrome. The
fastest way to *see a change working* is the **direct-invocation driver**
below: it loads a real checkpoint and renders the same figures the analysis app
draws, straight to PNG — no browser, no live kernel.

All paths below are relative to the repo root (`<unit>/`). Everything runs on
Windows (this repo's host); the MC falls back to CPU with no GPU present.

## Prerequisites

- `uv` (the project's package manager) and Python 3.13 (`.python-version`).
- No GPU needed — `cupy` import fails soft and the backend uses NumPy (CPU).
- For the GUI-screenshot path only: a Chromium browser. This host has
  `C:\Program Files\Google\Chrome\Application\chrome.exe`.

## Build

```bash
uv sync
```

Resolves/installs all 251 packages into `.venv`. Fast when already synced.

## Run — agent path (the driver)

The driver mirrors `analysis_app.py`'s data flow (load checkpoint → rebuild
cases → filter → plot) and writes PNGs + a ranking table:

```bash
uv run python .claude/skills/run-cxr-mc/smoke.py hopg "$TEMP/smoke_out"
```

- Arg 1 = material (any checkpoint under `checkpoints/`: `hopg`, `mos2`,
  `mose2`, `sapphire`, `wse2`, …). Default `hopg`.
- Arg 2 = output dir. Default `smoke_out/`.
- Writes `<material>_spectrum.png` (Altair intrinsic spectrum, via vl-convert)
  and `<material>_best_spectra.png` (matplotlib top-6 grid), prints the
  top-geometry table, exits 0 on success (2–4 if a checkpoint/figure is empty).

Verified output on `hopg`: `400 records, 10 tilts`, both PNGs written
(~155 KB / ~100 KB), `[smoke] OK`. The best-spectra grid shows the expected
sharp PXR lines riding the bremsstrahlung continuum.

## Run — CLI

```bash
uv run cxr --version          # cxr-mc 0.1.0
uv run cxr --help             # scan / export / slim / archive / restore / archives
uv run cxr archives           # list the long-term checkpoint shelf
uv run cxr scan --help
```

Run a real MC sweep (writes `checkpoints/<material>.pkl`):

```bash
uv run cxr scan mose2 --quick --checkpoint-dir "$TEMP/cxr-scan"
```

`--quick` is a "tiny" grid (20 cases). **On CPU this is still slow — roughly
90–135 s per case, ~30 min total.** It IS a real sweep, not a stub; use it only
when you specifically need fresh checkpoint data. To watch a change to the
physics render, prefer the driver above against an existing checkpoint. GPU is
the intended path for full sweeps.

## Run — GUI (marimo apps)

**To see the actual app UI (screenshot):** export the notebook to static HTML
(this executes every cell server-side with real Python — no live kernel — and
bakes the outputs in), then screenshot that file with headless Chrome:

```bash
uv run marimo export html notebooks/analysis_app.py -o "$TEMP/analysis.html"
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless=new --disable-gpu \
  --hide-scrollbars --screenshot="$(cygpath -w "$TEMP/analysis-html.png")" \
  --window-size=1400,3200 --virtual-time-budget=30000 \
  "file:///$(cygpath -m "$TEMP/analysis.html")"
```

Verified: the export logs `loaded 400 hopg records from …checkpoints\hopg.pkl`
and the PNG shows the rendered app (title, Material dropdown = `hopg`, tilt
selector, brem/x-limit/log-y controls). `scan_app.py` exports the same way.

**Live interactive server (human path):**

```bash
uv run marimo run notebooks/analysis_app.py --headless --host 127.0.0.1 --port 2718 --no-token
```

Opens a real editable app at `http://localhost:2718`. Useful when a human drives
it in a real browser; see Gotchas for why a headless *screenshot* of the live
server does not work.

## Test

```bash
uv run pytest        # ~241 tests, all pass (CPU, fast)
uv run ruff check .  # lint
uv run pyright       # type check
```

The heavy physics-validation runs live in `checks/`, not `pytest`.

## Gotchas

- **A headless screenshot of the *live* `marimo run` server shows "kernel not
  found".** The app needs a WebSocket kernel session that headless Chrome's
  `--virtual-time-budget` capture never finishes establishing (raising the
  budget doesn't help). Use the `marimo export html` → screenshot path above
  instead — it renders server-side, so there's no live kernel to miss.
- **`marimo export html` includes code cells** and leaves the expensive
  figures in `lazy` tabs (they compute only on expand), so the exported HTML
  shows the app's controls with real data loaded but not every plot. To see the
  actual figures headlessly, use the driver (`smoke.py`) — it calls the plot
  functions directly.
- **Windows DLL order: import `pyarrow` before `cxr_mc`.** Importing `cxr_mc`
  (numpy/scipy DLLs) first can leave pyarrow's Arrow DLLs bound to the wrong
  runtime and segfault. `tests/conftest.py` already does this for the suite; if
  you write a *standalone* script that uses pyarrow-backed Altair data
  transforms, `import pyarrow` at the very top. (The driver avoids this — it
  renders via vl-convert, not a pyarrow data transform.)
- **`cxr scan --quick` is a real sweep, not a smoke stub** — it will run for
  ~30 min on CPU. Don't use it as a quick liveness check; use `cxr --version`
  or `cxr archives` for that, and the driver to check figures.
- **No GPU is fine.** `montecarlo._backend` reports `_GPU=False, REAL=float64`
  and runs on NumPy; no cupy/CUDA install is required to run anything here.

## Troubleshooting

- **`cupy` / CUDA import errors on startup** — expected and harmless; the
  backend falls back to CPU. Nothing to fix.
- **Driver exits 2 ("no records")** — that material has no checkpoint. Run
  `uv run cxr archives` / `ls checkpoints/` and pass a material that exists.
- **Chrome screenshot is a mangled/blank page** — build the `file://` URL with
  `cygpath -m` (forward-slash Windows path), not `cygpath -w` (backslashes,
  which the shell mangles).

## The harness

`smoke.py` (in this skill dir) is the driver. It imports the exact functions
`analysis_app.py` imports (`load_checkpoint`, `cases_from_results`,
`filter_results`, `spectrum_chart`, `plot_best_spectra`, `top_geometries`) and
renders to `Agg`/vl-convert, so it runs headless. If the project's own e2e
suite ever wants it, graduate it to `scripts/` and update the path above.
