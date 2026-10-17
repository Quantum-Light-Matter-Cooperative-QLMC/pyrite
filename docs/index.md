---
myst:
  html_meta:
    description: >-
      PyRITE (pyrite-mc) is a Python Monte Carlo toolkit for electron transport
      and X-ray radiation simulation, including PXR and CBS in crystals.
---

# PyRITE — Electron transport and X-ray radiation simulation

**PyRITE (`pyrite-mc`)** models electron transport in crystalline and amorphous
solids, coherent PXR/CBS lines, bremsstrahlung, and characteristic radiation.
It is research software under active development: read the
[capabilities and limitations](physics/capabilities.md) and
[validation status](validation/status-summary.md) before interpreting predictions.

- **[Install and run](guides/getting-started.md)** — one small CPU simulation and its output.
- **[Understand the physics](guides/how-pyrite-works.md)** — follow a simulation from beam to observable.
- **[Use Python](guides/python-api-workflow.md)** — build a scene and inspect an in-memory result.

## Explore results in the browser

The demo below runs in your browser with no install: HOPG PXR+CBS spectra in
the Zhai et al. (2025) Fig. 1c geometry, bulk versus 29 nm film, beside a 3D
view of the simulated electron tracks. The first load fetches a Python runtime
and can take a minute. Its provenance footer lists the PyRITE version, model
markers, and the validation status of each claim it shows.

```{raw} html
<iframe class="pyrite-demo" src="demo/index.html" title="PyRITE results demo"
  loading="lazy" sandbox="allow-scripts allow-same-origin"
  style="width: 100%; height: 900px; border: 1px solid var(--pst-color-border); border-radius: 6px;">
  <img src="demo/preview.png" alt="PyRITE results demo preview">
</iframe>
<noscript><img src="demo/preview.png" alt="PyRITE results demo preview" style="width: 100%;"></noscript>
<p><a href="demo/index.html">Open the demo full screen</a>.</p>
```

## First simulation

Requires Python 3.13 or newer and [uv](https://docs.astral.sh/uv/).
The base install is CPU-only; reference tables download about 100 MB.

```bash
uv tool install pyrite-mc
pyrite --version
pyrite tables fetch
pyrite run quickstart
pyrite checkpoint list
```

If the shell cannot find `pyrite`, run `uv tool update-shell` and restart it.
The bundled `quickstart` profile runs one HOPG case and writes resumable
component checkpoints beneath `pyrite-output/checkpoints/` in your workspace.
The [getting-started guide](guides/getting-started.md) explains setup,
result inspection, and the optional analysis app.

## Models and evidence

The [physical models](physics/index.md) describe approximations and conventions;
[computation and statistics](computation/index.md) cover sampling, uncertainty,
execution, and reproducibility. Optional secondary tracking and experimental
channeling have narrower conditions than standard transport; consult the
[capability matrix](physics/capabilities.md) for their defaults and boundaries.

Start scientific review with the [validation status summary](validation/status-summary.md),
then follow [domain inventories](validation/domain-inventories.md) to the evidence.
Independent derivation, regression coverage, and human sign-off are distinct
states; running a test does not establish experimental accuracy.
The [bibliography](references.md) collects the scientific sources.

## Find your next task

[User guides](guides/index.md) cover profiles, materials, detectors, results,
and cluster execution. [Reference](reference/index.md) contains generated CLI
help, the supported Python API, and configuration contracts.

For contributors, start with [development setup](repo-design/development-workspace.md)
and the [development index](development/index.md). Source and issue tracking
live on [GitHub](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite);
see the repository [license](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/LICENSE)
and [README](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite#readme).

```{toctree}
:maxdepth: 2
:hidden:

Getting started <guides/getting-started>
User guide <guides/index>
Physics <physics/index>
Computation <computation/index>
Validation <validation/index>
Reference <reference/index>
Development <development/index>
```
