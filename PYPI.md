# PyRITE

![PyRITE](https://raw.githubusercontent.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/main/src/pyrite/_assets/branding/pyrite-wordmark.png)

PyRITE is a Python toolkit for Monte Carlo electron transport and X-ray
emission in crystalline and amorphous solids. It models parametric X-ray
radiation (PXR) and coherent bremsstrahlung (CBS) from crystals, alongside
incoherent bremsstrahlung and characteristic X-ray emission.

Use the Python API for individual simulations or the command-line interface
for resumable parameter sweeps. PyRITE includes material catalogs, configurable
beams and detectors, trajectory export, and interactive analysis apps.
CPU execution is included; NVIDIA GPU acceleration is available through the
`nvidia` extra.

## Install

Requires Python 3.13 or newer. The [installation guide](https://pyrite.readthedocs.io/en/latest/guides/getting-started.html#install-pyrite-on-cpu)
offers uv (recommended) and pip instructions, followed by GPU and extras options.

```bash
pip install pyrite-mc
pyrite tables fetch
pyrite run quickstart
```

Pint is included for physical quantity inputs through `pyrite.units`; numeric
configuration and simulation storage retain their documented units.

Reference physics tables are downloaded separately by `pyrite tables fetch`.
For NVIDIA GPUs, install `pyrite-mc[nvidia]` with compatible CUDA drivers.
For `pyrite.plots`, install `pyrite-mc[plots]`; for the interactive marimo apps
(`pyrite app ...`), install `pyrite-mc[apps]`.

## Research status and license

PyRITE is unvalidated research software under active development.

Distributed under the
[UCLA Academic Software License](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/LICENSE.txt)
for educational or academic research use by academic or nonprofit researchers.
Commercial use requires separate licensing.

[Documentation](https://pyrite.readthedocs.io/en/latest/) ·
[Source code](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite)
