# PyRITE

![PyRITE](src/pyrite/_assets/branding/pyrite-wordmark.png)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23487182.svg)](https://doi.org/10.5281/zenodo.23487182)  [![PyPI](https://img.shields.io/pypi/v/pyrite-mc)](https://pypi.org/project/pyrite-mc/) [![Documentation](https://readthedocs.org/projects/pyrite/badge/?version=latest)](https://pyrite.readthedocs.io/en/latest/) ![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FQuantum-Light-Matter-Cooperative-QLMC%2Fpyrite%2Fmain%2Fpyproject.toml) [![Downloads](https://img.shields.io/pypi/dm/pyrite-mc?label=downloads%2Fmonth)](https://pypistats.org/packages/pyrite-mc) [![License](https://img.shields.io/badge/License-UCLA-blue)](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/LICENSE.txt)

**Simulate how electrons travel through matter—and the X-rays they produce.**

PyRITE (**Py**thon toolkit for **R**adiation from **I**nteractions and **T**ransport of **E**lectrons) is a Python-based Monte Carlo simulation toolkit for electron transport and radiation in solids. It is designed for researchers studying electron–matter interactions, crystal-based X-ray sources, and related experiments.

Its particular focus is **parametric X-ray radiation (PXR)** and **coherent bremsstrahlung (CBS)**: radiation produced when electrons interact with the ordered structure of a crystal. PyRITE brings these specialized calculations together with electron transport and other X-ray emission processes in one workflow.

**[Try the interactive demo](https://pyrite.readthedocs.io/en/latest/demo/index.html)** · **[Read the documentation](https://pyrite.readthedocs.io/en/latest/)** · **[Explore the examples](https://pyrite.readthedocs.io/en/latest/guides/getting-started.html)**

[![Example PyRITE spectra and simulated electron trajectories](https://pyrite.readthedocs.io/en/latest/demo/preview.png)](https://pyrite.readthedocs.io/en/latest/demo/index.html)

*Example radiation spectra and electron trajectories. Click the image to explore the browser demo—no installation required.*

## What can PyRITE do?

- **Follow electrons through materials.** Model scattering and energy loss in crystalline and amorphous solids.
- **Predict X-ray emission.** Calculate PXR, CBS, bremsstrahlung, and characteristic X-ray spectra.
- **Explore experimental designs.** Vary electron beams, materials, crystal orientations, target thicknesses, and detector configurations.
- **Work the way you prefer.** Run simulations from the command line, use a Python API, or explore results through interactive applications.
- **Scale up when needed.** Start with a CPU simulation; optional GPU backends and workflows for larger studies are available.

PyRITE is especially useful for investigating **tunable, crystal-based X-ray sources**, including how electron scattering and background radiation affect the spectra that an experiment might measure. Unlike general-purpose transport codes such as PENELOPE and Geant4, its central focus includes native PXR/CBS calculations.

## Why Python?

Python makes simulation models **readable, accessible, and easy to modify or extend** when compared with simulations written in FORTRAN, C++, or other compiled languages. Its broad open-source ecosystem provides mature tools for numerical methods, data analysis, visualization, and reproducible research.

While base Python is far slower than its compiled colleagues, it acts primarily as a more intelligible wrapper around libraries which are far faster. **[Numba](https://numba.pydata.org/)** is used to JIT-compile performance-critical CPU code to native machine code, while **[CUDA](https://developer.nvidia.com/cuda/toolkit) and [CuPy](https://cupy.dev/)** allow us to write practical, high performance GPU acceleration code without rewriting large portions of PyRITE in C++ or Fortran. This combines a flexible research interface with highly optimized computational kernels, which are ideal given the [embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel) nature of Monte Carlo simulations.

Altogether, this makes for a far more accessible software to the uninitiated (especially given the ever-increasing popularity of Python) while retaining the raw performance of traditional compiled transport codes such as geant4.

## Try it in your browser

The **[interactive results demo](https://pyrite.readthedocs.io/en/latest/demo/index.html)** lets you explore sample X-ray spectra and electron trajectories without installing Python or downloading any datasets. It's the easiest way to see what PyRITE produces.

## Get started locally

To install the command-line tool with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install pyrite-mc
```

Download the reference data, then run the bundled quickstart example:

```bash
pyrite tables fetch
pyrite run quickstart
```

The reference data download is approximately **100 MB**. The quickstart is a small CPU simulation of X-ray emission from highly oriented pyrolytic graphite (HOPG); results are saved under `pyrite-output/`.

To use PyRITE as a Python library instead, install it in your environment:

```bash
pip install pyrite-mc
```

Pint is included for physical quantity inputs through `pyrite.units`; numeric
configuration and simulation storage retain their documented units.

For selectable **uv/pip installation instructions**, GPU and other extras, and your first results, see the **[getting-started guide](https://pyrite.readthedocs.io/en/latest/guides/getting-started.html)**. For library usage, see the **[Python API guide](https://pyrite.readthedocs.io/en/latest/guides/python-api-workflow.html)**.

## Research status and validation

> [!IMPORTANT]
> **PyRITE is research software under active development, not a fully validated radiation-transport code.** Different models and features have different levels of testing and physical validity. Check the [capability matrix](https://pyrite.readthedocs.io/en/latest/physics/capabilities.html) and [validation status](https://pyrite.readthedocs.io/en/latest/validation/status-summary.html) before relying on quantitative predictions.

PyRITE's development emphasizes transparent physical assumptions, reproducible calculations, and comparison with theory and published results. The [physics documentation](https://pyrite.readthedocs.io/en/latest/physics/index.html) explains the models and their limitations.

## Get involved

PyRITE is developed at UCLA's Quantum Light–Matter Cooperative (QLMC). Feedback from researchers—especially comparisons with analytical calculations, other simulation codes, or experimental measurements—is welcome.

If you encounter a problem, have a feature request, or would like to discuss a possible collaboration, [open an issue](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/issues) or start a [discussion](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/discussions).

## License

PyRITE is distributed under the [UCLA Academic Software License](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/blob/main/LICENSE.txt) for educational and academic research by eligible academic or nonprofit researchers. **Commercial use requires separate permission**; see the license for details.

---

**[Documentation](https://pyrite.readthedocs.io/en/latest/)** · **[Interactive demo](https://pyrite.readthedocs.io/en/latest/demo/index.html)** · **[PyPI](https://pypi.org/project/pyrite-mc/)** · **[Releases](https://github.com/Quantum-Light-Matter-Cooperative-QLMC/pyrite/releases)**
