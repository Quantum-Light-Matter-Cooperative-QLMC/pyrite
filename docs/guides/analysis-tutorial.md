# Analysis tutorial

This tutorial starts from an existing survey checkpoint and produces an interactive comparison. It does not rerun transport.

## 1. Find a dataset

Inspect the effective `pyrite-output/checkpoints/` directory or open the analysis app's dataset selector, then choose the identity-qualified stem produced by your run. `pyrite checkpoint list` lists only long-term archive labels, not active datasets. If no active checkpoint exists, complete the [Getting started](getting-started.md) survey first. Check that both line and bremsstrahlung components are present before comparing a total spectrum.

## 2. Launch analysis

```bash
pyrite app analysis launch hopg
```

The app reads checkpoints from the effective workspace. Select the dataset, beam energy, thickness, tilt, azimuth, and reflection controls. Keep the dataset identity visible when recording a figure: two stems with the same material label can represent different resolved inputs.

Line arrays are differential photon yield per electron, energy, and solid angle. Bremsstrahlung is stored on its own energy grid. Plotting applies the stored scale and detector metadata; do not manually multiply solid angle into arrays already represented by the result scale. See [Detector solid angle](../physics/detectors/detector-solid-angle.md).

## 3. Compare components and response

First inspect the unconvolved line and bremsstrahlung components. Then enable the configured detector response. This order helps distinguish source-model features from broadening, quantum efficiency, charge sharing, or optical throughput. The canonical stages are described in [Detector response](../physics/detectors/detector-response.md).

Survey fidelity is suitable for workflow checks and qualitative navigation, not final quantitative claims. Repeat selected cases at `full` fidelity and check numerical convergence before interpreting a peak or optimum.

## 4. Export a reproducible view

```bash
pyrite app analysis export hopg-analysis
```

This writes `pyrite-output/results/hopg-analysis.html`. Record the checkpoint stem, full identity digest, PyRITE revision, selected coordinates, response choice, and any post-processing settings beside exported figures. Static HTML is a presentation artifact, not a replacement for the component checkpoint and its provenance.

## 5. Validate the interpretation

Before making a scientific claim, locate every relevant model in the [validation ledger](../validation/physics-validation-ledger.md). In particular, separate transport, coherent line production, bremsstrahlung, absorption, and detector-response evidence. Human sign-off is distinct from successful code execution and from a visually plausible plot.

For checkpoint transforms, tables, archives, and cleanup, continue with [Working with results](working-with-results.md).
