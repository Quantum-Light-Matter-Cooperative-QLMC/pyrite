---
orphan: true
---

(your-first-python-simulation)=
# Run one simulation from Python

Use [Your first X-ray spectrum](../notebooks/fundamentals/01-first-spectrum.ipynb)
for the guided, editable example and its captured spectrum. The notebook is the
canonical learning source for constructing a beam, target and detector, running
`simulate`, and plotting PXR/CBS and bremsstrahlung.

## Run the notebook locally

Install `pyrite-mc[apps]` and
[fetch the reference tables](getting-started.md#fetch-reference-data).
Download the source from the tutorial page, then open it with marimo:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
marimo edit 01-first-spectrum.py
```

For a script or existing Python session, use the
[one-scene recipe](../reference/python-scene-recipes.md#run-one-scene-in-memory).
`simulate` returns an in-memory `Result`; it does not write a campaign checkpoint.
For a Python-driven multi-case checkpoint, follow
[Persisted parameter studies](../notebooks/designing/18-persisted-parameter-studies.ipynb);
for named profiles, follow the [CLI quickstart](getting-started.md).

## Inspect the source result

Plot each component against its own photon-energy grid. Source density has
units **photons/(incident electron · eV · sr)**; beam current, detector response
and exposure are separate operations. Follow
[Understanding results](../notebooks/fundamentals/02-understanding-results.ipynb)
for native grids, integrated yields and rates, and the
[spectral-observables explanation](../physics/radiation-physics/spectral-observables.md)
for normalization. Review [model capabilities](../physics/capabilities.md) and
[validation status](../validation/status-summary.md) before quantitative use.

The earlier fixed CPU calculation remains available as a
{download}`script <../examples/python_api_workflow.py>` with its
{download}`stored output and provenance <../examples/data/python_api_workflow.json>`
for reproduction.

(choose-physical-input-imports)=
(run-one-scene-in-memory)=
(interpret-the-result)=
(apply-detector-response-explicitly)=
(expand-a-path-addressed-sweep)=
(compare-photon-energies)=
(separate-physical-and-execution-controls)=
(choose-adaptive-sampling)=
(scalar-detector-acceptance)=
(place-a-finite-filter-over-detector-pixels)=
(compose-independent-observations)=
(select-execution-independently)=
(named-cli-pixel-profiles-use-the-same-composition)=
(reuse-bodies-as-targets-and-filters)=
(tilt-the-bunch-front-and-crystal-cut)=
(crystal-cuts)=

## More scene configurations

For detector response, observations, sweeps, adaptive sampling, filters and
crystal cuts, use the [Python scene recipes](../reference/python-scene-recipes.md).
The [Python API reference](../reference/python-api/index.md) owns generated
signatures and supported contracts; [Working with results](working-with-results.md)
covers saved results.
