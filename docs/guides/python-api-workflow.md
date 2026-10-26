# Your first Python simulation

Simulate 30 keV electrons crossing a 100 nm HOPG target, then plot the
raw emitted X-ray spectra at 135° to the incident beam.

Install `pyrite-mc[plots]` and [fetch the reference tables](getting-started.md#install-and-fetch-reference-data).
This small CPU example uses 32,768 line-scoring and 128 continuum-scoring electron trials;
first use also compiles CPU kernels. Download {download}`the complete script <../examples/python_api_workflow.py>`
and run it in your installed environment:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
PYRITE_MC_BACKEND=cpu python python_api_workflow.py
```

```{literalinclude} ../examples/python_api_workflow.py
:language: python
:lines: 3-
```

```{eval-rst}
.. plot::
   :alt: Simulated HOPG source spectrum with narrow PXR/CBS peaks and a broad bremsstrahlung continuum.
   :caption: Actual simulation output, precomputed from the downloadable script.

   from _tutorials import python_api_workflow
   python_api_workflow()
```

The narrow PXR/CBS structure sits above a broad bremsstrahlung background.
The total adds those two components after interpolating bremsstrahlung onto
the line grid; it excludes characteristic emission.
The vertical axis is **photons/(incident electron · eV · sr)**: no beam-current,
solid-angle, exposure or detector-response scaling has been applied. Target
thickness is in **Å**, beam dimensions in **mm**, and angles in **degrees**.
`simulate` returns an in-memory result without writing a checkpoint.

Single-scene simulation uses a fixed resolved seed (recorded in `data["seed"]`);
repeatability also depends on the backend, software and reference tables. The
{download}`stored output and provenance <../examples/data/python_api_workflow.json>` record
this example's configuration and identity. Review the
[capability matrix](../physics/capabilities.md) and
[validation status](../validation/status-summary.md) for model limitations.

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
The [supported API](../api.md) owns signatures and defaults;
[Working with results](working-with-results.md) covers saved results.
