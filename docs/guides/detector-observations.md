# From emitted photons to pixel counts

Predict an ideal detector image, then compare it with a simulated counting
realization from the same source. This example separates physical detector
geometry, readout response and exposure without rerunning transport.

Install `pyrite-mc[plots]` and [fetch the reference tables](getting-started.md#fetch-reference-data).
Download {download}`the complete runnable script <../examples/detector_observations.py>`:

<!-- verify: skip (external shell tools or downloadable Python example) -->
```bash
PYRITE_MC_BACKEND=cpu python detector_observations.py
```

The beam sends 30 keV electrons into 100 nm HOPG. A 16 × 16 mm ideal panel sits
100 mm from the source at a lab polar angle of 135° (0° follows the incoming
beam). Its 16 × 16 pixels have 1 mm pitch. The 1 MHz cadence and 0.01 pC bunch
charge give **10 nA**, or about 6.24 × 10¹⁰ incident electrons in the 1 s exposure.
Those physical electrons are distinct from the 64 histories sampled for each
source component.

```{literalinclude} ../examples/detector_observations.py
:language: python
:start-after: tutorial-start
:end-before: tutorial-end
```

`calculate()` returns the source result and one acquired batch. The script's
plotting helper reshapes `batch.expected.sum(axis=-1)` and `batch.total_counts`
to `(16, 16)`. The former retains fractional expectations; the latter sums the
integer Poisson draws across reporting bins and selected components.

```{eval-rst}
.. plot::
   :alt: Expected counts vary weakly across the ideal panel, while a seeded simulated count image fluctuates from pixel to pixel on the same count scale.
   :caption: Actual seeded CPU simulation: expected and realized ideal-counter counts in the half-open measured-energy interval [0.5, 4) keV during 1 s. Both panels share a color scale. These are simulated observations, not experimental measurements.

   from _tutorials import detector_observations
   detector_observations()
```

The nearly uniform expectation has a small angular variation; the stronger
pixel fluctuations in the second panel come from counting statistics. They do
not demonstrate fine angular structure. The acquisition seed `23` controls this
count realization; source transport resolves seed `1`. Changing a count seed
does not create an independent source simulation.

## What has been counted?

`IdealPhotonCounter()` registers one event per incident photon with unit
efficiency and no energy broadening. It is a deliberate idealization, not a
Timepix hardware model. `Observation` adds that readout, the exposure, reporting
edges and angular sampling to the geometry-only `Scene`.

`result.acquire(...)` includes PXR + CBS (`cxr`), bremsstrahlung and characteristic
emission explicitly. Source densities remain response-free; pixel acquisition
already integrates pixel solid angle, applies the response and multiplies by
physical source electrons. **Do not multiply these counts by solid angle or
beam current again.** This example has no downstream filters. Finite filters,
when configured in the scene, attenuate pixel rays before response and counting.

Only registered events within the selected reporting interval appear in the
images. Underflow, overflow and below-threshold expectations remain separate
in the batch; this is not the all-energy photon yield. For units, response
models and the accounting contract, see
[Detector response](../physics/detectors/detector-response.md#native-measured-bins-and-counting-acquisition).

## Resolution and limits

The 4 × 4 angular tiles reuse one source spectrum per tile across 16 × 16
pixel-centre rays. A fine pixel grid does not imply equally fine angular scoring.
The 64 source histories and two reflection families keep this example small;
they do not establish convergence. The displayed Poisson fluctuations exclude
uncertainty in the estimated source expectation, detector calibration and
external backgrounds. Refine source histories, angular tiles, energy grids and reflection
coverage before quantitative use.

The {download}`cached output and provenance <../examples/data/detector_observations.json>`
record actual counts, both seeds, models, table identity and observation settings.
Documentation builds render only that checked output. For finite filters,
independent observations and native reporting bins, see
[Python scene recipes](../reference/python-scene-recipes.md#compose-independent-observations).
