(from-emitted-photons-to-pixel-counts)=
# Acquire expected and realized pixel counts

Use [From X-rays to detector counts](../notebooks/designing/06-detector-counts.md)
for the guided, editable example, its complete public-API source and captured
images. This recipe assumes a physical beam and target are already configured.

## Configure and acquire

1. Construct a `PlanarDetector` with a `PlanarPose` and `PixelGrid`, and put it
   in the geometry-only `Scene`. Set distance and pixel pitch in **mm** and
   pose angles in **degrees**; these determine pixel rays and solid angles.
2. Construct an `Observation` selecting that detector, a response model,
   `AngularSampling` and an `Acquisition`. Declare exposure in **s**, measured
   energy-bin edges in **eV**, and a seed for `mode="poisson"`.
3. Pass `observation=observation` to `simulate`. Keep physical bunch charge and
   repetition rate separate from the sampled transport histories.
4. Call `result.acquire(region=(slice(None), slice(None)),
   components=("cxr", "brem", "characteristic"))` for the complete panel and
   explicitly selected source components. Reshape `batch.expected.sum(axis=-1)`
   and `batch.total_counts` to the panel's pixel shape to compare fractional
   expectations with integer realized counts.

Use the [independent-observation recipe](../reference/python-scene-recipes.md#compose-independent-observations)
for runnable construction and repeated acquisitions from one source result.
The [observations API reference](../reference/python-api/observations.md)
owns signatures; [finite-filter composition](../reference/python-scene-recipes.md#place-a-finite-filter-over-detector-pixels)
covers plates between source and detector.

## What has been counted?

`IdealPhotonCounter()` registers one event per incident photon with unit
efficiency and no energy broadening. It is an idealization, not a Timepix
hardware model. Pixel acquisition integrates pixel solid angle, applies the
response and multiplies by the physical source-electron population.
**Do not multiply acquired counts by solid angle or beam current again.**
Configured finite filters attenuate pixel rays before response and counting.

Only events in the selected reporting interval appear in the images. Underflow,
overflow and below-threshold expectations remain separate in the batch; a
windowed image is not the all-energy photon yield. See
[detector-response accounting](../physics/detectors/detector-response.md#native-measured-bins-and-counting-acquisition)
for units, response models and reporting-bin conventions.

## Resolution and limits

Angular tiles share source spectra across pixel-centre rays. A fine pixel grid
does not imply equally fine angular scoring. Refine histories, angular tiles,
energy grids and reflection coverage before quantitative use. Changing an
acquisition seed creates another count realization from the same estimated
source; it does not create an independent transport simulation. Poisson
fluctuations exclude source-estimation uncertainty, calibration uncertainty and
external backgrounds.

For campaign observations saved alongside checkpoints, use the
[pixel-observation reference](../reference/results-and-apps.md#pixel-detector-observations).
The earlier fixed CPU example remains downloadable as a
{download}`script <../examples/detector_observations.py>` with its
{download}`cached output and provenance <../examples/data/detector_observations.json>`;
its bounded sampling is illustrative, not a convergence or calibration study.
