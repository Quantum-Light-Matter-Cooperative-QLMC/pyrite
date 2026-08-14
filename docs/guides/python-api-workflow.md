# Python API workflow

Use the Python API for scalar, in-memory simulations and programmatic case
construction. Use `pyrite run` for named profiles, resumable checkpoints,
remote execution, and the checkpoint-driven analysis apps.

## Run one scene in memory

The root package exposes the supported high-level objects. `simulate` accepts
the scene's three components as separate arguments; it does not accept a
`Scene` object directly.

```python
import pyrite as pr

beam = pr.Beam(energy_keV=30.0)
target = pr.Slab("hopg", thickness_ang=10_000.0, tilt_deg=30.0)
detector = pr.Detector()

result = pr.simulate(
    beam,
    target,
    detector,
    numerics=pr.Numerics(
        n_electrons=450,
        n_electrons_brem=100,
        convergence=pr.Convergence(n_families=4),
    ),
)
```

`Beam.energy_keV`, target thickness, and target angles must be scalar in a
single scene. Put multiple values in a `Sweep` instead. Beam energy is in keV;
target thickness is in Å; angles are in degrees.

`simulate` runs the existing typed-case Monte Carlo path and returns a
`pyrite.Result`. It does not read or write the workspace, checkpoint cache, or
profile catalog.

## Interpret the result

`Result.energy_eV` and `Result.background_energy_eV` are the photon-energy
coordinates for `spectrum` and `background`. Both arrays are intrinsic photon
density per incident electron per eV per sr. They have not been multiplied by
detector acceptance or source current.

```python
line_energy_eV = result.energy_eV
intrinsic_line = result.spectrum
continuum_energy_eV = result.background_energy_eV
intrinsic_continuum = result.background

digest = result.provenance["identity_digest"]
backend = result.provenance["backend"]
device = result.provenance["device"]
resolved_case = result.case
```

Provenance also contains the resolved scene, numerics, and PyRITE/NumPy
versions. Record the digest and software revision with any derived figure or
table.

`emission="both"` computes incoherent and coherent spectra from the same
transport. `result.spectrum` remains the incoherent array and
`result.coherent_spectrum` holds the coherent companion. With
`emission="coherent"`, both names select the coherent array. This model remains
inside the [documented validation boundary](../physics/radiation-physics/coherent-emission.md).

The high-level API currently cannot inject an external bremsstrahlung array.
`brem_source="external"` and `brem_source="none"` therefore return zeros on
the resolved continuum grid; use `brem_source="mc"` for simulated background.

## Apply detector response explicitly

Simulation output remains intrinsic even when the scene's detector has a
response model. Apply read-time response with `Detector.score`:

```python
from pyrite.detectors import Timepix3

timepix = pr.Detector(response=Timepix3(thickness_um=500.0, bias_v=100.0))
detected_line = timepix.score(
    result.energy_eV,
    result.spectrum,
    scale=1.0,
)
```

`scale` is explicit. Supply the physically appropriate solid-angle/current
conversion for the observable; `Detector.score` does not infer it from a
`Result`. Do not reuse a scale or response across instruments without checking
the [detector-response model](../physics/detectors/detector-response.md) and
[solid-angle convention](../physics/detectors/detector-solid-angle.md).

## Expand a path-addressed sweep

A `Scene` is useful as the scalar base for a Cartesian product. Axis paths can
address nested fields and indexed stack layers:

```python
scene = pr.Scene(beam, target, detector)
sweep = pr.Sweep(
    base=scene,
    axes={
        "beam.energy_keV": [30.0, 45.0, 60.0],
        "target.tilt_deg": [15.0, 30.0, 45.0],
    },
)

expanded = sweep.expand()           # ordered (label, Scene) pairs
cases = sweep.cases(pr.Numerics())  # typed Monte Carlo Case records
```

Axis insertion order defines Cartesian-product order. Expanded cases receive
deterministic one-based seeds in that order. Invalid field names and
out-of-range indexes fail when the `Sweep` is constructed.

`Sweep.expand()` and `Sweep.cases()` do not create checkpoints. For a large or
resumable sweep, express the ranges in a profile and use `pyrite run`; for
direct low-level execution, `pyrite.montecarlo.run_cases` is the supported
kernel entry point.

## Separate physical and execution controls

- `Scene` owns physical configuration and model switches: beam, target,
  detector, emission mode, X-ray dispersion, and background source.
- `Numerics` owns electron counts and execution controls. Electron counts
  affect the computed content identity; backend, chunk, and transport-core
  choices select execution policy.
- `Numerics.convergence` owns result-affecting truncation and quadrature
  controls such as reflection families and mosaic nodes.
- `Analysis` is currently a presentation-control value object and a bridge for
  checkpoint-analysis compatibility. It is not accepted by `simulate`.

Select the array backend before importing PyRITE, for example:

<!-- verify: skip (illustrative script invocation, not a pyrite/pyrite-dev command) -->
```bash
PYRITE_MC_BACKEND=cpu uv run python my_simulation.py
```

An explicit `Numerics.backend` validates the already-active backend; it does
not switch an imported process to another device.

## Choose persistence deliberately

`Result` is an in-memory return type and has no supported save/load method.
Resumable campaigns store result mappings in component checkpoints. Current
`.pkl` paths contain versioned HDF5 data; older pickle generations remain
readable. See [Working with results](working-with-results.md), the
[result schema](../repo-design/storage/result-schema.md), and the
[supported API reference](../api.md).

No MCPL or other particle-interchange adapter is currently implemented.
