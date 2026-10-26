# Issue 703: detector/observation ownership and compatibility

This contract continues the shared Scene/Observation resolution on #695, the
public namespace manifest on #699, numerical policy ownership on #702, physical
geometry on #701 and scientific-result storage on #707.

| Domain | Canonical new-code concept | Supported compatibility |
| --- | --- | --- |
| `pyrite.detectors` | `AngularDetector`, `PlanarDetector`, `PixelGrid`, response classes/protocols | scalar `Detector`, `Timepix3`, `EagleXO` |
| `pyrite.geometry` | `Filter`, `FilterPlate` | `pyrite.instrument.FilterPlate`, `pyrite.detectors.FilterPlate` |
| `pyrite.config` | `AngularSampling`, `Timepix3ResponseNumerics` | `PixelScorer`, original `Timepix3(dE_mc=..., ...)` |
| `pyrite.observations` | `Observation`, `Acquisition` | `ResolvedObservation`, `ObservationPlan`, `PixelSampling`, `SweepObservation`, `NativeAcquisition` |
| `pyrite.results` | `PixelCountResult`, `AcquisitionBatch` | `pyrite.observations.AcquisitionBatch` |

`Observation` is the immutable user-owned request; `ObservationPlan` and
`PixelSampling` are derived work products, *not* preferred constructors for new
user workflows. Those existing advanced-stable types remain available under
#699's compatibility rules rather than being removed or marked private now.

Physical `PlanarDetector` dataclass state is only pose, size and pixel layout.
Its optional `response` and `scorer` constructor parameters are legacy
compatibility inputs; neither is physical canonical dataclass state. A run can
bind independent response and angular sampling to the same geometry without
modifying the input Scene. `PixelGrid.shape` is the number of physical sensor
pixels, whereas `AngularSampling.angular_shape` is the source-direction
approximation. Changing the latter alters the true-spatial identity if its
evaluated approximation changes; response/acquisition only alter their own
identity layers and preserve reusable source/true-spatial factors.

## Example

```python
from pyrite.config import AngularSampling, Timepix3ResponseNumerics
from pyrite.detectors import IdealPhotonCounter, Timepix3Response
from pyrite.observations import Observation, Acquisition

samples = AngularSampling(angular_shape=(2, 2), reconstruction="nearest_tile")
response = Timepix3Response(
    thickness_um=300,
    bias_v=100,
    numerics=Timepix3ResponseNumerics(dE_mc=200, dE_out=100, n_mc=128, seed=7),
)
request = Observation(
    detector="detector",  # physically registered Scene.detector
    sampling=samples,
    response=response,
    acquisition=Acquisition.native(exposure_s=1.0),
)
```

This example assumes a pixelated `PlanarDetector` registered on `Scene`. A
virtual scalar request may instead use `AngularDetector(...)`, without a
pixel sampler or pixel acquisition. `simulate(scene, observation=request)` and
observation production use the shared #695 binding path.

`Timepix3ResponseNumerics` distinguishes construction of the response matrix
from physical sensor thickness/bias and from later acquisition exposure/binning.
A corresponding nested `[profiles.*.detectors.*.response.numerics]` table is
accepted by the catalog parser. Flat numeric response fields and existing CLI
`--timepix-*` options still resolve to the same internal `Timepix3` values.
The grouped policy is an `InitVar` constructor input: it does not become a new
serialized field, does not change the response payload's type/config, and
preserves seeded matrix construction. Conflicting flat/grouped values fail.

`DetectorResponse.score()` accepts a true-energy grid in eV and a spectral
density array (last dimension matches the energy grid), with the same shape
on output. `NativeBinnedResponse.native_score()` returns measured **event mass**,
with native measured-energy edges in eV, keeping the input leading axes and
count normalization. Native response capability is optional. There is no
universal event-digitizer interface implied by these methods; transporting
photon histories and depositing energies is a separate scientific pathway.

Profile and CLI forms use their existing physical-detector geometry, scoring,
response and acquisition bindings. There is deliberately no removal of the
legacy profile keys or new event-transport scoring pipeline in this issue.
