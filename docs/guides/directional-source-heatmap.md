# Detector-free directional source heatmaps

```bash
pyrite app angular-heatmap launch
pyrite app angular-heatmap launch --smoke
pyrite app angular-heatmap export
```

The standalone app computes a bounded initial example: 30 keV point electrons,
20 Å silicon, target tilts 30° and 45°, eight incident histories per tilt,
72 periodic observation angles and 129 photon energies from 100 to 5000 eV.
It needs no detector, campaign or checkpoint. Edit the controls and submit
**Run source maps** to calculate another scene. Selecting/comparing computed
tilts, changing mean current, or choosing linear/log color only redraws cached
source arrays. Export writes the default example into static HTML; static HTML
cannot run new transports. The four-entry in-memory calculation cache is local
to the running app process.

Only the existing **incoherent PXR/CBS total** is scored, including its PXR–CBS
interference. Bremsstrahlung, characteristic, cross-electron coherent,
channeling and transition radiation are excluded from the map. Electron
transport retains the existing interaction policy. Photons retain target
escape/self-absorption; there is no downstream filter, aperture, response,
acquisition, or angular acceptance integration.

## Coordinates and units

The nominal incoming beam defines lab +z. For observation-plane azimuth φ,
its transverse basis is `(cos φ, sin φ, 0)` and the direction at signed angle θ
is `(sin θ cos φ, sin θ sin φ, cos θ)`. Default φ = 0 gives the x–z plane:
0° is forward, +90° is +x, −90° is −x and ±180° is backward. The target pose
rotates these lab vectors into the sample frame using the existing target
convention; target angles resolve to the nearest half degree. A map is a
**plane slice**, not an azimuth-integrated distribution.

Angles are periodic point samples on [−180°, +180°), without a repeated endpoint.
The plot closes the circle by duplicating the first sample for drawing only.
Sample-face grazing directions (`abs(n_sample,z) <= 1e-10`) are unsupported and
masked gray, not silently reported as zero. In log color, zero bins are also
gray; linear color distinguishes true zero from masked values.

The reusable result stores photons per incident electron per eV per sr. Its
`rate(current_na)` scales by the incident electron rate `I[nA] * 1e-9 / |e|`,
yielding **ph/s/eV/sr** with no solid-angle factor. Omitting current uses the
beam mean current: `Q[pC] * f[Hz] / 1000` nA. The initial beam has 1 pC at 5 kHz,
hence 5 nA. Supported bins vanish at zero current. This conversion assumes
independent incident electrons and applies to this incoherent component only.
See [spectral observables](../physics/radiation-physics/spectral-observables.md)
and the independently verified
[normalization claim](../validation/radiation-physics/directional-source-rate-normalization.md).

## Reusable source calculation

```python
import numpy as np
from pyrite import Beam, PointSpot, PointBunch, Slab, Numerics, directional_source

beam = Beam(
    30.0, transverse=PointSpot(), longitudinal=PointBunch(), bunch_charge_pc=1.0, rep_rate_hz=5000.0
)
target = Slab("silicon", 20.0, tilt_deg=30.0, footprint=None)
source = directional_source(
    beam,
    target,
    angle_deg=np.linspace(-180, 180, 72, endpoint=False),
    energy_eV=np.linspace(100, 5000, 129),
    numerics=Numerics(cxr_trials=8),
    seed=1,
    direction_batch=16,
)
rate = source.rate()  # shape (angle, energy), ph/s/eV/sr
rate_at_10_na = source.rate(10.0)  # rescales without transport
```

The provisional API also accepts flat stacks and finite rectangular footprints;
the app uses infinite flat slabs. Grooves, direction-targeted longitudinal
policies and adaptive precision are unsupported and rejected. Existing target
geometry restrictions still apply. It transports once per scalar target/tilt,
then evaluates bounded direction batches on the same segments. `brem_trials`
is replaced by `cxr_trials` to avoid an unnecessary second population; only the
CXR adapter is evaluated. Result metadata records the case, seed, resolved
numerics, plane azimuth, energy-grid choice, package/model/table provenance and
source identity. Component identity and independent energy coordinates leave
room for other future source components without assuming additive semantics.

## Resolution and model limits

Supply an explicit common photon-energy **point grid** for the entire circle.
It must increase, start at or above 10 eV, and end at or below the incident
electron kinetic energy. A narrow single-direction automatic line window is
never substituted. Out-of-window resonances can still be omitted by the
existing scorer, including their tails; extend bounds and refine energy and
angle spacing before trusting a peak. Low sample counts are noisy. The app
caps the reflection set at four; use `Numerics(cxr_reflections=...)` through
the API for a reflection-convergence study. Other source components may need
different native energy grids and different combination rules.

The thin initial target deliberately broadens features. A bounded CPU check
for its 30° map doubles 72→144 angles and 129→257 energy points on the same
100–5000 eV window. With seed 1 and eight histories, common-node densities agree
exactly, energy-integral L1 changes by 0.18%, peak density by 0.84%, and a
plane-slice quadrature by 0.74% (unsupported grazing samples omitted). That last
quantity is a convergence diagnostic, not a solid-angle-integrated photon
rate. Doubling incident histories to 16 changes the density L1 by 2.24%,
consistent with retaining per-incident-electron normalization. Regression:
`tests/notebooks/test_angular_heatmap_app.py::test_default_grid_refinement_and_history_normalization`.
These checks cover this broad thin-film example, not arbitrary thick targets
or sharply resolved resonances. The finite plotted window is not a total yield.

The quick local app allows up to four tilts, 256 histories, 720 angles,
2049 energies and 2000 Å thickness, with a combined four-million
history–angle–energy scoring budget. Tilts must remain distinct after the
shared half-degree pose resolution. Large combinations remain expensive;
route heavy sweeps and convergence studies through `pyrite remote` using a
small script around the reusable API. The existing kinematic PXR/CBS and escape
model validity limits apply unchanged.
